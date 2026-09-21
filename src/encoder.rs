//! The ModernBERT-family encoder (ModernBERT-large, mmBERT-base). See ADR-0001.
//!
//! Mirrors `transformers.models.modernbert.modeling_modernbert`: token embeddings + LayerNorm,
//! pre-norm layers alternating full and sliding-window attention with per-kind RoPE, GeGLU MLP,
//! a final LayerNorm, and no biases anywhere.

use crate::config::{EncoderConfig, LayerType};
use crate::error::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{layer_norm_no_bias, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};

/// Additive value for masked attention scores. Large enough to vanish under softmax in every
/// supported dtype, finite so that a fully masked (padding) query row yields a uniform
/// distribution instead of NaN.
pub(crate) const MASK_VALUE: f32 = -1e4;

/// Precomputed `cos`/`sin` tables for one RoPE theta, up to `max_position_embeddings`.
#[derive(Debug, Clone)]
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(
        theta: f64,
        head_dim: usize,
        max_len: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        // HF computes inv_freq and the outer product in f32; we do the same before casting.
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut freqs = Vec::with_capacity(max_len * half);
        for pos in 0..max_len {
            for f in &inv_freq {
                freqs.push(pos as f32 * f);
            }
        }
        let freqs = Tensor::from_vec(freqs, (max_len, half), dev)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, xs: &Tensor, seq_len: usize) -> Result<Tensor> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        Ok(candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)?)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    wqkv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            wqkv: linear_no_bias(d, 3 * d, vb.pp("Wqkv"))?,
            wo: linear_no_bias(d, d, vb.pp("Wo"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `xs`: (B, L, d); `mask`: (B, 1, L, L) additive; returns (B, L, d).
    fn forward(&self, xs: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let qkv = self
            .wqkv
            .forward(xs)?
            .reshape((b, l, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?; // (3, B, H, L, D)
        let q = rotary.apply(&qkv.get(0)?, l)?;
        let k = rotary.apply(&qkv.get(1)?, l)?;
        let v = qkv.get(2)?.contiguous()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out =
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        Ok(self.wo.forward(&out)?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    wi: Linear,
    wo: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wi: linear_no_bias(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("Wi"))?,
            wo: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?,
            intermediate: cfg.intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.wi.forward(xs)?;
        let input = h.narrow(D::Minus1, 0, self.intermediate)?;
        let gate = h.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        Ok(self.wo.forward(&(input.gelu_erf()? * gate)?)?)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    /// `None` on layer 0, where HF uses `nn.Identity`.
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
    kind: LayerType,
}

impl Layer {
    fn load(cfg: &EncoderConfig, index: usize, vb: VarBuilder) -> Result<Self> {
        let attn_norm = if index == 0 {
            None
        } else {
            Some(layer_norm_no_bias(
                cfg.hidden_size,
                cfg.norm_eps(),
                vb.pp("attn_norm"),
            )?)
        };
        Ok(Self {
            attn_norm,
            attn: Attention::load(cfg, vb.pp("attn"))?,
            mlp_norm: layer_norm_no_bias(cfg.hidden_size, cfg.norm_eps(), vb.pp("mlp_norm"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            kind: cfg.layer_type(index),
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(norm) => norm.forward(xs)?,
            None => xs.clone(),
        };
        let xs = (xs + self.attn.forward(&normed, rotary, mask)?)?;
        let xs = (&xs + self.mlp.forward(&self.mlp_norm.forward(&xs)?)?)?;
        Ok(xs)
    }
}

/// The encoder: `forward` returns the final-normed hidden states, `(B, L, hidden_size)`.
#[derive(Debug, Clone)]
pub struct Encoder {
    embeddings: Embedding,
    embeddings_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    global_rotary: Rotary,
    local_rotary: Rotary,
    half_window: usize,
    pub config: EncoderConfig,
}

impl Encoder {
    /// Loads from a `VarBuilder` positioned at the encoder's prefix (`encoder` in a Laya checkpoint).
    pub fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let (dtype, dev) = (vb.dtype(), vb.device().clone());
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(cfg, i, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings: candle_nn::embedding(
                cfg.vocab_size,
                d,
                vb.pp("embeddings.tok_embeddings"),
            )?,
            embeddings_norm: layer_norm_no_bias(d, cfg.norm_eps(), vb.pp("embeddings.norm"))?,
            layers,
            final_norm: layer_norm_no_bias(d, cfg.norm_eps(), vb.pp("final_norm"))?,
            global_rotary: Rotary::new(
                cfg.global_rope_theta()?,
                cfg.head_dim(),
                cfg.max_position_embeddings,
                dtype,
                &dev,
            )?,
            local_rotary: Rotary::new(
                cfg.local_rope_theta()?,
                cfg.head_dim(),
                cfg.max_position_embeddings,
                dtype,
                &dev,
            )?,
            half_window: cfg.local_attention / 2,
            config: cfg.clone(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.embeddings.embeddings().dtype()
    }

    pub fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    /// `input_ids`: (B, L) u32; `attention_mask`: (B, L) with 1 for real tokens, 0 for padding.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_, l) = input_ids.dims2()?;
        let (global_mask, local_mask) = self.attention_masks(attention_mask, l)?;
        let mut xs = self
            .embeddings_norm
            .forward(&self.embeddings.forward(input_ids)?)?;
        for layer in &self.layers {
            xs = match layer.kind {
                LayerType::FullAttention => {
                    layer.forward(&xs, &self.global_rotary, &global_mask)?
                }
                LayerType::SlidingAttention => {
                    layer.forward(&xs, &self.local_rotary, &local_mask)?
                }
            };
        }
        Ok(self.final_norm.forward(&xs)?)
    }

    /// Additive `(B, 1, L, L)` masks for global layers (padding only) and sliding layers
    /// (padding or outside the window).
    fn attention_masks(&self, attention_mask: &Tensor, l: usize) -> Result<(Tensor, Tensor)> {
        let dev = attention_mask.device();
        let dtype = self.dtype();
        let b = attention_mask.dim(0)?;
        // Key padding: (B, 1, 1, L) broadcast over queries.
        let key_ok = attention_mask.to_dtype(DType::U8)?.reshape((b, 1, 1, l))?;
        let window: Vec<u8> = (0..l)
            .flat_map(|q| (0..l).map(move |k| u8::from(q.abs_diff(k) <= self.half_window)))
            .collect();
        let in_window = Tensor::from_vec(window, (1, 1, l, l), dev)?;
        let allowed_global = key_ok.broadcast_as((b, 1, l, l))?.contiguous()?;
        let allowed_local = allowed_global.broadcast_mul(&in_window)?;
        let zero = Tensor::zeros((b, 1, l, l), dtype, dev)?;
        let masked = Tensor::full(MASK_VALUE, (b, 1, l, l), dev)?.to_dtype(dtype)?;
        Ok((
            allowed_global.where_cond(&zero, &masked)?,
            allowed_local.where_cond(&zero, &masked)?,
        ))
    }
}
