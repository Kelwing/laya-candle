//! The decision head trained on top of the encoder: kind embedding, post-encoder transformer
//! layers, the marker scorer, and the act head. Mirrors `DecisionModel` in the Python package.

use crate::config::AgentConfig;
use crate::encoder::MASK_VALUE;
use crate::error::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{layer_norm, linear, Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder};

/// One `torch.nn.TransformerEncoderLayer(norm_first=True)`: pre-norm self-attention and a ReLU
/// feed-forward block, both with biases.
#[derive(Debug, Clone)]
struct HeadLayer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    fn load(d: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = (d / 64).max(1);
        let ln = LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        Ok(Self {
            in_proj: Linear::new(
                vb.get((3 * d, d), "self_attn.in_proj_weight")?,
                Some(vb.get(3 * d, "self_attn.in_proj_bias")?),
            ),
            out_proj: linear(d, d, vb.pp("self_attn.out_proj"))?,
            linear1: linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: linear(4 * d, d, vb.pp("linear2"))?,
            norm1: layer_norm(d, ln, vb.pp("norm1"))?,
            norm2: layer_norm(d, ln, vb.pp("norm2"))?,
            num_heads,
            head_dim: d / num_heads,
        })
    }

    /// `xs`: (B, L, d); `key_mask`: (B, 1, 1, L) additive mask over keys.
    fn forward(&self, xs: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let normed = self.norm1.forward(xs)?;
        let qkv = self.in_proj.forward(&normed)?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, i * d, d)?
                .reshape((b, l, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(key_mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = probs.matmul(&v)?.transpose(1, 2)?.reshape((b, l, d))?;
        let xs = (xs + self.out_proj.forward(&attn)?)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&xs)?)?.relu()?)?;
        Ok((xs + ff)?)
    }
}

/// Raw model outputs for one batch of sequences.
#[derive(Debug, Clone)]
pub struct HeadOutput {
    /// `(B, K)` option logits, `MASK_VALUE` where a sequence has fewer than `K` options.
    pub logits: Tensor,
    /// `(B, n_act)` act-head logits.
    pub act_logits: Tensor,
}

#[derive(Debug, Clone)]
pub struct DecisionHead {
    type_emb: Embedding,
    layers: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer1: Linear,
    scorer2: Linear,
    act1: Linear,
    act2: Linear,
}

impl DecisionHead {
    /// Loads from a `VarBuilder` at the checkpoint root (keys `type_emb`, `head`, `scorer`, `act_head`).
    pub fn load(cfg: &AgentConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let d = hidden_size;
        let ln = LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let layers = (0..cfg.head_layers)
            .map(|i| HeadLayer::load(d, vb.pp(format!("head.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            type_emb: candle_nn::embedding(3, d, vb.pp("type_emb"))?,
            layers,
            scorer_norm: layer_norm(d, ln, vb.pp("scorer.0"))?,
            scorer1: linear(d, d, vb.pp("scorer.1"))?,
            scorer2: linear(d, 1, vb.pp("scorer.3"))?,
            act1: linear(d + 4, 256, vb.pp("act_head.0"))?,
            act2: linear(256, cfg.n_act(), vb.pp("act_head.2"))?,
        })
    }

    /// `hidden`: encoder output `(B, L, d)`; `attention_mask`: `(B, L)` 1/0; `marker_pos`: `(B, K)`
    /// u32 marker positions (0 where unused); `marker_mask`: `(B, K)` u8 1 for real markers;
    /// `kinds`: `(B,)` u32 question kinds.
    pub fn forward(
        &self,
        hidden: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        kinds: &Tensor,
    ) -> Result<HeadOutput> {
        let (b, l, d) = hidden.dims3()?;
        let dtype = hidden.dtype();
        let dev = hidden.device();
        let mut h = hidden.broadcast_add(&self.type_emb.forward(kinds)?.unsqueeze(1)?)?;
        let key_mask = attention_mask
            .to_dtype(DType::U8)?
            .reshape((b, 1, 1, l))?
            .where_cond(
                &Tensor::zeros((b, 1, 1, l), dtype, dev)?,
                &Tensor::full(MASK_VALUE, (b, 1, 1, l), dev)?.to_dtype(dtype)?,
            )?;
        for layer in &self.layers {
            h = layer.forward(&h, &key_mask)?;
        }

        // Gather the marker states: (B, K, d).
        let k = marker_pos.dim(1)?;
        let idx = marker_pos
            .unsqueeze(2)?
            .broadcast_as((b, k, d))?
            .contiguous()?;
        let m = h.gather(&idx, 1)?;
        let s = self.scorer2.forward(
            &self
                .scorer1
                .forward(&self.scorer_norm.forward(&m)?)?
                .gelu_erf()?,
        )?;
        let logits = s.squeeze(D::Minus1)?.to_dtype(DType::F32)?;
        let marker_mask = marker_mask.to_dtype(DType::U8)?;
        let logits = marker_mask.where_cond(&logits, &Tensor::full(MASK_VALUE, (b, k), dev)?)?;

        // Act head: pooled [CLS] state + summary features of the (detached) answer distribution.
        let p = candle_nn::ops::softmax_last_dim(&logits)?;
        let count = marker_mask
            .to_dtype(DType::F32)?
            .sum_keepdim(1)?
            .clamp(2.0, f64::MAX)?;
        let ent = (p.clone() * p.clamp(1e-9, f64::MAX)?.log()?)?
            .sum_keepdim(1)?
            .neg()?
            .div(&count.log()?)?;
        let top2 = top2(&p)?;
        let top1 = top2.narrow(1, 0, 1)?;
        let margin = (&top1 - top2.narrow(1, 1, 1)?)?;
        let feats = Tensor::cat(&[&top1, &margin, &ent, &(count / 255.0)?], 1)?;
        let pooled = h.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
        let act_in = Tensor::cat(&[&pooled, &feats], 1)?.to_dtype(dtype)?;
        let act_logits = self
            .act2
            .forward(&self.act1.forward(&act_in)?.gelu_erf()?)?
            .to_dtype(DType::F32)?;
        Ok(HeadOutput { logits, act_logits })
    }
}

/// The two largest values per row of a `(B, K)` f32 tensor, as `(B, 2)`.
fn top2(p: &Tensor) -> Result<Tensor> {
    let rows: Vec<Vec<f32>> = p.to_vec2()?;
    let mut out = Vec::with_capacity(rows.len() * 2);
    for row in rows {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &x in &row {
            if x > a {
                b = a;
                a = x;
            } else if x > b {
                b = x;
            }
        }
        out.push(a);
        out.push(if b.is_finite() { b } else { 0.0 });
    }
    Ok(Tensor::from_vec(out, (p.dim(0)?, 2), p.device())?)
}
