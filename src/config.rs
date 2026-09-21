use crate::error::{Error, Result};
use crate::question::Kind;
use indexmap::IndexMap;
use serde::Deserialize;
use std::path::Path;

/// `rl_agent_config.json`: budgets, calibration temperatures, and head shape for one checkpoint.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentConfig {
    /// Context budget: total tokens per question sequence.
    pub max_len: usize,
    /// Head budget: tokens shared by the instructions and all option texts.
    pub head_max_len: usize,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    /// Per-kind temperature, indexed by [`Kind`].
    #[serde(default = "default_temperature")]
    pub temperature: [f32; 3],
    /// Per temperature-bucket overrides, keyed like `"choice:3-5"`.
    #[serde(default)]
    pub temperature_by_options: IndexMap<String, f32>,
    /// Named non-default actions; the act head has `act_costs.len() + 1` outputs.
    #[serde(default)]
    pub act_costs: IndexMap<String, f32>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub encoder: Option<String>,
}

fn default_head_layers() -> usize {
    2
}

fn default_temperature() -> [f32; 3] {
    [1.0; 3]
}

impl AgentConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|_| Error::MissingFile(path.to_owned()))?;
        serde_json::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
    }

    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }

    /// Temperature for a question of `kind` with `k` options.
    pub fn temperature_for(&self, kind: Kind, k: usize) -> f32 {
        self.temperature_by_options
            .get(&temp_bucket(kind, k))
            .copied()
            .unwrap_or(self.temperature[kind as usize])
    }
}

/// Key for per-cardinality temperature lookup: a 2-option noul and a 20-option choice need
/// different scaling.
pub fn temp_bucket(kind: Kind, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{}", kind.name(), size)
}

/// `encoder/config.json` in transformers-v5 shape, reduced to what the encoder needs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
    pub pad_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    /// Absent on the English checkpoint; the tokenizer's `[MASK]` id is used instead.
    #[serde(default)]
    pub mask_token_id: Option<u32>,
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    /// Explicit per-layer kinds (`full_attention` / `sliding_attention`); falls back to
    /// `global_attn_every_n_layers` when absent.
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    /// Sliding window width; each token sees `local_attention / 2` neighbours on either side.
    pub local_attention: usize,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
}

fn default_global_every() -> usize {
    3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct RopeParameters {
    full_attention: RopeTheta,
    sliding_attention: RopeTheta,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct RopeTheta {
    rope_theta: f64,
}

impl EncoderConfig {
    /// Checkpoints carry both `norm_eps` and the older `layer_norm_eps`; `norm_eps` wins.
    pub fn norm_eps(&self) -> f64 {
        self.norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|_| Error::MissingFile(path.to_owned()))?;
        let cfg: Self = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        cfg.global_rope_theta()?;
        cfg.local_rope_theta()?;
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn layer_type(&self, layer: usize) -> LayerType {
        match &self.layer_types {
            Some(types) => types[layer],
            None if layer % self.global_attn_every_n_layers == 0 => LayerType::FullAttention,
            None => LayerType::SlidingAttention,
        }
    }

    pub fn global_rope_theta(&self) -> Result<f64> {
        self.rope_parameters
            .as_ref()
            .map(|r| r.full_attention.rope_theta)
            .or(self.global_rope_theta)
            .ok_or_else(|| Error::Config("missing rope theta for full attention".into()))
    }

    pub fn local_rope_theta(&self) -> Result<f64> {
        self.rope_parameters
            .as_ref()
            .map(|r| r.sliding_attention.rope_theta)
            .or(self.local_rope_theta)
            .ok_or_else(|| Error::Config("missing rope theta for sliding attention".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_buckets() {
        assert_eq!(temp_bucket(Kind::Noul, 2), "noul:2");
        assert_eq!(temp_bucket(Kind::Choice, 4), "choice:3-5");
        assert_eq!(temp_bucket(Kind::Choice, 10), "choice:6-10");
        assert_eq!(temp_bucket(Kind::Score, 11), "score:11+");
    }

    #[test]
    fn agent_config_lookup_prefers_bucket() {
        let cfg: AgentConfig = serde_json::from_str(
            r#"{"max_len": 512, "head_max_len": 192, "temperature": [1.5, 1.2, 1.9],
                "temperature_by_options": {"choice:2": 1.1}, "act_costs": {"escalate": 0.5}}"#,
        )
        .unwrap();
        assert_eq!(cfg.temperature_for(Kind::Choice, 2), 1.1);
        assert_eq!(cfg.temperature_for(Kind::Choice, 3), 1.5);
        assert_eq!(cfg.temperature_for(Kind::Noul, 2), 1.9);
        assert_eq!(cfg.n_act(), 2);
        assert_eq!(cfg.head_layers, 2);
    }

    #[test]
    fn encoder_config_v5_and_legacy_rope() {
        let v5: EncoderConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2,
                "intermediate_size": 96, "max_position_embeddings": 128, "pad_token_id": 0,
                "cls_token_id": 1, "sep_token_id": 2, "local_attention": 8,
                "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                                    "sliding_attention": {"rope_theta": 10000.0}}}"#,
        )
        .unwrap();
        assert_eq!(v5.global_rope_theta().unwrap(), 160000.0);
        assert_eq!(v5.local_rope_theta().unwrap(), 10000.0);
        assert_eq!(v5.layer_type(1), LayerType::SlidingAttention);
        assert_eq!(v5.layer_type(3), LayerType::FullAttention);
        assert_eq!(v5.norm_eps(), 1e-5);
        assert_eq!(v5.mask_token_id, None);

        let legacy: EncoderConfig = serde_json::from_str(
            r#"{"vocab_size": 8, "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2,
                "intermediate_size": 96, "max_position_embeddings": 128, "pad_token_id": 0,
                "cls_token_id": 1, "sep_token_id": 2, "local_attention": 8, "layer_norm_eps": 1e-6,
                "global_rope_theta": 160000.0, "local_rope_theta": 10000.0}"#,
        )
        .unwrap();
        assert_eq!(legacy.layer_type(0), LayerType::FullAttention);
        assert_eq!(legacy.layer_type(2), LayerType::SlidingAttention);
        assert_eq!(legacy.norm_eps(), 1e-6);
        assert_eq!(legacy.local_rope_theta().unwrap(), 10000.0);
    }
}
