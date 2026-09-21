//! Fetching checkpoints from the Hugging Face Hub into the standard HF cache (shared with the
//! Python install). Enabled by the `hub` feature.

use crate::agent::Agent;
use crate::checkpoint::{Checkpoint, BUNDLE_REPO};
use crate::error::Result;
use candle_core::{DType, Device};
use hf_hub::HFClient;
use std::path::PathBuf;

/// The files one checkpoint needs, relative to its root.
const FILES: [&str; 5] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
];

/// Downloads (or finds in the cache) one checkpoint of `repo` and returns its directory.
/// `subfolder` selects a bundled checkpoint; `None` means the repo root. Honors `HF_TOKEN`
/// and `HF_HUB_CACHE`.
pub fn fetch(repo: &str, subfolder: Option<&str>) -> Result<PathBuf> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| crate::Error::Config(format!("hub repo {repo:?} must be owner/name")))?;
    let mut builder = HFClient::builder();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            builder = builder.token(token);
        }
    }
    let client = builder.build_sync()?;
    let model = client.model(owner, name);
    let mut root = None;
    for file in FILES {
        let path = match subfolder {
            Some(sub) => format!("{sub}/{file}"),
            None => file.to_owned(),
        };
        let local = model.download_file().filename(path).send()?;
        if root.is_none() {
            // `<snapshot>/<subfolder>/rl_agent_config.json` -> the checkpoint directory.
            root = local.parent().map(PathBuf::from);
        }
    }
    root.ok_or_else(|| crate::Error::Config("no files downloaded".into()))
}

impl Agent {
    /// Loads a checkpoint from the Hub bundle repo, downloading only what that checkpoint needs.
    pub fn from_hub(checkpoint: Checkpoint, device: &Device, dtype: Option<DType>) -> Result<Self> {
        Self::from_hub_repo(BUNDLE_REPO, checkpoint.subfolder(), device, dtype)
    }

    /// Loads a checkpoint from any Hub repo laid out like the official ones.
    pub fn from_hub_repo(
        repo: &str,
        subfolder: Option<&str>,
        device: &Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Self::from_dir(fetch(repo, subfolder)?, device, dtype)
    }
}
