use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// A checkpoint's tokenizer plus the special-token ids the sequence builder needs.
///
/// Special ids come from the tokenizer (`tokenizer_config.json` names resolved through
/// `tokenizer.json`), which is what the Python reference uses; the encoder config's ids are
/// not trusted (the multilingual one disagrees with its own tokenizer).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub cls_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
    pub mask_id: u32,
    /// The mask token's text, scrubbed from user-supplied text before tokenizing.
    pub mask_token: String,
}

#[derive(Deserialize, Default)]
struct TokenizerConfig {
    #[serde(default)]
    cls_token: Option<TokenName>,
    #[serde(default)]
    sep_token: Option<TokenName>,
    #[serde(default)]
    pad_token: Option<TokenName>,
    #[serde(default)]
    mask_token: Option<TokenName>,
}

/// Older tokenizer configs spell special tokens as `{"content": "..."}` objects.
#[derive(Deserialize)]
#[serde(untagged)]
enum TokenName {
    Text(String),
    Object { content: String },
}

impl TokenName {
    fn as_str(&self) -> &str {
        match self {
            TokenName::Text(s) | TokenName::Object { content: s } => s,
        }
    }
}

impl Tokenizer {
    /// Loads `<dir>/tokenizer.json` and, if present, `<dir>/tokenizer_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let json = dir.join("tokenizer.json");
        if !json.exists() {
            return Err(Error::MissingFile(json));
        }
        let inner = tokenizers::Tokenizer::from_file(&json)?;
        let cfg: TokenizerConfig = match std::fs::read_to_string(dir.join("tokenizer_config.json"))
        {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|e| Error::Tokenizer(format!("tokenizer_config.json: {e}")))?,
            Err(_) => TokenizerConfig::default(),
        };
        let resolve = |name: &Option<TokenName>, default: &str| -> Result<(u32, String)> {
            let name = name.as_ref().map(TokenName::as_str).unwrap_or(default);
            inner
                .token_to_id(name)
                .map(|id| (id, name.to_owned()))
                .ok_or_else(|| {
                    Error::Tokenizer(format!("special token {name:?} is not in the vocabulary"))
                })
        };
        let (cls_id, _) = resolve(&cfg.cls_token, "[CLS]")?;
        let (sep_id, _) = resolve(&cfg.sep_token, "[SEP]")?;
        let (pad_id, _) = resolve(&cfg.pad_token, "[PAD]")?;
        let (mask_id, mask_token) = resolve(&cfg.mask_token, "[MASK]")?;
        Ok(Self {
            inner,
            cls_id,
            sep_id,
            pad_id,
            mask_id,
            mask_token,
        })
    }

    /// Token ids for `text` without special tokens, with any literal mask token replaced by a
    /// space first (as the reference does for instructions, options, and state).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let scrubbed = text.replace(&self.mask_token, " ");
        Ok(self
            .inner
            .encode(scrubbed.as_str(), false)?
            .get_ids()
            .to_vec())
    }
}
