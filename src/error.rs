use thiserror::Error;

/// Everything that can go wrong while loading a checkpoint or answering questions.
#[derive(Debug, Error)]
pub enum Error {
    /// A question's options (with their `[MASK]` markers) do not fit in the head budget.
    #[error("question {question_id:?}: options do not fit in head_max_len={head_max_len} tokens")]
    OptionsExceedHeadBudget {
        question_id: String,
        head_max_len: usize,
    },

    /// A Choice or Score question with fewer than two options cannot be answered.
    #[error("question {question_id:?}: needs at least two options, got {options}")]
    TooFewOptions { question_id: String, options: usize },

    /// A requested context budget exceeds what the encoder supports.
    #[error(
        "max_len={max_len} exceeds the encoder's max_position_embeddings={max_position_embeddings}"
    )]
    ContextBudgetTooLarge {
        max_len: usize,
        max_position_embeddings: usize,
    },

    /// A batch request failed validation; `request` is the index into the batch.
    #[error("request {request}: {source}")]
    Request {
        request: usize,
        #[source]
        source: Box<Error>,
    },

    #[error("checkpoint config: {0}")]
    Config(String),

    #[error("checkpoint is missing {0}")]
    MissingFile(std::path::PathBuf),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "hub")]
    #[error(transparent)]
    Hub(#[from] hf_hub::HFError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}
