//! Pure-Rust runtime for the Laya family of calibrated decision models.
//!
//! See `CONTEXT.md` in the repository for the vocabulary used throughout this crate.
//!
//! # Divergences from the Python reference
//!
//! - Choice and Score questions with fewer than two options are rejected with
//!   [`Error::TooFewOptions`]; the Python package answers them with a meaningless softmax.
//! - `RouteDecision::repo` is always `repo/subfolder`; the Python router leaks a raw
//!   `(repo, subfolder)` tuple when it matches a typed-decisions workflow.

pub mod config;
pub mod error;
pub mod pyjson;
pub mod question;
pub mod state;

pub use config::{AgentConfig, EncoderConfig};
pub use error::{Error, Result};
pub use question::{ChoiceCriteria, Instructions, Kind, NoulCriteria, Question, Questions};
pub use state::State;
pub mod sequence;
pub mod tokenizer;

pub use sequence::{build_sequence, Budgets, Sequence};
pub use tokenizer::Tokenizer;
pub mod encoder;

pub use encoder::Encoder;
pub mod agent;
pub mod answer;
pub mod head;

pub use agent::{Agent, BatchOptions, PredictOptions, Request, MODEL_NAME};
pub use answer::{Action, Answer, Response, Usage};
pub mod checkpoint;
#[cfg(feature = "hub")]
pub mod hub;

pub use checkpoint::{Checkpoint, BUNDLE_REPO};
pub mod lang;
pub mod router;

pub use lang::{analyse, Detection};
pub use router::{RouteDecision, RouteOverrides, Router, Source};
