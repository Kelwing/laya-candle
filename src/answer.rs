//! Typed answers and the response envelope, serialized in the Python package's wire shape.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The act head's verdict for one answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// Probability that the answer should be acted on rather than escalated.
    pub act_probability: f64,
}

/// The answer to one question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        /// The most probable option key.
        choice: String,
        /// Probability per option key, in criteria order.
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        action: Action,
    },
    Score {
        /// Expected level under the option probabilities.
        score: f64,
        /// Level index → the caller's level description.
        legend: IndexMap<String, Value>,
        /// Probability per level index.
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        action: Action,
    },
    Noul {
        /// Probability that the statement holds.
        noul: f64,
        confidence: f64,
        action: Action,
    },
}

impl Answer {
    pub fn confidence(&self) -> f64 {
        match self {
            Answer::Choice { confidence, .. }
            | Answer::Score { confidence, .. }
            | Answer::Noul { confidence, .. } => *confidence,
        }
    }

    pub fn action(&self) -> &Action {
        match self {
            Answer::Choice { action, .. }
            | Answer::Score { action, .. }
            | Answer::Noul { action, .. } => action,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Everything returned for one state: an answer per question, in question order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    /// Present only when a `Router` chose the checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Value>,
}
