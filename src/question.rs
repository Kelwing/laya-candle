use crate::error::{Error, Result};
use crate::pyjson;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The three question kinds, in the model's kind-embedding order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Noul => "noul",
        }
    }
}

/// Question instructions: a string, or any JSON value (rendered with `json.dumps`, ASCII-escaped,
/// exactly as the Python reference does).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Instructions {
    Text(String),
    Json(Value),
}

impl Instructions {
    pub fn render(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Instructions::Text(s) => s.as_str().into(),
            Instructions::Json(v) => pyjson::dumps(v, true).into(),
        }
    }
}

impl From<&str> for Instructions {
    fn from(s: &str) -> Self {
        Instructions::Text(s.to_owned())
    }
}

impl From<String> for Instructions {
    fn from(s: String) -> Self {
        Instructions::Text(s)
    }
}

/// Choice criteria: option key → optional description. On the wire this is either an object
/// (values may be `null`/`""` for "no description", or any JSON for a structured rubric) or a
/// plain list of keys.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[serde(transparent)]
pub struct ChoiceCriteria(pub IndexMap<String, Value>);

impl<'de> Deserialize<'de> for ChoiceCriteria {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Map(IndexMap<String, Value>),
            List(Vec<String>),
        }
        Ok(match Wire::deserialize(d)? {
            Wire::Map(m) => ChoiceCriteria(m),
            Wire::List(keys) => {
                ChoiceCriteria(keys.into_iter().map(|k| (k, Value::Null)).collect())
            }
        })
    }
}

impl<K: Into<String>, V: Into<Value>> FromIterator<(K, V)> for ChoiceCriteria {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        ChoiceCriteria(
            iter.into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }
}

/// Optional descriptions for a noul's two options.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub r#false: Value,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub r#true: Value,
}

/// A typed question about a state. Serializes to the Python wire shape
/// `{"type": "choice", "instructions": ..., "criteria": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice {
        instructions: Instructions,
        criteria: ChoiceCriteria,
    },
    Score {
        instructions: Instructions,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: Instructions,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

/// Questions keyed by caller-chosen id, in insertion order.
pub type Questions = IndexMap<String, Question>;

impl Question {
    pub fn choice(
        instructions: impl Into<Instructions>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<Value>)>,
    ) -> Self {
        Question::Choice {
            instructions: instructions.into(),
            criteria: criteria.into_iter().collect(),
        }
    }

    pub fn score(
        instructions: impl Into<Instructions>,
        levels: impl IntoIterator<Item = impl Into<Value>>,
    ) -> Self {
        Question::Score {
            instructions: instructions.into(),
            criteria: levels.into_iter().map(Into::into).collect(),
        }
    }

    pub fn noul(instructions: impl Into<Instructions>) -> Self {
        Question::Noul {
            instructions: instructions.into(),
            criteria: None,
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Question::Choice { .. } => Kind::Choice,
            Question::Score { .. } => Kind::Score,
            Question::Noul { .. } => Kind::Noul,
        }
    }

    pub fn instructions(&self) -> &Instructions {
        match self {
            Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Noul { instructions, .. } => instructions,
        }
    }

    /// Option texts in label-index order; a noul is always `[false, true]`.
    pub fn render_options(&self) -> Vec<String> {
        match self {
            Question::Choice { criteria, .. } => criteria
                .0
                .iter()
                .map(|(k, v)| match described(v) {
                    Some(desc) => format!("{k}: {desc}"),
                    None => k.clone(),
                })
                .collect(),
            Question::Score { criteria, .. } => criteria
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect(),
            Question::Noul { criteria, .. } => {
                let c = criteria.clone().unwrap_or_default();
                vec![
                    format!(
                        "false: {}",
                        described(&c.r#false)
                            .unwrap_or_else(|| "no, the statement does not hold".into())
                    ),
                    format!(
                        "true: {}",
                        described(&c.r#true).unwrap_or_else(|| "yes, the statement holds".into())
                    ),
                ]
            }
        }
    }

    /// Rejects questions the model cannot meaningfully answer (fewer than two options).
    /// The Python reference accepts these silently; see the crate docs for this divergence.
    pub fn validate(&self, question_id: &str) -> Result<()> {
        let options = match self {
            Question::Choice { criteria, .. } => criteria.0.len(),
            Question::Score { criteria, .. } => criteria.len(),
            Question::Noul { .. } => 2,
        };
        if options < 2 {
            return Err(Error::TooFewOptions {
                question_id: question_id.to_owned(),
                options,
            });
        }
        Ok(())
    }
}

/// `None`/`""` mean "no description"; everything else renders (0 and false are legitimate).
fn described(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        v => Some(render_criterion(v)),
    }
}

/// Strings pass through; anything structured becomes Python-style JSON.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn choice_options_render_like_python() {
        let q = Question::choice(
            "Which department?",
            [
                ("billing", json!("invoices")),
                ("other", json!(null)),
                ("blank", json!("")),
                ("zero", json!(0)),
            ],
        );
        assert_eq!(
            q.render_options(),
            ["billing: invoices", "other", "blank", "zero: 0"]
        );
    }

    #[test]
    fn score_and_noul_options() {
        let q = Question::score("How urgent?", ["low", "high"]);
        assert_eq!(q.render_options(), ["level 0: low", "level 1: high"]);
        let q = Question::noul("Refund?");
        assert_eq!(
            q.render_options(),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let q: Question = serde_json::from_value(json!({
            "type": "noul", "instructions": "x", "criteria": {"true": {"desc": "yes"}}
        }))
        .unwrap();
        assert_eq!(q.render_options()[1], "true: {\"desc\": \"yes\"}");
    }

    #[test]
    fn wire_shapes() {
        let q: Question = serde_json::from_value(json!({
            "type": "choice", "instructions": "x", "criteria": ["a", "b"]
        }))
        .unwrap();
        assert_eq!(q.render_options(), ["a", "b"]);
        let q: Question =
            serde_json::from_value(json!({"type": "noul", "instructions": {"rule": "é"}})).unwrap();
        assert_eq!(q.instructions().render(), "{\"rule\": \"\\u00e9\"}");
        assert_eq!(
            serde_json::to_value(&q).unwrap(),
            json!({"type": "noul", "instructions": {"rule": "é"}})
        );
        let qs: Questions = serde_json::from_value(json!({
            "b": {"type": "score", "instructions": "s", "criteria": ["x", "y"]},
            "a": {"type": "noul", "instructions": "n"}
        }))
        .unwrap();
        assert_eq!(qs.keys().collect::<Vec<_>>(), ["b", "a"]);
    }

    #[test]
    fn validation_rejects_degenerate_questions() {
        assert!(Question::choice("x", [("only", json!(null))])
            .validate("q")
            .is_err());
        assert!(Question::score("x", Vec::<&str>::new())
            .validate("q")
            .is_err());
        assert!(Question::noul("x").validate("q").is_ok());
    }
}
