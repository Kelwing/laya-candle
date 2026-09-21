use crate::pyjson;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The thing being judged: free text, or a JSON document rendered the way the Python reference
/// renders it (`json.dumps(state, ensure_ascii=False)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum State {
    Text(String),
    Json(Value),
}

impl State {
    /// The text the model sees for this state.
    pub fn render(&self) -> std::borrow::Cow<'_, str> {
        match self {
            State::Text(s) => s.as_str().into(),
            State::Json(v) => pyjson::dumps(v, false).into(),
        }
    }
}

impl From<&str> for State {
    fn from(s: &str) -> Self {
        State::Text(s.to_owned())
    }
}

impl From<String> for State {
    fn from(s: String) -> Self {
        State::Text(s)
    }
}

impl From<Value> for State {
    fn from(v: Value) -> Self {
        match v {
            Value::String(s) => State::Text(s),
            v => State::Json(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_state_renders_python_style() {
        let s = State::from(json!({"from": "a@b.c", "n": 2, "ok": false}));
        assert_eq!(s.render(), "{\"from\": \"a@b.c\", \"n\": 2, \"ok\": false}");
        assert_eq!(State::from("plain").render(), "plain");
    }

    #[test]
    fn deserializes_untagged() {
        let s: State = serde_json::from_str("\"hi\"").unwrap();
        assert_eq!(s, State::Text("hi".into()));
        let s: State = serde_json::from_str("[1, 2]").unwrap();
        assert_eq!(s, State::Json(json!([1, 2])));
    }
}
