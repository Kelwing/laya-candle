//! `State::render` and `Instructions::render` against renderings produced by CPython's `json.dumps`.

use laya_candle::{Instructions, State};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    value: serde_json::Value,
    state: String,
    ascii: String,
}

#[test]
fn matches_cpython_json_dumps() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/pyjson.json")).expect("fixture parses");
    assert!(!cases.is_empty());
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(
            State::Json(case.value.clone()).render(),
            case.state,
            "case {i} (ensure_ascii=False)"
        );
        assert_eq!(
            Instructions::Json(case.value.clone()).render(),
            case.ascii,
            "case {i} (ensure_ascii=True)"
        );
    }
}
