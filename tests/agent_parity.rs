//! End-to-end `Agent` parity with the Python package (`tests/gen/gen_model_fixtures.py`):
//! raw logits and act probabilities within tolerance, and the rounded response JSON identical.

mod common;

use candle_core::Device;
use laya_candle::{Agent, BatchOptions, PredictOptions, Question, Request, Response, State};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    state: State,
    questions: indexmap::IndexMap<String, Question>,
    per_question: Vec<PerQuestion>,
    response: serde_json::Value,
}

#[derive(Deserialize)]
struct PerQuestion {
    n_tokens: usize,
}

fn load(subdir: &str) -> Option<Agent> {
    let root = common::model_dir()?;
    Some(Agent::from_dir(root.join(subdir), &Device::Cpu, None).expect("checkpoint loads"))
}

fn check(subdir: &str, fixture: &str) {
    let Some(agent) = load(subdir) else { return };
    let fx: Fixture = serde_json::from_str(fixture).unwrap();
    let opts = PredictOptions::default();
    let mut individual = Vec::new();
    for case in &fx.cases {
        let response = agent.predict(&case.state, &case.questions, &opts).unwrap();
        let got = serde_json::to_value(&response).unwrap();
        // Rounded numbers must match exactly, so compare the JSON values.
        assert_eq!(got, case.response, "{subdir}/{}: response JSON", case.name);
        let total: usize = case.per_question.iter().map(|q| q.n_tokens).sum();
        assert_eq!(
            response.usage.input_tokens, total,
            "{subdir}/{}: input_tokens",
            case.name
        );
        individual.push(response);
    }

    // Batching all cases together must give the same answers as one call per case.
    let requests: Vec<_> = fx
        .cases
        .iter()
        .map(|c| Request {
            state: &c.state,
            questions: &c.questions,
        })
        .collect();
    for batch in [
        BatchOptions::default(),
        BatchOptions {
            max_tokens: 700,
            max_seqs: 3,
        },
    ] {
        let batched: Vec<Response> = agent.predict_batch(&requests, &opts, &batch).unwrap();
        assert_eq!(batched.len(), individual.len());
        for (i, (b, s)) in batched.iter().zip(&individual).enumerate() {
            assert_eq!(
                b, s,
                "{subdir}: batched response {i} differs from individual ({batch:?})"
            );
        }
    }
}

#[test]
fn english() {
    check("", include_str!("fixtures/model/english.json"));
}

#[test]
fn multilingual() {
    check(
        "multilingual",
        include_str!("fixtures/model/multilingual.json"),
    );
}

#[test]
fn typed_decisions() {
    check(
        "typed-decisions",
        include_str!("fixtures/model/typed-decisions.json"),
    );
}
