//! Accelerator parity: answers on CUDA (BF16) must agree with the Python CPU reference to
//! within bf16 tolerance. Only built with the `cuda` feature; skips without `LAYA_MODEL_DIR`.
#![cfg(feature = "cuda")]

mod common;

use candle_core::Device;
use laya_candle::{Agent, Answer, PredictOptions, Question, State};
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
    response: serde_json::Value,
}

#[test]
fn cuda_bf16_agrees_with_reference() {
    let Some(root) = common::model_dir() else {
        return;
    };
    let agent = Agent::from_dir(&root, &Device::new_cuda(0).unwrap(), None).unwrap();
    let fx: Fixture = serde_json::from_str(include_str!("fixtures/model/english.json")).unwrap();
    for case in &fx.cases {
        let response = agent
            .predict(&case.state, &case.questions, &PredictOptions::default())
            .unwrap();
        for (qid, answer) in &response.answers {
            let expected = &case.response["answers"][qid];
            match answer {
                Answer::Choice {
                    choice,
                    probabilities,
                    ..
                } => {
                    assert_eq!(
                        choice,
                        expected["choice"].as_str().unwrap(),
                        "{}/{qid}",
                        case.name
                    );
                    for (k, p) in probabilities {
                        let e = expected["probabilities"][k].as_f64().unwrap();
                        assert!((p - e).abs() < 0.03, "{}/{qid}/{k}: {p} vs {e}", case.name);
                    }
                }
                Answer::Score { score, .. } => {
                    let e = expected["score"].as_f64().unwrap();
                    assert!(
                        (score - e).abs() < 0.05,
                        "{}/{qid}: {score} vs {e}",
                        case.name
                    );
                }
                Answer::Noul { noul, .. } => {
                    let e = expected["noul"].as_f64().unwrap();
                    assert!(
                        (noul - e).abs() < 0.03,
                        "{}/{qid}: {noul} vs {e}",
                        case.name
                    );
                }
            }
        }
    }
}
