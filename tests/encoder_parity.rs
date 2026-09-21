//! Encoder hidden states against the Python reference (`tests/gen/gen_model_fixtures.py`).

mod common;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use laya_candle::{
    build_sequence, AgentConfig, Budgets, Encoder, EncoderConfig, Question, State, Tokenizer,
};
use serde::Deserialize;
use std::collections::HashMap;

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
}

#[derive(Deserialize)]
struct PerQuestion {
    n_tokens: usize,
    hidden: HashMap<String, Vec<f32>>,
}

fn check(subdir: &str, fixture: &str, tolerance: f32) {
    let Some(root) = common::model_dir() else {
        return;
    };
    let dir = root.join(subdir);
    let dev = Device::Cpu;
    let agent_cfg = AgentConfig::from_file(dir.join("rl_agent_config.json")).unwrap();
    let enc_cfg = EncoderConfig::from_file(dir.join("encoder/config.json")).unwrap();
    let tok = Tokenizer::from_dir(dir.join("tokenizer")).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)
    }
    .unwrap();
    let encoder = Encoder::load(&enc_cfg, vb.pp("encoder")).unwrap();

    let fx: Fixture = serde_json::from_str(fixture).unwrap();
    let case = fx
        .cases
        .iter()
        .find(|c| c.name == "long")
        .expect("long case with hidden samples");
    let budgets = Budgets {
        max_len: agent_cfg.max_len,
        head_max_len: agent_cfg.head_max_len,
    };
    let seqs: Vec<_> = case
        .questions
        .iter()
        .map(|(qid, q)| build_sequence(&tok, &case.state, qid, q, budgets, false).unwrap())
        .collect();
    let l = seqs.iter().map(|s| s.len()).max().unwrap();
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for s in &seqs {
        ids.extend(
            s.ids
                .iter()
                .copied()
                .chain(std::iter::repeat(tok.pad_id))
                .take(l),
        );
        mask.extend(
            std::iter::repeat_n(1u32, s.len())
                .chain(std::iter::repeat(0))
                .take(l),
        );
    }
    let ids = Tensor::from_vec(ids, (seqs.len(), l), &dev).unwrap();
    let mask = Tensor::from_vec(mask, (seqs.len(), l), &dev).unwrap();
    let h = encoder.forward(&ids, &mask).unwrap();

    for (i, pq) in case.per_question.iter().enumerate() {
        assert_eq!(seqs[i].len(), pq.n_tokens, "{subdir}: sequence {i} length");
        for (pos, expected) in &pq.hidden {
            let pos: usize = pos.parse().unwrap();
            let got: Vec<f32> = h.get(i).unwrap().get(pos).unwrap().to_vec1().unwrap();
            let diff = common::max_abs_diff(&got, expected);
            eprintln!("{subdir}: seq {i} pos {pos}: max abs diff {diff:.2e}");
            assert!(
                diff < tolerance,
                "{subdir}: seq {i} pos {pos}: max abs diff {diff}"
            );
        }
    }
}

#[test]
fn english() {
    check("", include_str!("fixtures/model/english.json"), 1e-3);
}

#[test]
fn multilingual() {
    check(
        "multilingual",
        include_str!("fixtures/model/multilingual.json"),
        1e-2,
    );
}

#[test]
fn typed_decisions() {
    check(
        "typed-decisions",
        include_str!("fixtures/model/typed-decisions.json"),
        1e-2,
    );
}
