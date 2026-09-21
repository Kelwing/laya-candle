//! Token-level parity of `build_sequence` against the Python reference, per checkpoint.
//!
//! Needs the checkpoint tokenizers; set `LAYA_MODEL_DIR` to the downloaded
//! `convaiinnovations/laya` snapshot (the directory holding `tokenizer/`, `multilingual/`,
//! `typed-decisions/`). Skips otherwise.

use laya_candle::{build_sequence, Budgets, Error, Question, State, Tokenizer};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    special: Special,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Special {
    cls: u32,
    sep: u32,
    pad: u32,
    mask: u32,
    mask_token: String,
}

#[derive(Deserialize)]
struct Case {
    state_name: String,
    state: State,
    question_id: String,
    question: Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
    options: Vec<String>,
    ids: Vec<u32>,
    markers: Vec<usize>,
    fits: bool,
}

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("LAYA_MODEL_DIR").map(PathBuf::from)
}

fn check(checkpoint: &str, subdir: &str, fixture: &str) {
    let Some(root) = model_dir() else {
        eprintln!("LAYA_MODEL_DIR not set; skipping {checkpoint} sequence parity");
        return;
    };
    let tok = Tokenizer::from_dir(root.join(subdir).join("tokenizer")).expect("tokenizer loads");
    let fx: Fixture = serde_json::from_str(fixture).expect("fixture parses");
    assert_eq!(
        (tok.cls_id, tok.sep_id, tok.pad_id, tok.mask_id),
        (
            fx.special.cls,
            fx.special.sep,
            fx.special.pad,
            fx.special.mask
        )
    );
    assert_eq!(tok.mask_token, fx.special.mask_token);
    for case in &fx.cases {
        let label = format!("{checkpoint}/{}/{}", case.state_name, case.question_id);
        assert_eq!(
            case.question.render_options(),
            case.options,
            "{label}: options"
        );
        let budgets = Budgets {
            max_len: case.max_len,
            head_max_len: case.head_max_len,
        };
        let result = build_sequence(
            &tok,
            &case.state,
            &case.question_id,
            &case.question,
            budgets,
            case.truncate_left,
        );
        if case.fits {
            let seq = result.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(seq.ids, case.ids, "{label}: ids");
            assert_eq!(seq.markers, case.markers, "{label}: markers");
        } else {
            assert!(
                matches!(result, Err(Error::OptionsExceedHeadBudget { .. })),
                "{label}: expected OptionsExceedHeadBudget, got {result:?}"
            );
        }
    }
    let unfit = fx.cases.iter().filter(|c| !c.fits).count();
    assert!(unfit > 0, "fixture should include a non-fitting case");
}

#[test]
fn english() {
    check(
        "english",
        "",
        include_str!("fixtures/sequences/english.json"),
    );
}

#[test]
fn multilingual() {
    check(
        "multilingual",
        "multilingual",
        include_str!("fixtures/sequences/multilingual.json"),
    );
}

#[test]
fn typed_decisions() {
    check(
        "typed-decisions",
        "typed-decisions",
        include_str!("fixtures/sequences/typed-decisions.json"),
    );
}
