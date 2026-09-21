//! Builds the token sequence the model reads for one question about one state:
//! `[CLS] <kind> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`.

use crate::error::{Error, Result};
use crate::question::{Kind, Question};
use crate::state::State;
use crate::tokenizer::Tokenizer;

/// Per-option token cap (after the marker).
const OPTION_CAP: usize = 48;
/// Below this much head budget left for instructions, option texts are shrunk evenly.
const MIN_OPT_BUDGET: usize = 16;
/// Floor on the per-option length when shrinking.
const MIN_PER_OPTION: usize = 4;
/// Floor on the instruction length.
const MIN_HEAD: usize = 8;

/// Token budgets for one request; defaults come from the checkpoint's `AgentConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    /// Context budget: total tokens for one question sequence.
    pub max_len: usize,
    /// Head budget: tokens shared by the instructions and all option texts.
    pub head_max_len: usize,
}

/// A tokenized question ready for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Position of each option's marker, in option order.
    pub markers: Vec<usize>,
    pub kind: Kind,
}

impl Sequence {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Tokenizes `question` about `state` under `budgets`.
///
/// `truncate_left` keeps the tail of the state instead of the head (used for conversations).
/// Fails with [`Error::OptionsExceedHeadBudget`] when not every option's marker survives.
pub fn build_sequence(
    tok: &Tokenizer,
    state: &State,
    question_id: &str,
    question: &Question,
    budgets: Budgets,
    truncate_left: bool,
) -> Result<Sequence> {
    let Budgets {
        max_len,
        head_max_len,
    } = budgets;
    let kind = question.kind();
    let opts = question.render_options();
    let mut head_ids = tok.encode(&format!(
        "{} question: {}",
        kind.name(),
        question.instructions().render()
    ))?;
    let mut opt_ids = Vec::with_capacity(opts.len());
    for opt in &opts {
        let mut ids = vec![tok.mask_id];
        let text = tok.encode(&format!(" {opt}"))?;
        ids.extend(text.into_iter().take(OPTION_CAP));
        opt_ids.push(ids);
    }
    let used = |opt_ids: &[Vec<u32>]| opt_ids.iter().map(Vec::len).sum::<usize>();
    let mut opt_budget = head_max_len.saturating_sub(used(&opt_ids));
    if opt_budget < MIN_OPT_BUDGET {
        // Too many / too long options: shrink every option text evenly.
        let per = (head_max_len.saturating_sub(MIN_OPT_BUDGET) / opt_ids.len().max(1))
            .max(MIN_PER_OPTION);
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len.saturating_sub(used(&opt_ids));
    }
    head_ids.truncate(opt_budget.max(MIN_HEAD));

    let mut ids = Vec::with_capacity(max_len);
    ids.push(tok.cls_id);
    ids.extend(head_ids);
    ids.push(tok.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(tok.sep_id);

    let room = max_len.saturating_sub(ids.len() + 1);
    let st = tok.encode(&state.render())?;
    let st = if truncate_left {
        &st[st.len().saturating_sub(room)..]
    } else {
        &st[..st.len().min(room)]
    };
    ids.extend_from_slice(st);
    ids.push(tok.sep_id);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);

    if markers.len() != opts.len() {
        return Err(Error::OptionsExceedHeadBudget {
            question_id: question_id.to_owned(),
            head_max_len,
        });
    }
    Ok(Sequence { ids, markers, kind })
}
