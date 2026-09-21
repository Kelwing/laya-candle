use crate::answer::{Action, Answer, Response, Usage};
use crate::config::{AgentConfig, EncoderConfig};
use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::head::DecisionHead;
use crate::question::{Question, Questions};
use crate::sequence::{build_sequence, Budgets, Sequence};
use crate::state::State;
use crate::tokenizer::Tokenizer;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use std::path::Path;

/// The `model` field reported in every response, matching the Python package.
pub const MODEL_NAME: &str = "laya-rl-agent";

/// Per-call overrides of the checkpoint's token budgets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PredictOptions {
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

/// How sequences are packed into forward passes: sub-batches never exceed `max_tokens` padded
/// tokens or `max_seqs` sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchOptions {
    pub max_tokens: usize,
    pub max_seqs: usize,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            max_tokens: 16384,
            max_seqs: 256,
        }
    }
}

/// One state and the questions to answer about it.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub state: &'a State,
    pub questions: &'a Questions,
}

/// A loaded checkpoint. Cheap to share across threads (`&self` inference).
#[derive(Debug, Clone)]
pub struct Agent {
    encoder: Encoder,
    head: DecisionHead,
    tokenizer: Tokenizer,
    config: AgentConfig,
}

impl Agent {
    /// Loads a checkpoint directory (`rl_agent_config.json`, `encoder/config.json`, `tokenizer/`,
    /// `model.safetensors`). `dtype` defaults to F32 on CPU and BF16 on accelerators.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device, dtype: Option<DType>) -> Result<Self> {
        let dir = dir.as_ref();
        let dtype = dtype.unwrap_or(if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        });
        let config = AgentConfig::from_file(dir.join("rl_agent_config.json"))?;
        let encoder_config = EncoderConfig::from_file(dir.join("encoder").join("config.json"))?;
        let tokenizer = Tokenizer::from_dir(dir.join("tokenizer"))?;
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            return Err(Error::MissingFile(weights));
        }
        // SAFETY: the file is memory-mapped read-only and is not expected to change underneath us.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let encoder = Encoder::load(&encoder_config, vb.pp("encoder"))?;
        let head = DecisionHead::load(&config, encoder_config.hidden_size, vb)?;
        Ok(Self {
            encoder,
            head,
            tokenizer,
            config,
        })
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn encoder_config(&self) -> &EncoderConfig {
        &self.encoder.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn device(&self) -> &Device {
        self.encoder.device()
    }

    pub fn dtype(&self) -> DType {
        self.encoder.dtype()
    }

    fn budgets(&self, opts: &PredictOptions) -> Result<Budgets> {
        let max_len = opts.max_len.unwrap_or(self.config.max_len);
        let max_position_embeddings = self.encoder.config.max_position_embeddings;
        if max_len > max_position_embeddings {
            return Err(Error::ContextBudgetTooLarge {
                max_len,
                max_position_embeddings,
            });
        }
        Ok(Budgets {
            max_len,
            head_max_len: opts.head_max_len.unwrap_or(self.config.head_max_len),
        })
    }

    /// Answers every question about `state` in one forward pass.
    pub fn predict(
        &self,
        state: &State,
        questions: &Questions,
        opts: &PredictOptions,
    ) -> Result<Response> {
        let mut out = self.predict_batch(
            &[Request { state, questions }],
            opts,
            &BatchOptions::default(),
        )?;
        Ok(out.pop().expect("one response per request"))
    }

    /// Answers the same questions about many states.
    pub fn predict_many(
        &self,
        states: &[State],
        questions: &Questions,
        opts: &PredictOptions,
        batch: &BatchOptions,
    ) -> Result<Vec<Response>> {
        let requests: Vec<_> = states
            .iter()
            .map(|state| Request { state, questions })
            .collect();
        self.predict_batch(&requests, opts, batch)
    }

    /// Answers many requests, packing their sequences into as few forward passes as `batch`
    /// allows. Every request is validated and tokenized before any model work; the first bad
    /// request fails the whole call with [`Error::Request`]. Responses come back in input order.
    pub fn predict_batch(
        &self,
        requests: &[Request<'_>],
        opts: &PredictOptions,
        batch: &BatchOptions,
    ) -> Result<Vec<Response>> {
        let budgets = self.budgets(opts)?;
        let mut items: Vec<Item> = Vec::new();
        for (r, req) in requests.iter().enumerate() {
            for (qid, q) in req.questions {
                let seq = q
                    .validate(qid)
                    .and_then(|()| {
                        build_sequence(&self.tokenizer, req.state, qid, q, budgets, false)
                    })
                    .map_err(|e| Error::Request {
                        request: r,
                        source: Box::new(e),
                    })?;
                items.push(Item {
                    request: r,
                    seq,
                    result: None,
                });
            }
        }

        // Length-sorted packing under the padded-token budget, like the reference `predict_items`.
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].seq.len());
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            let mut longest = 0;
            while j < order.len()
                && j - i < batch.max_seqs
                && longest.max(items[order[j]].seq.len()) * (j - i + 1) <= batch.max_tokens
            {
                longest = longest.max(items[order[j]].seq.len());
                j += 1;
            }
            let j = j.max(i + 1);
            let selected: Vec<usize> = order[i..j].to_vec();
            let results = self.run(selected.iter().map(|&s| &items[s].seq))?;
            for (s, res) in selected.into_iter().zip(results) {
                items[s].result = Some(res);
            }
            i = j;
        }

        let mut responses: Vec<Response> = requests
            .iter()
            .map(|_| Response {
                model: MODEL_NAME.to_owned(),
                answers: IndexMap::new(),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
                routing: None,
            })
            .collect();
        for item in items {
            let raw = item.result.expect("every item was scored");
            let response = &mut responses[item.request];
            let (qid, q) = requests[item.request]
                .questions
                .get_index(response.answers.len())
                .expect("answers are filled in question order");
            response.usage.input_tokens += item.seq.len();
            response.answers.insert(qid.clone(), self.finish(q, &raw));
        }
        Ok(responses)
    }

    /// Runs one padded sub-batch through the encoder and head.
    fn run<'s>(
        &self,
        seqs: impl ExactSizeIterator<Item = &'s Sequence> + Clone,
    ) -> Result<Vec<Raw>> {
        let dev = self.device();
        let b = seqs.len();
        let l = seqs.clone().map(Sequence::len).max().unwrap_or(0);
        let k = seqs.clone().map(|s| s.markers.len()).max().unwrap_or(0);
        let pad = self.tokenizer.pad_id;
        let (mut ids, mut mask, mut mpos, mut mmask, mut kinds) = (
            Vec::with_capacity(b * l),
            Vec::with_capacity(b * l),
            Vec::with_capacity(b * k),
            Vec::with_capacity(b * k),
            Vec::with_capacity(b),
        );
        for s in seqs.clone() {
            ids.extend(s.ids.iter().copied().chain(std::iter::repeat(pad)).take(l));
            mask.extend(
                std::iter::repeat_n(1u32, s.len())
                    .chain(std::iter::repeat(0))
                    .take(l),
            );
            mpos.extend(
                s.markers
                    .iter()
                    .map(|&m| m as u32)
                    .chain(std::iter::repeat(0))
                    .take(k),
            );
            mmask.extend(
                std::iter::repeat_n(1u8, s.markers.len())
                    .chain(std::iter::repeat(0))
                    .take(k),
            );
            kinds.push(s.kind as u32);
        }
        let ids = Tensor::from_vec(ids, (b, l), dev)?;
        let mask = Tensor::from_vec(mask, (b, l), dev)?;
        let mpos = Tensor::from_vec(mpos, (b, k), dev)?;
        let mmask = Tensor::from_vec(mmask, (b, k), dev)?;
        let kinds = Tensor::from_vec(kinds, b, dev)?;

        let hidden = self.encoder.forward(&ids, &mask)?;
        let out = self.head.forward(&hidden, &mask, &mpos, &mmask, &kinds)?;
        let logits: Vec<Vec<f32>> = out.logits.to_vec2()?;
        let act: Vec<Vec<f32>> = candle_nn::ops::softmax_last_dim(&out.act_logits)?.to_vec2()?;
        Ok(seqs
            .zip(logits)
            .zip(act)
            .map(|((s, logits), act)| Raw {
                logits: logits[..s.markers.len()].to_vec(),
                act_probability: act[0],
            })
            .collect())
    }

    /// Calibrates the logits and shapes the typed answer, rounding like the Python package.
    fn finish(&self, q: &Question, raw: &Raw) -> Answer {
        let k = raw.logits.len();
        let t = self.config.temperature_for(q.kind(), k).max(1e-3);
        let p = softmax_f32(&raw.logits, t);
        let action = Action {
            act_probability: round4(raw.act_probability as f64),
        };
        match q {
            Question::Choice { criteria, .. } => {
                let best = (0..k).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0);
                Answer::Choice {
                    choice: criteria.0.keys().nth(best).cloned().unwrap_or_default(),
                    probabilities: criteria
                        .0
                        .keys()
                        .zip(&p)
                        .map(|(key, &v)| (key.clone(), round4(v as f64)))
                        .collect(),
                    confidence: round4(confidence(&p) as f64),
                    action,
                }
            }
            Question::Score { criteria, .. } => Answer::Score {
                score: round4(
                    p.iter()
                        .enumerate()
                        .map(|(i, &v)| i as f64 * v as f64)
                        .sum(),
                ),
                legend: criteria
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i.to_string(), c.clone()))
                    .collect(),
                probabilities: p
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i.to_string(), round4(v as f64)))
                    .collect(),
                confidence: round4(confidence(&p) as f64),
                action,
            },
            Question::Noul { .. } => {
                let noul = p[1] as f64;
                Answer::Noul {
                    noul: round4(noul),
                    confidence: round4(noul.max(1.0 - noul)),
                    action,
                }
            }
        }
    }
}

struct Item {
    request: usize,
    seq: Sequence,
    result: Option<Raw>,
}

#[derive(Debug, Clone)]
struct Raw {
    logits: Vec<f32>,
    act_probability: f32,
}

/// Temperature-scaled softmax in f32, as the reference does with numpy float32.
fn softmax_f32(logits: &[f32], temperature: f32) -> Vec<f32> {
    let z: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&v| v / sum).collect()
}

/// `1 - normalized entropy` of the answer distribution, in f32 like the reference.
fn confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p.iter().map(|&v| v * v.clamp(1e-12, 1.0).ln()).sum::<f32>();
    1.0 - ent / (k as f32).ln()
}

/// Python's `round(x, 4)`: the double nearest to the correctly rounded 4-decimal string.
pub(crate) fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}
