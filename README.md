# laya-candle

A pure-Rust runtime for the [Laya](https://huggingface.co/convaiinnovations/laya) family of
non-autoregressive, calibrated decision models, built on [candle](https://github.com/huggingface/candle).
No Python, no PyTorch: the ModernBERT / mmBERT encoder, the decision head, the sequence builder,
calibration, and the script/language router are all implemented here and verified against the
Python package with golden fixtures (token ids, encoder states, and final JSON are identical).

Status: pre-release (`0.1.0`). The design vocabulary lives in [`CONTEXT.md`](./CONTEXT.md) and the
decisions in [`docs/adr/`](./docs/adr/).

## Usage

```rust
use candle_core::Device;
use laya_candle::{Agent, Checkpoint, PredictOptions, Question, Questions, State};
use serde_json::json;

let agent = Agent::from_hub(Checkpoint::English, &Device::Cpu, None)?; // or Agent::from_dir(path, ..)

let state = State::from(json!({"subject": "Duplicate charge", "body": "Please refund or we cancel."}));
let questions: Questions = [
    ("department".to_string(), Question::choice("Which department?", [("billing", "refunds"), ("other", "else")])),
    ("urgency".to_string(), Question::score("How urgent?", ["not urgent", "soon", "critical"])),
    ("churn_risk".to_string(), Question::noul("Does the user threaten to leave?")),
].into_iter().collect();

let response = agent.predict(&state, &questions, &PredictOptions::default())?;
println!("{}", serde_json::to_string_pretty(&response)?); // same JSON shape as the Python package
```

Questions and responses (de)serialize to the Python wire format, so JSON from an existing Laya
client works unchanged: `let questions: Questions = serde_json::from_str(json)?`.

- **Batching**: `agent.predict_batch(&requests, &opts, &BatchOptions::default())` packs many
  states' questions into as few forward passes as the token budget allows; `predict_many` fans one
  question set over many states.
- **Routing**: `Router::new(device)` picks `english` / `multilingual` / `typed-decisions` per
  request from the state's script and language (same heuristics as upstream), holds loaded agents
  with LRU eviction, and batches across checkpoints. Responses carry a `routing` field.
- **Devices**: CPU (F32) by default; `--features cuda` or `metal` for accelerators (BF16 default).
- **Budgets**: `PredictOptions { max_len, head_max_len }` overrides the checkpoint defaults per call.

### CLI

```sh
cargo install --path . --features cli
laya predict -c english --state '{"body": "Refund me or I cancel"}' --questions @questions.json
laya batch -c auto --device cuda --questions @questions.json < states.jsonl   # routes per line
```

## Cargo features

| feature | default | effect |
|---|---|---|
| `hub` | yes | `Agent::from_hub`, `Router` loading from the Hugging Face Hub (`hf-hub`, shares the HF cache) |
| `cli` | no | the `laya` binary (`clap`) |
| `cuda`, `cudnn`, `metal` | no | passed through to candle |

## Development

```sh
nix develop            # CPU toolchain
nix develop .#cuda     # adds the CUDA toolkit (CUDA_COMPUTE_CAP=86 for an RTX 30xx)
cargo test             # weight-free tier: JSON formatter, configs, routing fixtures
LAYA_MODEL_DIR=~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/<sha> \
  cargo test --release # adds tokenizer, encoder, and end-to-end parity against the fixtures
```

Fixtures under `tests/fixtures/` were produced by the Python reference; the generators in
`tests/gen/` document how (they need the `laya` pip package sources and a torch environment).

## Divergences from the Python package

- Choice/Score questions with fewer than two options are rejected instead of answered.
- `RouteDecision::repo` is always `repo/subfolder` (upstream leaks a tuple in one branch).
- JSON states are rendered exactly as `json.dumps` would (see ADR-0002); `NaN`/`Infinity` are not.

## License

Apache-2.0 (see [LICENSE](./LICENSE)), matching the upstream Laya weights and code from Convai
Innovations.
