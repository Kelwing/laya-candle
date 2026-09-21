//! Wall-clock per call and per question for a few batch shapes.
//!
//! cargo run --release --example bench -- <checkpoint-dir> [cpu|cuda]

use candle_core::Device;
use laya_candle::{Agent, BatchOptions, PredictOptions, Question, Questions, Request, State};
use std::time::Instant;

fn main() -> laya_candle::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: bench <checkpoint-dir> [cpu|cuda]");
    let device = match std::env::args().nth(2).as_deref() {
        Some("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let started = Instant::now();
    let agent = Agent::from_dir(dir, &device, None)?;
    eprintln!(
        "load: {:.0} ms ({:?}, {:?})",
        started.elapsed().as_secs_f64() * 1e3,
        agent.device(),
        agent.dtype()
    );

    let state = State::from("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.");
    let questions: Questions = (0..10)
        .map(|i| {
            let q = match i % 3 {
                0 => Question::choice(
                    "Which department?",
                    [
                        ("billing", "refunds"),
                        ("technical", "bugs"),
                        ("other", "else"),
                    ],
                ),
                1 => Question::score("How urgent?", ["not urgent", "soon", "critical"]),
                _ => Question::noul("Does the user threaten to leave?"),
            };
            (format!("q{i}"), q)
        })
        .collect();
    let opts = PredictOptions::default();

    let one: Questions = questions
        .iter()
        .take(1)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    agent.predict(&state, &one, &opts)?; // warm-up
    for n in [1usize, 5, 10] {
        let qs: Questions = questions
            .iter()
            .take(n)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let t = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            agent.predict(&state, &qs, &opts)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        eprintln!(
            "{n:2} questions/call: {ms:7.1} ms/call, {:6.1} ms/question",
            ms / n as f64
        );
    }

    let states: Vec<State> = (0..32)
        .map(|i| State::from(format!("Ticket {i}: my order is late and I want a refund.")))
        .collect();
    let requests: Vec<_> = states
        .iter()
        .map(|s| Request {
            state: s,
            questions: &one,
        })
        .collect();
    for max_tokens in [2048usize, 8192, 16384] {
        let batch = BatchOptions {
            max_tokens,
            max_seqs: 256,
        };
        let t = Instant::now();
        agent.predict_batch(&requests, &opts, &batch)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "32 states x 1 question, max_tokens={max_tokens:5}: {ms:7.1} ms total, {:5.1} ms/state",
            ms / 32.0
        );
    }
    Ok(())
}
