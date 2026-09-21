//! Answers the README questions about an email with a local checkpoint directory.
//!
//! cargo run --release --example quickstart -- <checkpoint-dir>

use candle_core::Device;
use laya_candle::{Agent, PredictOptions, Question, Questions, State};
use serde_json::json;

fn main() -> laya_candle::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: quickstart <checkpoint-dir>");
    let agent = Agent::from_dir(dir, &Device::Cpu, None)?;

    let state = State::from(json!({
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    }));
    let questions: Questions = [
        (
            "department".to_owned(),
            Question::choice(
                "Which department should handle this request?",
                [
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("sales", "pricing, new contracts"),
                    ("other", "everything else"),
                ],
            ),
        ),
        (
            "urgency".to_owned(),
            Question::score(
                "How urgent is this request?",
                ["not urgent", "soon", "critical deadline or blocking issue"],
            ),
        ),
        (
            "churn_risk".to_owned(),
            Question::noul("Does the user threaten to cancel or leave?"),
        ),
        (
            "refund_requested".to_owned(),
            Question::noul("Does the user explicitly request a refund?"),
        ),
    ]
    .into_iter()
    .collect();

    let started = std::time::Instant::now();
    let response = agent.predict(&state, &questions, &PredictOptions::default())?;
    eprintln!("{:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
