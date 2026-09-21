//! `laya predict` / `laya batch`: answer typed questions from the command line.

use candle_core::{DType, Device};
use clap::{Args, Parser, Subcommand, ValueEnum};
use laya_candle::{
    Agent, BatchOptions, Checkpoint, PredictOptions, Questions, Request, RouteOverrides, Router,
    State,
};
use std::io::{BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "laya",
    version,
    about = "Laya calibrated decision models, in Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Answer questions about one state.
    Predict {
        #[command(flatten)]
        model: ModelArgs,
        /// The state: inline text, inline JSON, or `@path` to a file (JSON if it parses, else text).
        #[arg(long)]
        state: String,
        /// Questions as JSON (inline or `@path`), keyed by question id.
        #[arg(long)]
        questions: String,
    },
    /// Answer many requests from JSONL on stdin (`{"state": ..., "questions": {...}}` per line).
    Batch {
        #[command(flatten)]
        model: ModelArgs,
        /// Questions to apply to every line that lacks its own (JSON, inline or `@path`).
        #[arg(long)]
        questions: Option<String>,
        #[arg(long, default_value_t = 16384)]
        max_tokens: usize,
        #[arg(long, default_value_t = 256)]
        max_seqs: usize,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// Checkpoint name (english, multilingual, typed-decisions), `auto` to route per request by
    /// script/language, or a local checkpoint directory.
    #[arg(long, short = 'c', default_value = "english")]
    checkpoint: String,
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
    /// Weight dtype; defaults to f32 on CPU and bf16 elsewhere.
    #[arg(long, value_enum)]
    dtype: Option<DTypeArg>,
    /// Override the context budget (total tokens per question).
    #[arg(long)]
    max_len: Option<usize>,
    /// Override the head budget (tokens for instructions plus options).
    #[arg(long)]
    head_max_len: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, ValueEnum)]
enum DTypeArg {
    F32,
    F16,
    Bf16,
}

/// Either a fixed agent or a router that picks one per request.
enum Loaded {
    Agent(Box<Agent>),
    Router(Router),
}

impl Loaded {
    fn predict_batch(
        &self,
        requests: &[Request<'_>],
        opts: &PredictOptions,
        batch: &BatchOptions,
    ) -> laya_candle::Result<Vec<laya_candle::Response>> {
        match self {
            Loaded::Agent(agent) => agent.predict_batch(requests, opts, batch),
            Loaded::Router(router) => {
                router.predict_batch(requests, &RouteOverrides::default(), opts, batch)
            }
        }
    }
}

impl ModelArgs {
    fn load(&self) -> anyhow_lite::Result<(Loaded, PredictOptions)> {
        let device = match self.device {
            DeviceArg::Cpu => Device::Cpu,
            DeviceArg::Cuda => Device::new_cuda(0)?,
            DeviceArg::Metal => Device::new_metal(0)?,
        };
        let dtype = self.dtype.map(|d| match d {
            DTypeArg::F32 => DType::F32,
            DTypeArg::F16 => DType::F16,
            DTypeArg::Bf16 => DType::BF16,
        });
        let dir = PathBuf::from(&self.checkpoint);
        let agent = if self.checkpoint.eq_ignore_ascii_case("auto") {
            let mut router = Router::new(device).with_max_loaded(3);
            if let Some(dtype) = dtype {
                router = router.with_dtype(dtype);
            }
            Loaded::Router(router)
        } else if dir.is_dir() {
            Loaded::Agent(Box::new(Agent::from_dir(dir, &device, dtype)?))
        } else {
            let checkpoint: Checkpoint =
                self.checkpoint.parse().map_err(anyhow_lite::Error::msg)?;
            #[cfg(feature = "hub")]
            {
                Loaded::Agent(Box::new(Agent::from_hub(checkpoint, &device, dtype)?))
            }
            #[cfg(not(feature = "hub"))]
            {
                return Err(anyhow_lite::Error::msg(format!(
                    "{checkpoint} is not a directory and this build has no `hub` feature"
                )));
            }
        };
        let opts = PredictOptions {
            max_len: self.max_len,
            head_max_len: self.head_max_len,
        };
        Ok((agent, opts))
    }
}

/// `@path` reads the file; anything else is the literal text.
fn arg_text(arg: &str) -> anyhow_lite::Result<String> {
    match arg.strip_prefix('@') {
        Some(path) => Ok(std::fs::read_to_string(path)?),
        None => Ok(arg.to_owned()),
    }
}

fn parse_state(arg: &str) -> anyhow_lite::Result<State> {
    let text = arg_text(arg)?;
    Ok(match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) => State::from(v),
        Err(_) => State::from(text),
    })
}

fn parse_questions(arg: &str) -> anyhow_lite::Result<Questions> {
    Ok(serde_json::from_str(&arg_text(arg)?)?)
}

/// Minimal error plumbing so the binary needs no extra dependency.
mod anyhow_lite {
    pub type Result<T> = std::result::Result<T, Error>;

    #[derive(Debug)]
    pub struct Error(String);

    impl Error {
        pub fn msg(s: impl Into<String>) -> Self {
            Error(s.into())
        }
    }

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl<E: std::error::Error> From<E> for Error {
        fn from(e: E) -> Self {
            Error(e.to_string())
        }
    }
}

fn run() -> anyhow_lite::Result<()> {
    let cli = Cli::parse();
    let stdout = std::io::stdout();
    match cli.command {
        Command::Predict {
            model,
            state,
            questions,
        } => {
            let (agent, opts) = model.load()?;
            let state = parse_state(&state)?;
            let questions = parse_questions(&questions)?;
            let response = agent
                .predict_batch(
                    &[Request {
                        state: &state,
                        questions: &questions,
                    }],
                    &opts,
                    &BatchOptions::default(),
                )?
                .pop()
                .expect("one response");
            writeln!(
                stdout.lock(),
                "{}",
                serde_json::to_string_pretty(&response)?
            )?;
        }
        Command::Batch {
            model,
            questions,
            max_tokens,
            max_seqs,
        } => {
            let (agent, opts) = model.load()?;
            let shared = questions.map(|q| parse_questions(&q)).transpose()?;
            #[derive(serde::Deserialize)]
            struct Line {
                state: State,
                #[serde(default)]
                questions: Option<Questions>,
            }
            let mut lines = Vec::new();
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: Line = serde_json::from_str(&line)?;
                let questions = match (parsed.questions, &shared) {
                    (Some(q), _) => q,
                    (None, Some(q)) => q.clone(),
                    (None, None) => {
                        return Err(anyhow_lite::Error::msg(
                            "line has no questions and --questions not given",
                        ))
                    }
                };
                lines.push((parsed.state, questions));
            }
            let requests: Vec<_> = lines
                .iter()
                .map(|(state, questions)| Request { state, questions })
                .collect();
            let responses = agent.predict_batch(
                &requests,
                &opts,
                &BatchOptions {
                    max_tokens,
                    max_seqs,
                },
            )?;
            let mut out = stdout.lock();
            for response in responses {
                writeln!(out, "{}", serde_json::to_string(&response)?)?;
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
