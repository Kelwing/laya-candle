//! Chooses which checkpoint answers a request and holds the loaded agents. Ported from the
//! Python package's `laya/router.py`.
//!
//! Precedence: explicit checkpoint, then explicit task, then a detected typed-decisions workflow
//! (opt-in), then explicit language, then detected script/language, then the default. `typed-decisions` is never chosen
//! automatically unless `auto_task_detection` is on: it is fine-tuned on four specific synthetic
//! workflows and should not be a silent default.

use crate::agent::{Agent, BatchOptions, PredictOptions, Request};
use crate::answer::Response;
use crate::checkpoint::{Checkpoint, BUNDLE_REPO};
use crate::error::{Error, Result};
use crate::lang::{analyse, Detection};
use crate::pyjson::repr_str;
use crate::question::Questions;
use crate::state::State;
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Question-id signatures of the four typed-decisions workflows.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// Name of the typed-decisions workflow whose question ids these are, if any. Requires an exact
/// id-set match, so an unrelated schema that happens to contain `urgency` is never captured.
pub fn match_typed_decisions_workflow(questions: &Questions) -> Option<&'static str> {
    let ids: BTreeSet<&str> = questions.keys().map(String::as_str).collect();
    TYPED_DECISION_WORKFLOWS
        .iter()
        .find(|(_, sig)| sig.iter().copied().collect::<BTreeSet<_>>() == ids)
        .map(|(name, _)| *name)
}

/// Where a checkpoint's files come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A Hub repo and optional subfolder (needs the `hub` feature to load).
    Hub {
        repo: String,
        subfolder: Option<String>,
    },
    /// A local checkpoint directory.
    Dir(PathBuf),
}

impl Source {
    /// `repo/subfolder` or the directory path, as reported in routing metadata.
    pub fn describe(&self) -> String {
        match self {
            Source::Hub {
                repo,
                subfolder: Some(sub),
            } => format!("{repo}/{sub}"),
            Source::Hub {
                repo,
                subfolder: None,
            } => repo.clone(),
            Source::Dir(path) => path.display().to_string(),
        }
    }
}

/// Caller overrides for one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RouteOverrides {
    /// Use exactly this checkpoint.
    pub model: Option<Checkpoint>,
    /// A task name (`typed_decisions` / `typed-decisions` or any checkpoint alias).
    pub task: Option<String>,
    /// A language tag; `en`/`eng`/`english` (any region) routes to English, anything else to
    /// multilingual.
    pub lang: Option<String>,
}

/// The routing outcome: which checkpoint, why, and what was detected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub model: Checkpoint,
    pub repo: String,
    pub reason: String,
    pub detection: Option<Detection>,
    pub workflow: Option<String>,
}

#[derive(Default)]
struct Loaded {
    agents: HashMap<Checkpoint, Arc<Agent>>,
    /// Least-recently-used first.
    order: Vec<Checkpoint>,
}

/// Routes requests to checkpoints and lazily loads them, keeping at most `max_loaded` resident.
///
/// Loading happens under the router's lock, so concurrent callers wait for a cold load; once
/// preloaded, routing costs only detection.
pub struct Router {
    sources: HashMap<Checkpoint, Source>,
    device: Device,
    dtype: Option<DType>,
    max_loaded: Mutex<usize>,
    default: Checkpoint,
    auto_task_detection: bool,
    loaded: Mutex<Loaded>,
}

impl Router {
    /// A router over the official bundle repo, loading onto `device`.
    pub fn new(device: Device) -> Self {
        let sources = Checkpoint::ALL
            .into_iter()
            .map(|c| {
                (
                    c,
                    Source::Hub {
                        repo: BUNDLE_REPO.to_owned(),
                        subfolder: c.subfolder().map(str::to_owned),
                    },
                )
            })
            .collect();
        Self {
            sources,
            device,
            dtype: None,
            max_loaded: Mutex::new(1),
            default: Checkpoint::English,
            auto_task_detection: false,
            loaded: Mutex::new(Loaded::default()),
        }
    }

    /// Override where one checkpoint is loaded from (a local directory or another Hub repo).
    pub fn with_source(mut self, checkpoint: Checkpoint, source: Source) -> Self {
        self.sources.insert(checkpoint, source);
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// How many checkpoints stay resident (least-recently-used eviction); at least 1.
    pub fn with_max_loaded(self, max_loaded: usize) -> Self {
        *self.max_loaded.lock().unwrap() = max_loaded.max(1);
        self
    }

    /// The checkpoint used when a state has no letters at all.
    pub fn with_default(mut self, default: Checkpoint) -> Self {
        self.default = default;
        self
    }

    /// Route to `typed-decisions` when the question ids exactly match one of its workflows.
    pub fn with_auto_task_detection(mut self, on: bool) -> Self {
        self.auto_task_detection = on;
        self
    }

    pub fn source(&self, checkpoint: Checkpoint) -> &Source {
        &self.sources[&checkpoint]
    }

    /// Checkpoints currently resident, least-recently-used first.
    pub fn loaded(&self) -> Vec<Checkpoint> {
        self.loaded.lock().unwrap().order.clone()
    }

    /// Returns the agent for `checkpoint`, loading it on first use.
    pub fn load(&self, checkpoint: Checkpoint) -> Result<Arc<Agent>> {
        let mut loaded = self.loaded.lock().unwrap();
        if let Some(agent) = loaded.agents.get(&checkpoint).cloned() {
            touch(&mut loaded.order, checkpoint);
            return Ok(agent);
        }
        let agent = Arc::new(self.build(checkpoint)?);
        loaded.agents.insert(checkpoint, agent.clone());
        loaded.order.push(checkpoint);
        self.evict(&mut loaded);
        Ok(agent)
    }

    fn build(&self, checkpoint: Checkpoint) -> Result<Agent> {
        match &self.sources[&checkpoint] {
            Source::Dir(dir) => Agent::from_dir(dir, &self.device, self.dtype),
            #[cfg(feature = "hub")]
            Source::Hub { repo, subfolder } => Agent::from_hub_repo(repo, subfolder.as_deref(), &self.device, self.dtype),
            #[cfg(not(feature = "hub"))]
            Source::Hub { repo, .. } => Err(Error::Config(format!(
                "checkpoint {checkpoint} comes from hub repo {repo} but the `hub` feature is disabled"
            ))),
        }
    }

    fn evict(&self, loaded: &mut Loaded) {
        let max = *self.max_loaded.lock().unwrap();
        while loaded.order.len() > max {
            let victim = loaded.order.remove(0);
            loaded.agents.remove(&victim);
        }
    }

    /// Registers an already-built agent instead of loading a second copy. Raises `max_loaded`
    /// so the attached agent is not immediately evicted.
    pub fn attach(&self, checkpoint: Checkpoint, agent: Arc<Agent>) {
        let mut loaded = self.loaded.lock().unwrap();
        loaded.agents.insert(checkpoint, agent);
        touch(&mut loaded.order, checkpoint);
        let mut max = self.max_loaded.lock().unwrap();
        *max = (*max).max(loaded.agents.len());
    }

    /// Loads checkpoints up front (all of them when `checkpoints` is empty), raising
    /// `max_loaded` to fit, so no request pays a cold load.
    pub fn preload(&self, checkpoints: &[Checkpoint]) -> Result<()> {
        let names: Vec<Checkpoint> = if checkpoints.is_empty() {
            Checkpoint::ALL.to_vec()
        } else {
            checkpoints.to_vec()
        };
        {
            let loaded = self.loaded.lock().unwrap();
            let mut max = self.max_loaded.lock().unwrap();
            *max = (*max).max(names.len()).max(loaded.agents.len());
        }
        for c in names {
            self.load(c)?;
        }
        Ok(())
    }

    /// Frees one checkpoint, or all of them.
    pub fn unload(&self, checkpoint: Option<Checkpoint>) {
        let mut loaded = self.loaded.lock().unwrap();
        match checkpoint {
            None => {
                loaded.agents.clear();
                loaded.order.clear();
            }
            Some(c) => {
                loaded.agents.remove(&c);
                loaded.order.retain(|&x| x != c);
            }
        }
    }

    /// Decides which checkpoint should answer, without loading or running anything.
    pub fn route(
        &self,
        state: &State,
        questions: &Questions,
        overrides: &RouteOverrides,
    ) -> Result<RouteDecision> {
        let decision = |model: Checkpoint,
                        reason: String,
                        detection: Option<Detection>,
                        workflow: Option<&str>| RouteDecision {
            model,
            repo: self.sources[&model].describe(),
            reason,
            detection,
            workflow: workflow.map(str::to_owned),
        };
        if let Some(model) = overrides.model {
            return Ok(decision(
                model,
                format!("explicit model={}", repr_str(model.name())),
                None,
                None,
            ));
        }
        if let Some(task) = &overrides.task {
            let key = if task.to_lowercase().replace('-', "_") == "typed_decisions" {
                Checkpoint::TypedDecisions
            } else {
                Checkpoint::parse(task)
                    .ok_or_else(|| Error::Config(format!("unknown task {task:?}")))?
            };
            return Ok(decision(
                key,
                format!("explicit task={}", repr_str(task)),
                None,
                None,
            ));
        }
        let workflow = match_typed_decisions_workflow(questions);
        if let (Some(wf), true) = (workflow, self.auto_task_detection) {
            return Ok(decision(
                Checkpoint::TypedDecisions,
                format!(
                    "question ids match the {} typed-decisions workflow",
                    repr_str(wf)
                ),
                None,
                workflow,
            ));
        }
        if let Some(lang) = &overrides.lang {
            let primary = lang.to_lowercase();
            let primary = primary.split('-').next().unwrap_or("");
            let key = if matches!(primary, "en" | "eng" | "english") {
                Checkpoint::English
            } else {
                Checkpoint::Multilingual
            };
            return Ok(decision(
                key,
                format!("explicit lang={}", repr_str(lang)),
                None,
                workflow,
            ));
        }
        let det = analyse(state);
        let (key, reason) = if det.script == "unknown" {
            (
                self.default,
                format!(
                    "no letters detected in state; using default ({})",
                    self.default.name()
                ),
            )
        } else if det.script != "latin" {
            (
                Checkpoint::Multilingual,
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            (
                Checkpoint::Multilingual,
                format!(
                    "Latin script but language looks like {}, not English",
                    repr_str(det.language.as_deref().unwrap_or(""))
                ),
            )
        } else {
            (Checkpoint::English, "English Latin text".to_owned())
        };
        Ok(decision(key, reason, Some(det), workflow))
    }

    /// Routes, then answers every question in one forward pass on the chosen checkpoint. The
    /// response carries the decision under `routing`.
    pub fn predict(
        &self,
        state: &State,
        questions: &Questions,
        overrides: &RouteOverrides,
        opts: &PredictOptions,
    ) -> Result<Response> {
        let decision = self.route(state, questions, overrides)?;
        let agent = self.load(decision.model)?;
        let mut response = agent.predict(state, questions, opts)?;
        response.routing = Some(serde_json::to_value(&decision)?);
        Ok(response)
    }

    /// Routes each request, runs one agent batch per chosen checkpoint, and returns responses in
    /// input order. A batch that mixes checkpoints under `max_loaded = 1` reloads once per
    /// group, not once per request.
    pub fn predict_batch(
        &self,
        requests: &[Request<'_>],
        overrides: &RouteOverrides,
        opts: &PredictOptions,
        batch: &BatchOptions,
    ) -> Result<Vec<Response>> {
        let decisions = requests
            .iter()
            .enumerate()
            .map(|(i, r)| {
                self.route(r.state, r.questions, overrides)
                    .map_err(|e| Error::Request {
                        request: i,
                        source: Box::new(e),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut groups: Vec<(Checkpoint, Vec<usize>)> = Vec::new();
        for (i, d) in decisions.iter().enumerate() {
            match groups.iter_mut().find(|(c, _)| *c == d.model) {
                Some((_, idx)) => idx.push(i),
                None => groups.push((d.model, vec![i])),
            }
        }
        let mut responses: Vec<Option<Response>> = (0..requests.len()).map(|_| None).collect();
        for (checkpoint, idx) in groups {
            let agent = self.load(checkpoint)?;
            let subset: Vec<Request<'_>> = idx.iter().map(|&i| requests[i]).collect();
            let out = agent
                .predict_batch(&subset, opts, batch)
                .map_err(|e| match e {
                    // Re-index a per-request failure to the caller's numbering.
                    Error::Request { request, source } => Error::Request {
                        request: idx[request],
                        source,
                    },
                    e => e,
                })?;
            for (i, mut response) in idx.into_iter().zip(out) {
                response.routing = Some(serde_json::to_value(&decisions[i])?);
                responses[i] = Some(response);
            }
        }
        Ok(responses
            .into_iter()
            .map(|r| r.expect("every request answered"))
            .collect())
    }
}

fn touch(order: &mut Vec<Checkpoint>, checkpoint: Checkpoint) {
    order.retain(|&c| c != checkpoint);
    order.push(checkpoint);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::question::Question;

    fn questions(ids: &[&str]) -> Questions {
        ids.iter()
            .map(|id| (id.to_string(), Question::noul("x")))
            .collect()
    }

    #[test]
    fn workflow_needs_exact_id_set() {
        assert_eq!(
            match_typed_decisions_workflow(&questions(&[
                "urgency",
                "action",
                "needs_review",
                "outcome",
                "risk"
            ])),
            Some("agent_trace_observability")
        );
        assert_eq!(
            match_typed_decisions_workflow(&questions(&["urgency", "action"])),
            None
        );
    }

    #[test]
    fn precedence() {
        let router = Router::new(Device::Cpu);
        let hindi = State::from("मुझसे दो बार शुल्क लिया गया");
        let q = questions(&["urgency", "action", "needs_review", "outcome", "risk"]);
        let d = router
            .route(&hindi, &q, &RouteOverrides::default())
            .unwrap();
        assert_eq!(d.model, Checkpoint::Multilingual);
        assert_eq!(d.workflow.as_deref(), Some("agent_trace_observability"));
        assert!(d
            .reason
            .starts_with("non-Latin script (devanagari, 100% of letters)"));

        let auto = Router::new(Device::Cpu).with_auto_task_detection(true);
        assert_eq!(
            auto.route(&hindi, &q, &RouteOverrides::default())
                .unwrap()
                .model,
            Checkpoint::TypedDecisions
        );

        let lang = RouteOverrides {
            lang: Some("en-GB".into()),
            ..Default::default()
        };
        let d = router.route(&hindi, &q, &lang).unwrap();
        assert_eq!(d.model, Checkpoint::English);
        assert_eq!(d.repo, "convaiinnovations/laya");

        let task = RouteOverrides {
            task: Some("typed-decisions".into()),
            ..Default::default()
        };
        assert_eq!(
            router.route(&hindi, &q, &task).unwrap().model,
            Checkpoint::TypedDecisions
        );
        let d = router
            .route(&State::from("12345"), &q, &RouteOverrides::default())
            .unwrap();
        assert_eq!(d.model, Checkpoint::English);
        assert_eq!(
            d.reason,
            "no letters detected in state; using default (english)"
        );
    }
}
