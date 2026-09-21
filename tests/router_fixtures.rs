//! `analyse` and `Router::route` against the Python package (`tests/gen/gen_router_fixtures.py`).
//! Needs no weights: routing never loads a model.

use candle_core::Device;
use laya_candle::{analyse, Checkpoint, Questions, RouteOverrides, Router, State};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    analyse: Vec<AnalyseCase>,
    routes: Vec<RouteCase>,
}

#[derive(Deserialize)]
struct AnalyseCase {
    name: String,
    state: State,
    analyse: Value,
}

#[derive(Deserialize)]
struct RouteCase {
    name: String,
    state: State,
    questions: Questions,
    auto: bool,
    overrides: Overrides,
    decision: Value,
}

#[derive(Deserialize, Default)]
struct Overrides {
    model: Option<String>,
    task: Option<String>,
    lang: Option<String>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/router.json")).unwrap()
}

#[test]
fn analyse_matches_python() {
    let fx = fixture();
    assert!(fx.analyse.len() >= 30);
    for case in &fx.analyse {
        let got = serde_json::to_value(analyse(&case.state)).unwrap();
        assert_eq!(got, case.analyse, "analyse({})", case.name);
    }
}

#[test]
fn route_matches_python() {
    let fx = fixture();
    let plain = Router::new(Device::Cpu);
    let auto = Router::new(Device::Cpu)
        .with_auto_task_detection(true)
        .with_default(Checkpoint::Multilingual);
    for case in &fx.routes {
        let router = if case.auto { &auto } else { &plain };
        let overrides = RouteOverrides {
            model: case.overrides.model.as_deref().map(|m| m.parse().unwrap()),
            task: case.overrides.task.clone(),
            lang: case.overrides.lang.clone(),
        };
        let got = serde_json::to_value(
            router
                .route(&case.state, &case.questions, &overrides)
                .unwrap(),
        )
        .unwrap();
        // Python's `explicit model=...` reason echoes the alias as typed; ours names the checkpoint.
        let mut expected = case.decision.clone();
        if let Some(model) = &case.overrides.model {
            let canonical: Checkpoint = model.parse().unwrap();
            expected["reason"] = Value::String(format!("explicit model='{}'", canonical.name()));
        }
        // Upstream reports the raw `(repo, subfolder)` tuple in the workflow branch; we always use
        // the `repo/subfolder` string every other branch uses.
        if let Value::Array(parts) = &expected["repo"] {
            let joined: Vec<&str> = parts.iter().filter_map(Value::as_str).collect();
            expected["repo"] = Value::String(joined.join("/"));
        }
        assert_eq!(got, expected, "route({})", case.name);
    }
}
