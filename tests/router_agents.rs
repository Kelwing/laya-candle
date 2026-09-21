//! `Router` end to end with local checkpoints: routing attaches metadata, mixed-language batches
//! answer like the individually chosen agents, and LRU eviction behaves.

mod common;

use candle_core::Device;
use laya_candle::{
    BatchOptions, Checkpoint, PredictOptions, Question, Questions, Request, RouteOverrides, Router,
    Source, State,
};

fn questions() -> Questions {
    [
        (
            "department".to_owned(),
            Question::choice(
                "Which department?",
                [
                    ("billing", "refunds"),
                    ("technical", "bugs"),
                    ("other", "else"),
                ],
            ),
        ),
        (
            "churn_risk".to_owned(),
            Question::noul("Does the user threaten to cancel or leave?"),
        ),
    ]
    .into_iter()
    .collect()
}

#[test]
fn routes_and_batches_across_checkpoints() {
    let Some(root) = common::model_dir() else {
        return;
    };
    let router = Router::new(Device::Cpu)
        .with_source(Checkpoint::English, Source::Dir(root.clone()))
        .with_source(
            Checkpoint::Multilingual,
            Source::Dir(root.join("multilingual")),
        )
        .with_source(
            Checkpoint::TypedDecisions,
            Source::Dir(root.join("typed-decisions")),
        )
        .with_max_loaded(1);
    let qs = questions();
    let states = [
        State::from("We were billed twice for March. Please refund the duplicate or we cancel."),
        State::from("मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"),
        State::from("Mein Konto wurde zweimal belastet und ich möchte das Geld zurück, bitte."),
    ];
    let opts = PredictOptions::default();

    let single: Vec<_> = states
        .iter()
        .map(|s| {
            router
                .predict(s, &qs, &RouteOverrides::default(), &opts)
                .unwrap()
        })
        .collect();
    let models: Vec<&str> = single
        .iter()
        .map(|r| r.routing.as_ref().unwrap()["model"].as_str().unwrap())
        .collect();
    assert_eq!(models, ["english", "multilingual", "multilingual"]);
    assert_eq!(
        router.loaded(),
        [Checkpoint::Multilingual],
        "max_loaded=1 keeps only the last used"
    );

    let requests: Vec<_> = states
        .iter()
        .map(|s| Request {
            state: s,
            questions: &qs,
        })
        .collect();
    let batched = router
        .predict_batch(
            &requests,
            &RouteOverrides::default(),
            &opts,
            &BatchOptions::default(),
        )
        .unwrap();
    assert_eq!(
        batched, single,
        "batched routing must equal individual routing"
    );

    // Explicit override wins and is reported.
    let forced = RouteOverrides {
        model: Some(Checkpoint::English),
        ..Default::default()
    };
    let r = router.predict(&states[1], &qs, &forced, &opts).unwrap();
    assert_eq!(r.routing.as_ref().unwrap()["model"], "english");

    router
        .preload(&[Checkpoint::English, Checkpoint::Multilingual])
        .unwrap();
    assert_eq!(router.loaded().len(), 2);
    router.unload(None);
    assert!(router.loaded().is_empty());
}
