#![allow(dead_code)]

use std::path::PathBuf;

/// The downloaded `convaiinnovations/laya` snapshot, or `None` to skip weight-dependent tests.
pub fn model_dir() -> Option<PathBuf> {
    match std::env::var_os("LAYA_MODEL_DIR") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            eprintln!("LAYA_MODEL_DIR not set; skipping test that needs checkpoint weights");
            None
        }
    }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
