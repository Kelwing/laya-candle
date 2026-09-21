---
status: accepted
---

# Implement the ModernBERT encoder in this crate instead of depending on candle-transformers

`candle-transformers` ships a `modernbert` module, but its `Config` expects the pre-v5 flat
`global_rope_theta`/`local_rope_theta` keys while Laya's checkpoints carry transformers-v5
`rope_parameters`, and the decision head on top of the encoder has no upstream equivalent
anyway. We implement the encoder ourselves (with candle's module as a reference) so the crate
depends only on `candle-core`/`candle-nn`, we control mask construction and padding behaviour,
and numerical parity with the Python reference is verified against our own code rather than
tracked across upstream releases.

## Consequences

- Any ModernBERT fix or optimisation landing upstream must be ported by hand.
- The encoder must be validated by golden-fixture parity tests, not assumed correct because it
  loads.
