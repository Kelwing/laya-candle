---
status: accepted
---

# Serialize JSON states exactly as Python's `json.dumps(ensure_ascii=False)` would

The checkpoints were trained on states rendered by Python's `json.dumps`, which emits
`{"a": 1, "b": [1, 2]}` with insertion-ordered keys, `", "`/`": "` separators, unescaped
non-ASCII, and Python `repr` float formatting (`1e-05`, `1.0`). `serde_json::to_string` emits
`{"a":1,"b":[1,2]}` and `1e-5`, which tokenizes differently and silently shifts model inputs.
We therefore render states with a custom formatter that reproduces Python's output byte for
byte, and fixture tests assert token-id equality against the Python reference.

## Consequences

- Questions and answers are kept in insertion-ordered maps (`IndexMap`, `serde_json`
  `preserve_order`) so key order survives a round trip.
- Python's `NaN`/`Infinity` output is not reproduced; serde never produces those values.
