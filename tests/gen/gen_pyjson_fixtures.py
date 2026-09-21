"""Generates tests/fixtures/pyjson.json: JSON values and their exact `json.dumps` renderings.

Run with any Python 3:  python3 tests/gen/gen_pyjson_fixtures.py
"""
import json
import os

CASES = [
    {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411", "body": "Hi,\nwe were billed twice."},
    {"n": 1, "f": 1.0, "neg": -0.0, "small": 1e-5, "tiny": 1.2345e-7, "big": 1e16, "edge": 1e15, "pi": 3.141592653589793},
    {"unicode": "मुझसे दो बार शुल्क लिया गया", "emoji": "😀", "quote": "\"q\"", "bs": "a\\b", "ctl": "\x01\x1f\x7f", "tab": "\t"},
    {"nested": {"z": [1, [2, {"y": None}], {}], "a": []}, "b": True, "c": False},
    ["turn 1", {"role": "user", "content": "héllo"}, 2, 2.5, None],
    {"order": 1, "keeps": 2, "insertion": 3, "aardvark": 0},
    {"exp": 123456789012345678.0, "half": 0.5, "third": 0.3333333333333333, "hundred": 100.0, "k": 1000.0, "milli": 0.001, "tenth": 0.0001},
]

out = []
for state in CASES:
    out.append({
        "value": state,
        "state": json.dumps(state, ensure_ascii=False),
        "ascii": json.dumps(state),
    })

path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "pyjson.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", len(out), "cases to", os.path.normpath(path))
