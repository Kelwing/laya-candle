"""Generates tests/fixtures/sequences/<checkpoint>.json from the Python reference `build_sequence`.

Usage:
    python3 tests/gen/gen_sequence_fixtures.py <path-to-laya-pip-package-parent> <checkpoint-root-dir>

Needs the `laya` pip package sources on sys.path (they import torch) and the downloaded
checkpoint directory that holds `tokenizer/`, `multilingual/`, `typed-decisions/`.
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
from transformers import AutoTokenizer  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import build_sequence, render_options  # noqa: E402

ROOT = sys.argv[2]
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sequences")
os.makedirs(OUT, exist_ok=True)

EMAIL = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
LONG = " ".join("Paragraph %d of a long support thread about a delayed shipment and an unanswered ticket." % i for i in range(120))
CONVO = [{"role": "user", "content": "My order is late"}, {"role": "agent", "content": "Sorry! Checking now."},
         {"role": "user", "content": "It has been [MASK] three weeks. I want a refund."}]
HINDI = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?",
               "criteria": {"true": "an explicit ask for money back", "false": ""}},
    "list_choice": {"type": "choice", "instructions": {"rule": "pick the tøne"}, "criteria": ["angry", "neutral", "happy"]},
    "rubric": {"type": "choice", "instructions": "Classify",
               "criteria": {"a": {"desc": "first", "weight": 1.5}, "b": 0, "c": None, "d": "", "e": [1, "x"]}},
    "many": {"type": "choice", "instructions": "Which intent? " * 30,
             "criteria": {"intent_%02d" % i: "a fairly long description of intent number %d that keeps going" % i
                          for i in range(40)}},
    "masky": {"type": "noul", "instructions": "Contains [MASK] literally?"},
}

CHECKPOINTS = {
    "english": ("", {"max_len": 512, "head_max_len": 192}),
    "multilingual": ("multilingual", {"max_len": 1024, "head_max_len": 256}),
    "typed-decisions": ("typed-decisions", {"max_len": 1024, "head_max_len": 256}),
}

CASES = []
for state_name, state in [("email", EMAIL), ("text", "Refund me now or I cancel."), ("long", LONG),
                          ("convo", CONVO), ("hindi", HINDI)]:
    for qid, q in QUESTIONS.items():
        CASES.append({"state_name": state_name, "state": state, "qid": qid, "q": q, "truncate_left": False})
CASES.append({"state_name": "convo", "state": CONVO, "qid": "churn_risk", "q": QUESTIONS["churn_risk"], "truncate_left": True})
CASES.append({"state_name": "long", "state": LONG, "qid": "urgency", "q": QUESTIONS["urgency"], "truncate_left": True})
CASES.append({"state_name": "email", "state": EMAIL, "qid": "many", "q": QUESTIONS["many"],
              "truncate_left": False, "budgets": {"max_len": 1024, "head_max_len": 512}})
CASES.append({"state_name": "email", "state": EMAIL, "qid": "urgency", "q": QUESTIONS["urgency"],
              "truncate_left": False, "budgets": {"max_len": 64, "head_max_len": 40}})

CASES.append({"state_name": "email", "state": EMAIL, "qid": "many", "q": QUESTIONS["many"],
              "truncate_left": False, "budgets": {"max_len": 64, "head_max_len": 192}})  # markers fall off the end

for name, (sub, budgets) in CHECKPOINTS.items():
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, sub, "tokenizer"))
    out = {"special": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id,
                       "mask": tok.mask_token_id, "mask_token": tok.mask_token}, "cases": []}
    for case in CASES:
        b = case.get("budgets", budgets)
        q = Agent._to_internal(case["q"])
        ids, markers = build_sequence(tok, case["state"], q, b["max_len"], b["head_max_len"],
                                      truncate_left=case["truncate_left"])
        out["cases"].append({
            "state_name": case["state_name"], "state": case["state"], "question_id": case["qid"],
            "question": case["q"], "max_len": b["max_len"], "head_max_len": b["head_max_len"],
            "truncate_left": case["truncate_left"], "options": render_options(q),
            "ids": ids, "markers": markers, "fits": len(markers) == len(render_options(q)),
        })
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(name, len(out["cases"]), "cases;", sum(not c["fits"] for c in out["cases"]), "do not fit")
