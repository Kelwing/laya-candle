"""Generates tests/fixtures/model/<checkpoint>.json: end-to-end responses, raw logits/act
probabilities, and sampled encoder hidden states from the Python reference (pip package, CPU, f32).

Usage:
    USE_TF=0 python3 tests/gen/gen_model_fixtures.py <path-to-laya-pip-package-parent> <checkpoint-root-dir>
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items  # noqa: E402

ROOT = sys.argv[2]
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "model")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

EMAIL = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
LONG = " ".join(
    "Update %d: the shipment for order 8812 is still delayed, support ticket 5531 has no reply, and the customer "
    "is asking for a partial credit while threatening to dispute the charge." % i for i in range(12))
HINDI = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}

Q = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
    "tone": {"type": "choice", "instructions": {"rule": "pick the tøne"}, "criteria": ["angry", "neutral", "happy"]},
    "rubric": {"type": "choice", "instructions": "Classify", "criteria": {"a": {"desc": "first"}, "b": 0, "c": None}},
    "sentiment": {"type": "score", "instructions": "Overall sentiment", "criteria": ["negative", "mixed", "positive", "delighted", "ecstatic", "overjoyed"]},
}

CASES = [
    ("email", EMAIL, ["department", "urgency", "churn_risk", "refund_requested"]),
    ("long", LONG, ["department", "churn_risk", "sentiment"]),
    ("hindi", HINDI, ["department", "churn_risk"]),
    ("text", "Refund me now or I cancel.", ["tone", "rubric"]),
]

CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


def r(x, nd=6):
    return [round(float(v), nd) for v in x]


for name, sub in CHECKPOINTS.items():
    agent = Agent(ROOT, device="cpu", subfolder=sub)
    agent.model.eval()
    fixture = {"cases": []}
    for case_name, state, qids in CASES:
        questions = {q: Q[q] for q in qids}
        items = []
        for qid in qids:
            q = Agent._to_internal(questions[qid])
            ids, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            h = agent.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        act = torch.softmax(act.float(), -1)
        per_q = []
        for i, qid in enumerate(qids):
            k = len(items[i]["markers"])
            n = int(b["attention_mask"][i].sum())
            positions = sorted(set([0, items[i]["markers"][0], items[i]["markers"][-1], n // 2, n - 1]))
            per_q.append({
                "question_id": qid, "n_tokens": n,
                "logits": r(logits[i, :k]), "act": r(act[i]),
                "hidden": {str(p): r(h[i, p], 5) for p in positions} if case_name == "long" else {},
            })
        response = agent.predict(state, questions)
        fixture["cases"].append({"name": case_name, "state": state, "questions": questions,
                                 "per_question": per_q, "response": response})
        print(name, case_name, "L=%d" % b["input_ids"].shape[1], {q: response["answers"][q].get("choice", response["answers"][q].get("score", response["answers"][q].get("noul"))) for q in qids})
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False)
    del agent
