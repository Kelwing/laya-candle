"""Generates tests/fixtures/router.json from the Python package's `analyse` and `Router.route`.

Usage:  USE_TF=0 python3 tests/gen/gen_router_fixtures.py <path-to-laya-pip-package-parent>
`route()` never loads a model, so this needs no checkpoint (but the package imports torch).
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
from laya.lang import analyse  # noqa: E402
from laya.router import Router  # noqa: E402

STATES = {
    "english_email": {"from": "user@acme.com", "subject": "Duplicate charge",
                      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
    "english_short": "Refund me now",
    "english_terse": "ok thanks bye now",
    "hindi": {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    "german": "Mein Konto wurde zweimal belastet und ich möchte das Geld zurück, bitte.",
    "german_short": "Mein Konto wurde belastet",
    "french": "Bonjour, nous avons été facturés deux fois pour le mois de mars et nous voulons un remboursement.",
    "spanish": "Hola, se nos ha cobrado dos veces por el mes de marzo y queremos un reembolso por favor.",
    "portuguese": "Olá, fomos cobrados duas vezes pelo mês de março e queremos um reembolso, por favor.",
    "italian": "Ciao, siamo stati addebitati due volte per il mese di marzo e vogliamo un rimborso.",
    "dutch": "Hallo, we zijn twee keer gefactureerd voor de maand maart en we willen een terugbetaling.",
    "mixed_latin_cyrillic": "Please refund: Мне дважды выставили счет за март.",
    "mostly_english_with_han": "Order 8812 delayed 请退款 thanks",
    "japanese": "3月に二重請求されました。返金をお願いします。",
    "korean": "3월에 이중 청구되었습니다. 환불해 주세요.",
    "arabic": "تم تحصيل الرسوم مني مرتين، يرجى رد المبلغ.",
    "greek": "Χρεώθηκα δύο φορές, παρακαλώ επιστρέψτε τα χρήματα.",
    "thai": "ฉันถูกเรียกเก็บเงินสองครั้ง กรุณาคืนเงิน",
    "hebrew": "חויבתי פעמיים, אנא החזירו את הכסף.",
    "khmer": "ខ្ញុំត្រូវបានគិតថ្លៃពីរដង សូមសងប្រាក់វិញ។",
    "vietnamese": "Tôi đã bị tính phí hai lần, vui lòng hoàn tiền cho tôi.",
    "turkish": "Benden iki kez ücret alındı, lütfen parayı iade edin.",
    "polish": "Zostałem obciążony dwukrotnie, proszę o zwrot pieniędzy.",
    "digits_only": "4411 #8812 --- 2026",
    "empty": "",
    "json_nested": {"ticket": {"id": 5531, "messages": [{"role": "user", "text": "Мой заказ опаздывает"}, {"role": "agent", "text": "Checking"}]}},
    "json_deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": "мир"}}}}}}}, "note": "hello there friend"},
    "list_state": ["Ich habe eine Frage zu meiner Rechnung.", "Danke!"],
    "accents_english": "The café's naïve résumé was a cliché, but the crème brûlée was fine.",
    "long_english": " ".join("The shipment for order %d is late and the customer wants a refund." % i for i in range(300)),
    "emoji_only": "😀🎉👍",
    "typed_workflow_ids": "Invoice 4411 does not match the purchase order and looks like a duplicate.",
}

Q_PLAIN = {"department": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
           "urgency": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]}}
Q_INVOICE = {k: {"type": "noul", "instructions": "x"} for k in
             ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]}

router = Router()
router_auto = Router(auto_task_detection=True, default="multilingual")

cases = []
for name, state in STATES.items():
    cases.append({"name": name, "state": state, "analyse": analyse(state)})

routes = []
for name, state in STATES.items():
    routes.append({"name": name, "state": state, "questions": Q_PLAIN, "auto": False,
                   "overrides": {}, "decision": dict(router.route(state, Q_PLAIN))})
for name in ["english_email", "hindi", "digits_only", "typed_workflow_ids"]:
    state = STATES[name]
    routes.append({"name": name + "/invoice/auto", "state": state, "questions": Q_INVOICE, "auto": True,
                   "overrides": {}, "decision": dict(router_auto.route(state, Q_INVOICE))})
    routes.append({"name": name + "/invoice", "state": state, "questions": Q_INVOICE, "auto": False,
                   "overrides": {}, "decision": dict(router.route(state, Q_INVOICE))})
for ov in [{"model": "ml"}, {"model": "typed_decisions"}, {"task": "typed-decisions"}, {"task": "en"},
           {"lang": "en-GB"}, {"lang": "hi"}, {"lang": "English"}, {"model": "en", "lang": "hi"}]:
    routes.append({"name": "override:" + json.dumps(ov), "state": STATES["hindi"], "questions": Q_INVOICE,
                   "auto": False, "overrides": ov, "decision": dict(router.route(STATES["hindi"], Q_INVOICE, **ov))})

out = os.path.join(os.path.dirname(__file__), "..", "fixtures", "router.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"analyse": cases, "routes": routes}, f, ensure_ascii=False, indent=1)
print("wrote", len(cases), "analyse cases and", len(routes), "routes")
