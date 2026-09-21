"""Quickstart: three typed questions, one request, gated decisions.

    pip install -e .
    export TYPESAFE_API_KEY=...    # or use --mock below
    python examples/quickstart.py [--mock]
"""

import sys

from jev_harness import Choice, Gate, JevBackend, JsonlTrace, MockBackend, Noul, Score, Traced

state = {
    "ticket": "I was charged twice for order A-104. Please refund the duplicate. This is urgent!",
    "policy": "Duplicate charges are eligible for a refund.",
}
questions = {
    "refund_requested": Noul("Does `ticket` request a refund?"),
    "policy_allows": Noul("Does `policy` support the refund requested in `ticket`?"),
    "department": Choice(
        "Which team should handle `ticket`?",
        {"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": "Pricing"},
    ),
    "urgency": Score("How urgent is `ticket`?", ["can wait", "this week", "today"]),
}

backend = MockBackend() if "--mock" in sys.argv else JevBackend()
backend = Traced(backend, JsonlTrace("runs/quickstart.jsonl"))

resp = backend.evaluate(state, questions)
gate = Gate(act_at=0.75, reject_below=0.3)

print(f"model={resp.model} tokens={resp.usage.input_tokens} latency={resp.latency_ms:.0f}ms")
for qid, ans in resp.answers.items():
    print(f"{qid:>18}: {ans.to_wire()}  -> {gate.decide(ans, qid).value}")

# Compose in code: auto-refund only when both nouls are confident yeses.
auto = (
    resp.nouls["refund_requested"].noul > 0.8
    and resp.nouls["policy_allows"].noul > 0.8
    and gate.decide(resp.choices["department"]).value == "act"
)
print("auto-refund:", auto)
