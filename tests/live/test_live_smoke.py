"""Live smoke test against the real TypeSafe API.

Run:  TYPESAFE_API_KEY=... pytest -m live -v
Results should be summarised in docs/live-validation.md.
"""

import json
from pathlib import Path

import pytest

from jev_harness import Choice, Gate, JevBackend, JsonlTrace, Noul, Score, Traced

pytestmark = pytest.mark.live

STATE = {
    "ticket_message": "My flight was cancelled. Can I get a refund?",
    "refund_policy": "Cancelled flights are eligible for a full refund.",
}
QUESTIONS = {
    "refund_requested": Noul("Does `ticket_message` request a refund?"),
    "request_type": Choice(
        "What is the main request in `ticket_message`?",
        {
            "refund": "The customer wants money returned.",
            "rebooking": "The customer wants a replacement flight.",
            "information": "The customer is asking for information only.",
        },
    ),
    "frustration": Score(
        "How frustrated does the customer appear in `ticket_message`?",
        ["Calm and neutral.", "Concerned but civil.", "Very angry or using strong language."],
    ),
}


def test_live_smoke(tmp_path):
    trace_path = Path("docs") / "live-trace.jsonl"
    trace_path.parent.mkdir(exist_ok=True)
    backend = Traced(JevBackend(), JsonlTrace(trace_path))
    resp = backend.evaluate(STATE, QUESTIONS)

    assert resp.model.startswith("jev-")
    assert resp.usage.input_tokens > 0
    assert resp.nouls["refund_requested"].noul > 0.5
    assert resp.choices["request_type"].choice == "refund"
    assert abs(sum(resp.choices["request_type"].probabilities.values()) - 1) < 0.02
    assert 0 <= resp.scores["frustration"].score <= 2
    decisions = Gate(act_at=0.6).decide_all(resp)

    summary = {
        "model": resp.model,
        "usage": resp.usage.to_wire(),
        "latency_ms": round(resp.latency_ms),
        "answers": resp.to_wire()["answers"],
        "decisions": {k: v.value for k, v in decisions.items()},
    }
    (Path("docs") / "live-last-run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def test_live_chinese_state():
    """Jev's primary language is English; this only records what CJK looks like."""
    resp = JevBackend().evaluate(
        {"message": "我的航班被取消了，可以退款吗？"},
        {"refund": Noul("Does `message` ask for a refund?")},
    )
    assert 0 <= resp.nouls["refund"].noul <= 1
    print("zh refund noul:", resp.nouls["refund"].noul)
