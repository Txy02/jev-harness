import json

import pytest

from jev_harness.backends import MockBackend
from jev_harness.errors import JevError
from jev_harness.trace import JsonlTrace, Traced, no_preview, replay, state_digest
from jev_harness.types import Noul, NoulAnswer


def test_one_line_per_call_with_fields(tmp_path):
    path = tmp_path / "t.jsonl"
    b = Traced(MockBackend().add_rule("q", NoulAnswer(0.7)), JsonlTrace(path))
    b.evaluate({"msg": "hello"}, {"q": Noul("Is it a greeting?")}, model="jev-1.13.0")
    b.evaluate("second", {"q": Noul("x")})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["backend"] == "mock"
    assert rec["model"] == "jev-1.13.0"
    assert rec["requested_model"] == "jev-1.13.0"
    assert rec["question_ids"] == ["q"]
    assert rec["answers"]["q"] == {"type": "noul", "noul": 0.7}
    assert rec["state_preview"] == '{"msg": "hello"}'
    assert rec["state_digest"] == state_digest({"msg": "hello"})
    assert rec["error"] is None


def test_digest_stable_regardless_of_key_order():
    assert state_digest({"a": 1, "b": 2}) == state_digest({"b": 2, "a": 1})


def test_redact_removes_preview(tmp_path):
    path = tmp_path / "t.jsonl"
    Traced(MockBackend(), JsonlTrace(path, redact=no_preview)).evaluate("secret", {"q": Noul("x")})
    assert json.loads(path.read_text())["state_preview"] == ""


def test_replay_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    b = Traced(MockBackend(), JsonlTrace(path))
    for i in range(3):
        b.evaluate(f"s{i}", {"q": Noul("x")})
    recs = replay(path)
    assert [r.state_preview for r in recs] == ["s0", "s1", "s2"]
    assert recs[0].to_json() == path.read_text(encoding="utf-8").splitlines()[0]


class Boom:
    name = "boom"

    def evaluate(self, state, questions, *, model=None):
        raise JevError("kaboom")

    async def aevaluate(self, state, questions, *, model=None):
        raise JevError("kaboom")


def test_errors_are_recorded_and_reraised(tmp_path):
    path = tmp_path / "t.jsonl"
    with pytest.raises(JevError):
        Traced(Boom(), JsonlTrace(path)).evaluate("s", {"q": Noul("x")})
    rec = replay(path)[0]
    assert rec.error == "JevError: kaboom"
    assert rec.answers == {}


async def test_async_traced(tmp_path):
    path = tmp_path / "t.jsonl"
    r = await Traced(MockBackend(), JsonlTrace(path)).aevaluate("s", {"q": Noul("x")})
    assert r.nouls["q"].noul == 0.5
    assert len(replay(path)) == 1


def test_traced_preserves_response(tmp_path):
    b = Traced(MockBackend().add_rule("q", NoulAnswer(0.2)), JsonlTrace(tmp_path / "t.jsonl"))
    assert b.evaluate("s", {"q": Noul("x")}).nouls["q"].noul == 0.2
    assert b.name == "traced(mock)"
