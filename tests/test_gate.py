import pytest

from jev_harness.backends import MockBackend, choice_answer
from jev_harness.errors import ValidationError
from jev_harness.gate import Decision, Gate, confidence_of
from jev_harness.types import Choice, ChoiceAnswer, Noul, NoulAnswer, ScoreAnswer


def test_boundaries():
    g = Gate(act_at=0.8, reject_below=0.3)
    assert g.decide(ChoiceAnswer("a", {"a": 1}, 0.8)) is Decision.ACT
    assert g.decide(ChoiceAnswer("a", {"a": 1}, 0.79)) is Decision.REVIEW
    assert g.decide(ChoiceAnswer("a", {"a": 1}, 0.3)) is Decision.REVIEW
    assert g.decide(ChoiceAnswer("a", {"a": 1}, 0.29)) is Decision.REJECT


def test_noul_uses_certainty():
    g = Gate(act_at=0.8, reject_below=0.3)
    assert g.decide(NoulAnswer(0.95)) is Decision.ACT  # certainty 0.9
    assert g.decide(NoulAnswer(0.05)) is Decision.ACT  # strong no is also certain
    assert g.decide(NoulAnswer(0.55)) is Decision.REJECT  # certainty 0.1
    assert confidence_of(NoulAnswer(0.75)) == pytest.approx(0.5)


def test_score_uses_confidence():
    a = ScoreAnswer(1.0, {"0": "a", "1": "b"}, {"0": 0.1, "1": 0.9}, 0.85)
    assert Gate().decide(a) is Decision.ACT


def test_overrides_by_id():
    g = Gate(act_at=0.8, reject_below=0.3, overrides={"refund": (0.95, 0.5)})
    a = ChoiceAnswer("a", {"a": 1}, 0.9)
    assert g.decide(a) is Decision.ACT
    assert g.decide(a, "refund") is Decision.REVIEW
    assert g.decide(ChoiceAnswer("a", {"a": 1}, 0.4), "refund") is Decision.REJECT


def test_invalid_thresholds():
    with pytest.raises(ValidationError):
        Gate(act_at=1.5)
    with pytest.raises(ValidationError):
        Gate(act_at=0.3, reject_below=0.5)
    with pytest.raises(ValidationError):
        Gate(overrides={"x": (0.2, 0.5)})


def test_decide_all():
    b = MockBackend().add_rule("dept", choice_answer("billing", ["billing", "tech"], 0.9))
    r = b.evaluate("s", {"dept": Choice("q", {"billing": None, "tech": None}), "u": Noul("q")})
    d = Gate().decide_all(r)
    assert d == {"dept": Decision.ACT, "u": Decision.REJECT}
