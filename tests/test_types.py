import pytest

from jev_harness.errors import ValidationError
from jev_harness.types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    from_wire,
    questions_to_wire,
)


def test_noul_to_wire_matches_docs():
    q = Noul("Does this convey urgency?", criteria={"true": "Explicit", "false": "None"})
    assert q.to_wire() == {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicit", "false": "None"},
    }


def test_noul_without_criteria_omits_key():
    assert "criteria" not in Noul("Is it urgent?").to_wire()


def test_noul_rejects_bad_criteria_keys():
    with pytest.raises(ValidationError):
        Noul("q", criteria={"yes": "x"})


def test_choice_to_wire_and_options():
    q = Choice("Which team?", {"billing": "Payments", "technical": None})
    assert q.to_wire()["criteria"] == {"billing": "Payments", "technical": None}
    assert q.options == ["billing", "technical"]


def test_choice_rejects_too_many_options():
    with pytest.raises(ValidationError):
        Choice("q", {f"o{i}": None for i in range(256)})


def test_choice_rejects_empty():
    with pytest.raises(ValidationError):
        Choice("q", {})


def test_score_levels_bounds():
    Score("q", ["a", "b"])
    Score("q", [str(i) for i in range(10)])
    with pytest.raises(ValidationError):
        Score("q", ["only"])
    with pytest.raises(ValidationError):
        Score("q", [str(i) for i in range(11)])


def test_structured_instructions_allowed():
    q = Noul({"question": "Same person as `dup`?", "dup": {"name": "John"}})
    assert q.to_wire()["instructions"]["dup"] == {"name": "John"}


def test_empty_instructions_rejected():
    with pytest.raises(ValidationError):
        Noul("   ")


def test_questions_to_wire_requires_questions():
    with pytest.raises(ValidationError):
        questions_to_wire({})


def test_from_wire_parses_docs_examples():
    questions = {
        "is_urgent": Noul("Does this convey urgency?"),
        "department": Choice("Which team?", {"billing": None, "technical": None, "sales": None}),
        "frustration": Score("How frustrated?", ["Calm", "Frustrated", "Very angry"]),
    }
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                "confidence": 0.81,
            },
            "frustration": {
                "type": "score",
                "score": 1.05,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                "confidence": 0.92,
            },
        },
        "usage": {"input_tokens": 318, "output_tokens": 34},
    }
    r = from_wire(questions, payload, latency_ms=12.5)
    assert r.model == "jev-1.13.0"
    assert r.nouls["is_urgent"].noul == 0.95
    assert r.choices["department"].choice == "billing"
    assert r.choices["department"].margin() == pytest.approx(0.76)
    assert r.scores["frustration"].top_level == 1
    assert r.usage.input_tokens == 318
    assert r.latency_ms == 12.5
    assert r["is_urgent"].is_yes


def test_from_wire_type_mismatch():
    with pytest.raises(ValidationError):
        from_wire({"q": Noul("x")}, {"answers": {"q": {"type": "choice", "choice": "a"}}})


def test_from_wire_missing_answer():
    with pytest.raises(ValidationError):
        from_wire({"q": Noul("x")}, {"answers": {}})


def test_certainty_and_margin():
    assert NoulAnswer(0.95).certainty() == pytest.approx(0.9)
    assert NoulAnswer(0.5).certainty() == 0.0
    assert ChoiceAnswer("a", {"a": 0.6, "b": 0.4}, 0.2).margin() == pytest.approx(0.2)
    assert ChoiceAnswer("a", {"a": 1.0}, 1.0).margin() == 1.0


def test_response_round_trip_to_wire():
    qs = {"q": Noul("x")}
    payload = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.2}}, "usage": {}}
    r = from_wire(qs, payload)
    assert r.to_wire()["answers"]["q"] == {"type": "noul", "noul": 0.2}
