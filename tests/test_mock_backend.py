import pytest

from jev_harness.backends import MockBackend, choice_answer, score_answer
from jev_harness.types import Choice, ChoiceAnswer, Noul, NoulAnswer, Score


def _questions():
    return {
        "urgent": Noul("Is it urgent?"),
        "dept": Choice("Which dept?", {"billing": None, "tech": None}),
        "mood": Score("Mood?", ["calm", "angry"]),
    }


def test_uniform_defaults():
    r = MockBackend().evaluate("hi", _questions())
    assert r.nouls["urgent"].noul == 0.5
    assert r.choices["dept"].probabilities == {"billing": 0.5, "tech": 0.5}
    assert r.choices["dept"].confidence == 0.0
    assert r.scores["mood"].score == 0.5
    assert r.model == "mock-jev"


def test_rule_by_id_and_predicate():
    b = MockBackend()
    b.add_rule("urgent", NoulAnswer(0.9))
    b.add_rule(lambda qid, q: isinstance(q, Choice), choice_answer("tech", ["billing", "tech"]))
    r = b.evaluate("hi", _questions())
    assert r.nouls["urgent"].noul == 0.9
    assert r.choices["dept"].choice == "tech"
    assert r.choices["dept"].probabilities["tech"] == pytest.approx(0.9)


def test_callable_answer_sees_state():
    b = MockBackend().add_rule(
        "urgent", lambda state, qid, q: NoulAnswer(0.99 if "ASAP" in state else 0.01)
    )
    assert b.evaluate("fix ASAP", _questions()).nouls["urgent"].noul == 0.99
    assert b.evaluate("whenever", _questions()).nouls["urgent"].noul == 0.01


def test_calls_recorded_and_model_override():
    b = MockBackend()
    b.evaluate("s", {"q": Noul("x")}, model="jev-1.13.0")
    assert b.calls[0]["state"] == "s"
    assert b.calls[0]["model"] == "jev-1.13.0"


def test_scripted_answers_consumed_in_order():
    b = MockBackend().script({"urgent": NoulAnswer(0.1)}, {"urgent": NoulAnswer(0.8)})
    assert b.evaluate("s", _questions()).nouls["urgent"].noul == 0.1
    assert b.evaluate("s", _questions()).nouls["urgent"].noul == 0.8
    assert b.evaluate("s", _questions()).nouls["urgent"].noul == 0.5  # back to uniform


def test_score_answer_helper():
    a = score_answer(1, ["calm", "angry"], confidence=0.8)
    assert a.top_level == 1
    assert a.score == pytest.approx(0.8)
    assert isinstance(a, type(score_answer(0, ["a", "b"])))


async def test_async_path():
    r = await MockBackend().aevaluate("s", {"q": Noul("x")})
    assert isinstance(r.answers["q"], NoulAnswer)


def test_choice_answer_type():
    assert isinstance(choice_answer("a", ["a", "b"]), ChoiceAnswer)
