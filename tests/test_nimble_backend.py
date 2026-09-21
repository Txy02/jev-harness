import pytest

from jev_harness.backends import NimbleBackend
from jev_harness.backends.nimble import question_to_schema
from jev_harness.errors import UnsupportedQuestionError
from jev_harness.types import Choice, Noul, Score


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, context, schema):
        self.calls.append((context, schema))
        fields = {}
        output = {}
        for name, f in schema.items():
            if f["type"] == "boolean":
                fields[name] = {"scores": {"true": 0.8, "false": 0.2}}
                output[name] = True
            else:
                choices = f["choices"]
                fields[name] = {"scores": {c: (0.7 if i == 0 else 0.3 / (len(choices) - 1))
                                           for i, c in enumerate(choices)}}
                output[name] = choices[0]
        return {"output": output, "fields": fields}


def test_schema_mapping():
    assert question_to_schema(Noul("Is it urgent?")) == {
        "type": "boolean",
        "description": "Is it urgent?",
    }
    s = question_to_schema(Choice("Team?", {"billing": "Payments", "tech": None}))
    assert s["type"] == "enum"
    assert s["choices"] == ["billing", "tech"]
    assert s["choice_descriptions"] == {"billing": "Payments"}


def test_score_unsupported():
    with pytest.raises(UnsupportedQuestionError):
        question_to_schema(Score("q", ["a", "b"]))


def test_too_many_enum_choices():
    with pytest.raises(UnsupportedQuestionError):
        question_to_schema(Choice("q", {f"o{i}": None for i in range(27)}))


def test_evaluate_maps_answers():
    scorer = FakeScorer()
    b = NimbleBackend(scorer)
    r = b.evaluate(
        {"ticket": "payment down"},
        {"urgent": Noul("Urgent?"), "team": Choice("Team?", {"billing": None, "tech": None})},
    )
    assert r.model == "bespoke-nimble-9b"
    assert r.nouls["urgent"].noul == pytest.approx(0.8)
    assert r.choices["team"].choice == "billing"
    assert r.choices["team"].probabilities["billing"] == pytest.approx(0.7)
    assert 0 < r.choices["team"].confidence <= 1
    context, schema = scorer.calls[0]
    assert '"ticket": "payment down"' in context
    assert set(schema) == {"urgent", "team"}


async def test_async_evaluate():
    r = await NimbleBackend(FakeScorer()).aevaluate("s", {"u": Noul("q")})
    assert r.nouls["u"].is_yes
