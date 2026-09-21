"""NimbleBackend: adapter for Bespoke Labs' open Nimble scorer.

Nimble (https://github.com/bespokelabsai/nimble) is an open, Jev-inspired
model. Its scorer takes a text context and a flat schema of enum/boolean
fields and returns typed picks plus per-candidate probabilities.

This adapter maps:
    Choice -> {"type": "enum", "choices": [...]}
    Noul   -> {"type": "boolean"}
    Score  -> UnsupportedQuestionError (Nimble has no ordered-level primitive;
              model it as an enum Choice if you need it)

Nimble limits: 1..26 enum choices, ~2048 prompt tokens, text only.

NOTE: this adapter has been tested against a fake scorer only. It has not
been validated on a real Nimble checkpoint in this repository.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from ..errors import UnsupportedQuestionError, ValidationError
from ..types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Response,
    Score,
    StateType,
    Usage,
    questions_to_wire,
)

NIMBLE_MAX_ENUM = 26


class NimbleScorer(Protocol):
    def score(self, context: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def _describe(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False)


def question_to_schema(question: Question) -> dict[str, Any]:
    if isinstance(question, Noul):
        return {"type": "boolean", "description": _describe(question.instructions)}
    if isinstance(question, Choice):
        opts = question.options
        if len(opts) > NIMBLE_MAX_ENUM:
            raise UnsupportedQuestionError(
                f"Nimble enum fields support at most {NIMBLE_MAX_ENUM} choices, got {len(opts)}"
            )
        field: dict[str, Any] = {
            "type": "enum",
            "choices": opts,
            "description": _describe(question.instructions),
        }
        descs = {k: _describe(v) for k, v in question.criteria.items() if v is not None}
        if descs:
            field["choice_descriptions"] = descs
        return field
    if isinstance(question, Score):
        raise UnsupportedQuestionError(
            "Nimble has no Score primitive; express ordered levels as a Choice instead"
        )
    raise ValidationError(f"unknown question {question!r}")


def _to_answer(question: Question, field_result: dict[str, Any]) -> Answer:
    scores = field_result.get("scores") or field_result.get("probabilities") or {}
    if isinstance(question, Noul):
        p_true = scores.get("true", scores.get(True))
        if p_true is None:
            output = field_result.get("output")
            p_true = 1.0 if output is True else 0.0
        return NoulAnswer(float(p_true))
    if isinstance(question, Choice):
        opts = question.options
        probs = {o: float(scores.get(o, 0.0)) for o in opts}
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {o: 1.0 / len(opts) for o in opts}
        choice = field_result.get("output") or max(probs, key=probs.get)
        n = len(opts)
        peak = max(probs.values())
        confidence = (n * peak - 1) / (n - 1) if n > 1 else 1.0
        return ChoiceAnswer(
            choice=str(choice), probabilities=probs, confidence=max(0.0, confidence)
        )
    raise UnsupportedQuestionError(f"cannot convert {question.type}")


class NimbleBackend:
    """Wrap a Nimble `scorer.score(context, schema)` object as a Backend."""

    name = "nimble"

    def __init__(self, scorer: NimbleScorer, *, model: str = "bespoke-nimble-9b") -> None:
        self.scorer = scorer
        self.model = model

    @staticmethod
    def state_to_context(state: StateType) -> str:
        if isinstance(state, str):
            return state
        return json.dumps(state, ensure_ascii=False, indent=2)

    def evaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        questions_to_wire(questions)
        schema = {qid: question_to_schema(q) for qid, q in questions.items()}
        result = self.scorer.score(self.state_to_context(state), schema)
        fields = result.get("fields", {})
        outputs = result.get("output", {})
        answers: dict[str, Answer] = {}
        for qid, q in questions.items():
            fr = dict(fields.get(qid, {}))
            if "output" not in fr and qid in outputs:
                fr["output"] = outputs[qid]
            answers[qid] = _to_answer(q, fr)
        return Response(model=model or self.model, answers=answers, usage=Usage())

    async def aevaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        return await asyncio.to_thread(self.evaluate, state, questions, model=model)
