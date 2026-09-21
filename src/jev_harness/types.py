"""Typed questions and answers mirroring the TypeSafe System One wire format.

Wire format (https://docs.typesafe.ai/api):

    question: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}
    answer:   {"type": "noul", "noul": 0.95}
              {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.81}
              {"type": "score", "score": 1.05, "legend": {...},
               "probabilities": {...}, "confidence": 0.92}
    response: {"model": "jev-1.13.0", "answers": {...},
               "usage": {"input_tokens": 296, "output_tokens": 20}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError

Instructions = str | dict | list
StateType = str | dict | list

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


# --------------------------------------------------------------------------- questions


def _check_instructions(instructions: Any) -> None:
    if not isinstance(instructions, (str, dict, list)):
        raise ValidationError("instructions must be a string, object, or array")
    if isinstance(instructions, str) and not instructions.strip():
        raise ValidationError("instructions must not be empty")


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is the probability that the answer is yes."""

    instructions: Instructions
    criteria: dict[str, Any] | None = None
    type: str = field(default="noul", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if self.criteria is not None:
            if not isinstance(self.criteria, dict):
                raise ValidationError("noul criteria must be an object with 'true'/'false' keys")
            bad = set(self.criteria) - {"true", "false"}
            if bad:
                raise ValidationError(
                    f"noul criteria keys must be 'true'/'false', got {sorted(bad)}"
                )

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            wire["criteria"] = self.criteria
        return wire


@dataclass(frozen=True)
class Choice:
    """Pick one option from a defined set. Max 255 options."""

    instructions: Instructions
    criteria: dict[str, Any]
    type: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if not isinstance(self.criteria, dict) or not self.criteria:
            raise ValidationError(
                "choice criteria must be a non-empty mapping of option -> description"
            )
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValidationError(
                f"choice supports at most {MAX_CHOICE_OPTIONS} options, got {len(self.criteria)}"
            )
        for key in self.criteria:
            if not isinstance(key, str) or not key:
                raise ValidationError("choice option keys must be non-empty strings")

    @property
    def options(self) -> list[str]:
        return list(self.criteria)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Rate the state along ordered levels. Between 2 and 10 levels."""

    instructions: Instructions
    criteria: list[Any]
    type: str = field(default="score", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if not isinstance(self.criteria, (list, tuple)):
            raise ValidationError("score criteria must be an ordered list of level descriptions")
        n = len(self.criteria)
        if n < MIN_SCORE_LEVELS or n > MAX_SCORE_LEVELS:
            raise ValidationError(
                f"score needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {n}"
            )
        object.__setattr__(self, "criteria", list(self.criteria))

    @property
    def levels(self) -> list[Any]:
        return list(self.criteria)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score


def questions_to_wire(questions: dict[str, Question]) -> dict[str, dict[str, Any]]:
    if not questions:
        raise ValidationError("at least one question is required")
    out: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ValidationError("question ids must be non-empty strings")
        if not isinstance(q, (Noul, Choice, Score)):
            raise ValidationError(f"question {qid!r} is not a Noul/Choice/Score instance")
        out[qid] = q.to_wire()
    return out


# --------------------------------------------------------------------------- answers


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = field(default="noul", init=False)

    def certainty(self) -> float:
        """Distance from 0.5 rescaled to 0..1. Convenience only; not an official field."""
        return min(1.0, max(0.0, abs(self.noul - 0.5) * 2))

    @property
    def is_yes(self) -> bool:
        return self.noul >= 0.5

    def to_wire(self) -> dict[str, Any]:
        return {"type": "noul", "noul": self.noul}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = field(default="choice", init=False)

    def margin(self) -> float:
        """Top probability minus runner-up probability."""
        ps = sorted(self.probabilities.values(), reverse=True)
        if len(ps) < 2:
            return ps[0] if ps else 0.0
        return ps[0] - ps[1]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float
    type: str = field(default="score", init=False)

    @property
    def top_level(self) -> int:
        return int(max(self.probabilities.items(), key=lambda kv: kv[1])[0])

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def to_wire(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class Response:
    model: str
    answers: dict[str, Answer]
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]

    def to_wire(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: v.to_wire() for k, v in self.answers.items()},
            "usage": self.usage.to_wire(),
        }


# --------------------------------------------------------------------------- parsing


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{where} must be a number, got {value!r}")
    return float(value)


def _probs(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValidationError(f"{where} must be an object")
    return {str(k): _num(v, f"{where}[{k!r}]") for k, v in value.items()}


def answer_from_wire(question: Question, payload: Any, where: str = "answer") -> Answer:
    if not isinstance(payload, dict):
        raise ValidationError(f"{where} must be an object")
    kind = payload.get("type")
    if kind != question.type:
        raise ValidationError(f"{where}: expected type {question.type!r}, got {kind!r}")
    if kind == "noul":
        return NoulAnswer(noul=_num(payload.get("noul"), f"{where}.noul"))
    if kind == "choice":
        choice = payload.get("choice")
        if not isinstance(choice, str):
            raise ValidationError(f"{where}.choice must be a string")
        return ChoiceAnswer(
            choice=choice,
            probabilities=_probs(payload.get("probabilities"), f"{where}.probabilities"),
            confidence=_num(payload.get("confidence"), f"{where}.confidence"),
        )
    if kind == "score":
        legend = payload.get("legend")
        if not isinstance(legend, dict):
            raise ValidationError(f"{where}.legend must be an object")
        return ScoreAnswer(
            score=_num(payload.get("score"), f"{where}.score"),
            legend={str(k): v for k, v in legend.items()},
            probabilities=_probs(payload.get("probabilities"), f"{where}.probabilities"),
            confidence=_num(payload.get("confidence"), f"{where}.confidence"),
        )
    raise ValidationError(f"{where}: unknown answer type {kind!r}")


def from_wire(
    questions: dict[str, Question], payload: Any, latency_ms: float = 0.0
) -> Response:
    """Parse a System One response body against the questions that produced it."""
    if not isinstance(payload, dict):
        raise ValidationError("response body must be an object")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        raise ValidationError("response.answers must be an object")
    answers: dict[str, Answer] = {}
    for qid, q in questions.items():
        if qid not in raw_answers:
            raise ValidationError(f"response is missing answer for question {qid!r}")
        answers[qid] = answer_from_wire(q, raw_answers[qid], where=f"answers[{qid!r}]")
    usage_raw = payload.get("usage") or {}
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
        output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
    )
    model = payload.get("model")
    return Response(
        model=str(model) if model is not None else "unknown",
        answers=answers,
        usage=usage,
        latency_ms=latency_ms,
    )
