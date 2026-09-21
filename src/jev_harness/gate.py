"""Confidence gating: turn an answer into ACT / REVIEW / REJECT.

Implements TypeSafe's "confidence-gated routing" pattern
(https://docs.typesafe.ai/patterns/confidence-routing): the answer says
*what*, the confidence says *whether* to act on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .errors import ValidationError
from .types import Answer, ChoiceAnswer, NoulAnswer, Response, ScoreAnswer


class Decision(str, Enum):
    ACT = "act"
    REVIEW = "review"
    REJECT = "reject"


def confidence_of(answer: Answer) -> float:
    """Choice/Score carry `confidence`; Noul uses its certainty (|p-0.5|*2)."""
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return float(answer.confidence)
    if isinstance(answer, NoulAnswer):
        return answer.certainty()
    raise ValidationError(f"unsupported answer {answer!r}")


def _check(act_at: float, reject_below: float) -> None:
    for name, v in (("act_at", act_at), ("reject_below", reject_below)):
        if not 0.0 <= v <= 1.0:
            raise ValidationError(f"{name} must be within 0..1, got {v}")
    if reject_below > act_at:
        raise ValidationError("reject_below must be <= act_at")


@dataclass(frozen=True)
class Gate:
    """confidence >= act_at -> ACT; < reject_below -> REJECT; else REVIEW.

    `overrides` maps a question id to its own (act_at, reject_below).
    """

    act_at: float = 0.8
    reject_below: float = 0.3
    overrides: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self.act_at, self.reject_below)
        for qid, (a, r) in self.overrides.items():
            try:
                _check(a, r)
            except ValidationError as exc:
                raise ValidationError(f"override for {qid!r}: {exc}") from exc

    def thresholds(self, question_id: str | None = None) -> tuple[float, float]:
        if question_id is not None and question_id in self.overrides:
            return self.overrides[question_id]
        return self.act_at, self.reject_below

    def decide(self, answer: Answer, question_id: str | None = None) -> Decision:
        act_at, reject_below = self.thresholds(question_id)
        c = confidence_of(answer)
        if c >= act_at:
            return Decision.ACT
        if c < reject_below:
            return Decision.REJECT
        return Decision.REVIEW

    def decide_all(self, response: Response) -> dict[str, Decision]:
        return {qid: self.decide(a, qid) for qid, a in response.answers.items()}
