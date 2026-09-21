"""MockBackend: deterministic, offline answers for tests and demos."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Response,
    Score,
    ScoreAnswer,
    StateType,
    Usage,
    questions_to_wire,
)

Matcher = str | Callable[[str, Question], bool]
AnswerFactory = Answer | Callable[[StateType, str, Question], Answer]


@dataclass(frozen=True)
class Rule:
    """Match a question (by id or predicate) and produce an answer."""

    match: Matcher
    answer: AnswerFactory

    def matches(self, qid: str, question: Question) -> bool:
        if isinstance(self.match, str):
            return self.match == qid
        return bool(self.match(qid, question))

    def produce(self, state: StateType, qid: str, question: Question) -> Answer:
        if callable(self.answer):
            return self.answer(state, qid, question)
        return self.answer


def uniform_answer(question: Question) -> Answer:
    """The answer a maximally uncertain model would give."""
    if isinstance(question, Noul):
        return NoulAnswer(0.5)
    if isinstance(question, Choice):
        opts = question.options
        p = 1.0 / len(opts)
        return ChoiceAnswer(choice=opts[0], probabilities={o: p for o in opts}, confidence=0.0)
    if isinstance(question, Score):
        n = len(question.levels)
        p = 1.0 / n
        return ScoreAnswer(
            score=(n - 1) / 2,
            legend={str(i): lvl for i, lvl in enumerate(question.levels)},
            probabilities={str(i): p for i in range(n)},
            confidence=0.0,
        )
    raise TypeError(f"unsupported question {question!r}")


def choice_answer(choice: str, options: Iterable[str], confidence: float = 0.9) -> ChoiceAnswer:
    """Helper: a peaked distribution on `choice` for tests."""
    opts = list(options)
    rest = (1.0 - confidence) / max(1, len(opts) - 1)
    probs = {o: (confidence if o == choice else rest) for o in opts}
    return ChoiceAnswer(choice=choice, probabilities=probs, confidence=confidence)


def score_answer(level: int, levels: list[Any], confidence: float = 0.9) -> ScoreAnswer:
    n = len(levels)
    rest = (1.0 - confidence) / max(1, n - 1)
    probs = {str(i): (confidence if i == level else rest) for i in range(n)}
    score = sum(i * p for i, p in enumerate(probs.values()))
    return ScoreAnswer(
        score=score,
        legend={str(i): lvl for i, lvl in enumerate(levels)},
        probabilities=probs,
        confidence=confidence,
    )


@dataclass
class MockBackend:
    """Offline backend. Unmatched questions get uniform answers."""

    rules: list[Rule] = field(default_factory=list)
    model: str = "mock-jev"
    name: str = "mock"
    calls: list[dict[str, Any]] = field(default_factory=list)
    scripted: list[dict[str, Answer]] = field(default_factory=list)
    """If non-empty, each call pops the next dict of answers (by question id) before rules."""

    def add_rule(self, match: Matcher, answer: AnswerFactory) -> MockBackend:
        self.rules.append(Rule(match, answer))
        return self

    def script(self, *answer_sets: dict[str, Answer]) -> MockBackend:
        self.scripted.extend(answer_sets)
        return self

    def _answer_for(
        self, state: StateType, qid: str, q: Question, scripted: dict[str, Answer] | None
    ) -> Answer:
        if scripted and qid in scripted:
            return scripted[qid]
        for rule in self.rules:
            if rule.matches(qid, q):
                return rule.produce(state, qid, q)
        return uniform_answer(q)

    def evaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        questions_to_wire(questions)  # validates
        scripted = self.scripted.pop(0) if self.scripted else None
        answers = {qid: self._answer_for(state, qid, q, scripted) for qid, q in questions.items()}
        self.calls.append({"state": state, "questions": dict(questions), "model": model})
        return Response(
            model=model or self.model,
            answers=answers,
            usage=Usage(input_tokens=0, output_tokens=0),
            latency_ms=0.0,
        )

    async def aevaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        await asyncio.sleep(0)
        return self.evaluate(state, questions, model=model)
