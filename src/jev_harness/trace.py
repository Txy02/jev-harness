"""JSONL tracing for every evaluate() call.

Records are append-only, one JSON object per line. They include a preview of
the state, so keep trace files private or pass a `redact` function.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Question, Response, StateType

PREVIEW_CHARS = 200


def state_digest(state: StateType) -> str:
    raw = json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def state_preview(state: StateType, chars: int = PREVIEW_CHARS) -> str:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return text[:chars]


@dataclass
class TraceRecord:
    ts: str
    backend: str
    model: str
    question_ids: list[str]
    questions: dict[str, Any]
    answers: dict[str, Any]
    usage: dict[str, int]
    latency_ms: float
    state_digest: str
    state_preview: str
    requested_model: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> TraceRecord:
        data = json.loads(line)
        return cls(**data)


Redactor = Callable[[TraceRecord], TraceRecord]


class JsonlTrace:
    """Append trace records to a JSONL file."""

    def __init__(self, path: str | Path, *, redact: Redactor | None = None) -> None:
        self.path = Path(path)
        self.redact = redact
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        backend: str,
        state: StateType,
        questions: dict[str, Question],
        response: Response | None,
        requested_model: str | None = None,
        error: BaseException | None = None,
        latency_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> TraceRecord:
        rec = TraceRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            backend=backend,
            model=response.model if response else "",
            question_ids=list(questions),
            questions={k: q.to_wire() for k, q in questions.items()},
            answers={k: a.to_wire() for k, a in response.answers.items()} if response else {},
            usage=response.usage.to_wire() if response else {},
            latency_ms=(
                response.latency_ms if response and latency_ms is None else (latency_ms or 0.0)
            ),
            state_digest=state_digest(state),
            state_preview=state_preview(state),
            requested_model=requested_model,
            error=f"{type(error).__name__}: {error}" if error else None,
            extra=dict(extra or {}),
        )
        if self.redact is not None:
            rec = self.redact(rec)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(rec.to_json() + "\n")
        return rec


def no_preview(rec: TraceRecord) -> TraceRecord:
    """A ready-made redactor that drops the state preview."""
    rec.state_preview = ""
    return rec


def replay(path: str | Path) -> list[TraceRecord]:
    """Read a trace file back. Read-only inspection, not re-execution."""
    out: list[TraceRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TraceRecord.from_json(line))
    return out


class Traced:
    """Wrap any Backend so every call is recorded, including failures."""

    def __init__(self, backend: Any, trace: JsonlTrace) -> None:
        self.backend = backend
        self.trace = trace
        self.name = f"traced({getattr(backend, 'name', type(backend).__name__)})"

    def evaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        started = time.perf_counter()
        try:
            resp = self.backend.evaluate(state, questions, model=model)
        except BaseException as exc:
            self.trace.record(
                backend=self.backend.name,
                state=state,
                questions=questions,
                response=None,
                requested_model=model,
                error=exc,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        self.trace.record(
            backend=self.backend.name,
            state=state,
            questions=questions,
            response=resp,
            requested_model=model,
        )
        return resp

    async def aevaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        started = time.perf_counter()
        try:
            resp = await self.backend.aevaluate(state, questions, model=model)
        except BaseException as exc:
            self.trace.record(
                backend=self.backend.name,
                state=state,
                questions=questions,
                response=None,
                requested_model=model,
                error=exc,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        self.trace.record(
            backend=self.backend.name,
            state=state,
            questions=questions,
            response=resp,
            requested_model=model,
        )
        return resp
