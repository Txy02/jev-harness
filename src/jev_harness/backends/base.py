"""Backend protocol and retry policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..types import Question, Response, StateType


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff for retryable statuses (429 rate limit, 529 overloaded)."""

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_statuses: tuple[int, ...] = (429, 529)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before the given (1-based) attempt's retry."""
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_delay)
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)


@runtime_checkable
class Backend(Protocol):
    """Anything that can evaluate typed questions against a state."""

    name: str

    def evaluate(
        self,
        state: StateType,
        questions: dict[str, Question],
        *,
        model: str | None = None,
    ) -> Response: ...

    async def aevaluate(
        self,
        state: StateType,
        questions: dict[str, Question],
        *,
        model: str | None = None,
    ) -> Response: ...
