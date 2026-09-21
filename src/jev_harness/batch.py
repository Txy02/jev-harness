"""Evaluate the same questions against many states concurrently."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from .types import Question, Response, StateType

Result = Response | BaseException


async def abatch(
    backend: Any,
    states: Iterable[StateType],
    questions: dict[str, Question],
    *,
    concurrency: int = 8,
    model: str | None = None,
) -> list[Result]:
    """Run `backend.aevaluate` for each state. Order is preserved.

    A failing state yields its exception in place instead of aborting the batch.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    sem = asyncio.Semaphore(concurrency)

    async def one(state: StateType) -> Result:
        async with sem:
            try:
                return await backend.aevaluate(state, questions, model=model)
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                return exc

    return await asyncio.gather(*(one(s) for s in states))


def batch(
    backend: Any,
    states: Iterable[StateType],
    questions: dict[str, Question],
    *,
    concurrency: int = 8,
    model: str | None = None,
) -> list[Result]:
    """Synchronous wrapper around `abatch`."""
    return asyncio.run(
        abatch(backend, list(states), questions, concurrency=concurrency, model=model)
    )


def successes(results: list[Result]) -> list[Response]:
    return [r for r in results if isinstance(r, Response)]


def failures(results: list[Result]) -> list[tuple[int, BaseException]]:
    return [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
