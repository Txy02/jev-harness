import asyncio

import pytest

from jev_harness.backends import MockBackend
from jev_harness.batch import abatch, batch, failures, successes
from jev_harness.errors import JevError
from jev_harness.types import Noul, NoulAnswer


def _backend():
    return MockBackend().add_rule(
        "q", lambda state, qid, q: NoulAnswer(float(state.split(":")[1]))
    )


def test_batch_preserves_order():
    states = [f"s:{i / 10}" for i in range(5)]
    results = batch(_backend(), states, {"q": Noul("x")}, concurrency=2)
    assert [r.nouls["q"].noul for r in results] == [0.0, 0.1, 0.2, 0.3, 0.4]


class Flaky(MockBackend):
    async def aevaluate(self, state, questions, *, model=None):
        if "bad" in state:
            raise JevError("boom")
        return await super().aevaluate(state, questions, model=model)


def test_failure_does_not_abort():
    results = batch(Flaky(), ["ok", "bad", "ok"], {"q": Noul("x")})
    assert len(successes(results)) == 2
    idx, exc = failures(results)[0]
    assert idx == 1 and isinstance(exc, JevError)


class Counting(MockBackend):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.peak = 0

    async def aevaluate(self, state, questions, *, model=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return self.evaluate(state, questions, model=model)


async def test_concurrency_bound():
    b = Counting()
    await abatch(b, [str(i) for i in range(20)], {"q": Noul("x")}, concurrency=3)
    assert b.peak == 3


def test_bad_concurrency():
    with pytest.raises(ValueError):
        batch(MockBackend(), ["a"], {"q": Noul("x")}, concurrency=0)
