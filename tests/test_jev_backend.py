import json

import httpx
import pytest
import respx

from jev_harness.backends import JevBackend, RetryPolicy
from jev_harness.errors import JevAPIError, MissingAPIKeyError, RequestTooLargeError
from jev_harness.types import Choice, Noul

URL = "https://api.typesafe.ai/v1/systemone"


def _ok_payload():
    return {
        "model": "jev-1.13.0",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }


def _backend(**kw):
    slept = []
    b = JevBackend(api_key="test-key", sleep=slept.append, **kw)
    b._slept = slept
    return b


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        JevBackend()


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    assert JevBackend().api_key == "env-key"


@respx.mock
def test_request_body_and_headers():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_payload()))
    b = _backend()
    r = b.evaluate(
        "Help! My payouts have been failing for 3 days.",
        {"is_urgent": Noul("Does this convey urgency?")},
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
    assert r.model == "jev-1.13.0"
    assert r.nouls["is_urgent"].noul == 0.95
    assert r.usage.input_tokens == 296
    assert r.latency_ms >= 0


@respx.mock
def test_model_override_per_call():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_payload()))
    _backend(model="jev-1.13.0").evaluate("s", {"is_urgent": Noul("q")}, model="jev-preview")
    assert json.loads(route.calls.last.request.content)["model"] == "jev-preview"


@respx.mock
def test_retries_on_429_then_succeeds():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(529, json={"error": "overloaded"}, headers={"retry-after": "2"}),
            httpx.Response(200, json=_ok_payload()),
        ]
    )
    b = _backend(retry=RetryPolicy(max_attempts=4, base_delay=0.5))
    r = b.evaluate("s", {"is_urgent": Noul("q")})
    assert r.nouls["is_urgent"].noul == 0.95
    assert route.call_count == 3
    assert b._slept == [0.5, 2.0]  # backoff, then retry-after honoured


@respx.mock
def test_gives_up_after_max_attempts():
    respx.post(URL).mock(return_value=httpx.Response(429, json={"error": "rl"}))
    b = _backend(retry=RetryPolicy(max_attempts=2))
    with pytest.raises(JevAPIError) as ei:
        b.evaluate("s", {"is_urgent": Noul("q")})
    assert ei.value.status == 429
    assert len(b._slept) == 1


@respx.mock
def test_422_raises_with_body():
    respx.post(URL).mock(
        return_value=httpx.Response(422, json={"detail": "questions.x.criteria missing"})
    )
    with pytest.raises(JevAPIError) as ei:
        _backend().evaluate("s", {"x": Choice("q", {"a": None})})
    assert ei.value.status == 422
    assert "criteria" in json.dumps(ei.value.body)


@respx.mock
def test_401_not_retried():
    route = respx.post(URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    with pytest.raises(JevAPIError) as ei:
        _backend().evaluate("s", {"x": Noul("q")})
    assert ei.value.status == 401
    assert route.call_count == 1


def test_request_too_large():
    b = _backend(max_request_bytes=500)
    with pytest.raises(RequestTooLargeError):
        b.evaluate("x" * 1000, {"q": Noul("q")})


@respx.mock
async def test_async_evaluate_and_retry():
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=_ok_payload())]
    )
    slept = []

    async def asleep(d):
        slept.append(d)

    b = JevBackend(api_key="k", asleep=asleep)
    r = await b.aevaluate("s", {"is_urgent": Noul("q")})
    assert r.nouls["is_urgent"].noul == 0.95
    assert route.call_count == 2
    assert slept == [0.5]
    await b.aclose()


def test_retry_policy_delays():
    p = RetryPolicy(base_delay=0.5, max_delay=3.0)
    assert p.delay_for(1) == 0.5
    assert p.delay_for(2) == 1.0
    assert p.delay_for(4) == 3.0
    assert p.delay_for(1, retry_after=10) == 3.0


@respx.mock
def test_base_url_override_and_context_manager():
    route = respx.post("https://proxy.example/v1/systemone").mock(
        return_value=httpx.Response(200, json=_ok_payload())
    )
    with JevBackend(api_key="k", base_url="https://proxy.example/") as b:
        b.evaluate("s", {"is_urgent": Noul("q")})
    assert route.called
