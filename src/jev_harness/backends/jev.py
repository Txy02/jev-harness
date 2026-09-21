"""JevBackend: HTTP client for POST https://api.typesafe.ai/v1/systemone."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..errors import JevAPIError, MissingAPIKeyError, RequestTooLargeError
from ..types import Question, Response, StateType, from_wire, questions_to_wire
from .base import RetryPolicy

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
ENDPOINT = "/v1/systemone"
ENV_API_KEY = "TYPESAFE_API_KEY"
ENV_BASE_URL = "TYPESAFE_BASE_URL"
ENV_MODEL = "TYPESAFE_MODEL"


def _retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _error_body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


class JevBackend:
    """Synchronous + asynchronous client with exponential backoff on 429/529.

    The `max_request_bytes` guard is a coarse local check on serialized size,
    not a tokenizer. The API enforces 64k tokens per request and 32k for
    state + the longest question.
    """

    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        retry: RetryPolicy | None = None,
        max_request_bytes: int = 200_000,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        key = api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise MissingAPIKeyError(
                f"pass api_key=... or set {ENV_API_KEY} (create one at https://console.typesafe.ai)"
            )
        self.api_key = key
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        self.max_request_bytes = max_request_bytes
        self._client = client
        self._async_client = async_client
        self._sleep = sleep
        self._asleep = asleep

    # ------------------------------------------------------------------ helpers

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "jev-harness/0.1",
        }

    @property
    def url(self) -> str:
        return self.base_url + ENDPOINT

    def build_body(
        self, state: StateType, questions: dict[str, Question], model: str | None = None
    ) -> dict[str, Any]:
        body = {
            "state": state,
            "model": model or self.model,
            "questions": questions_to_wire(questions),
        }
        size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        if size > self.max_request_bytes:
            raise RequestTooLargeError(
                f"request is {size} bytes, over the local budget of {self.max_request_bytes}; "
                "split the state or reduce questions"
            )
        return body

    def _should_retry(self, status: int, attempt: int) -> bool:
        return status in self.retry.retry_statuses and attempt < self.retry.max_attempts

    def _handle(
        self, resp: httpx.Response, questions: dict[str, Question], started: float
    ) -> Response:
        if resp.status_code >= 400:
            raise JevAPIError(resp.status_code, _error_body(resp))
        try:
            payload = resp.json()
        except ValueError as exc:
            raise JevAPIError(resp.status_code, resp.text, "response was not JSON") from exc
        return from_wire(questions, payload, latency_ms=(time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ sync

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def evaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        body = self.build_body(state, questions, model)
        client = self._get_client()
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            resp = client.post(self.url, json=body, headers=self._headers)
            if self._should_retry(resp.status_code, attempt):
                self._sleep(self.retry.delay_for(attempt, _retry_after(resp.headers)))
                continue
            return self._handle(resp, questions, started)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> JevBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ async

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    async def aevaluate(
        self, state: StateType, questions: dict[str, Question], *, model: str | None = None
    ) -> Response:
        body = self.build_body(state, questions, model)
        client = self._get_async_client()
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            resp = await client.post(self.url, json=body, headers=self._headers)
            if self._should_retry(resp.status_code, attempt):
                await self._asleep(self.retry.delay_for(attempt, _retry_after(resp.headers)))
                continue
            return self._handle(resp, questions, started)

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def __aenter__(self) -> JevBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
