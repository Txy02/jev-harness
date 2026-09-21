"""Error hierarchy for jev-harness."""

from __future__ import annotations


class JevError(Exception):
    """Base class for every error raised by jev-harness."""


class ValidationError(JevError, ValueError):
    """A question, answer, or configuration value failed local validation."""


class MissingAPIKeyError(JevError):
    """No API key was supplied and TYPESAFE_API_KEY is not set."""


class JevAPIError(JevError):
    """The TypeSafe API returned a non-retryable error status."""

    def __init__(self, status: int, body: object, message: str | None = None):
        self.status = status
        self.body = body
        super().__init__(message or f"TypeSafe API error {status}: {body!r}")


class RequestTooLargeError(JevError):
    """The serialized request exceeds the local byte budget.

    This guard is a coarse size check, not a tokenizer. The real limit is
    enforced server-side in tokens (64k total, 32k for state + longest question).
    """


class UnsupportedQuestionError(JevError):
    """The selected backend cannot evaluate this question type."""
