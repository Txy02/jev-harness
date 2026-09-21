"""jev-harness: typed decisions with TypeSafe's Jev System One model.

    from jev_harness import JevBackend, Noul, Choice, Score, Gate

    backend = JevBackend()  # reads TYPESAFE_API_KEY
    resp = backend.evaluate(
        {"ticket": "I was charged twice. Please fix this ASAP."},
        {
            "billing": Noul("Is `ticket` about billing?"),
            "tone": Choice("Customer tone?", {"calm": None, "frustrated": None, "angry": None}),
            "urgency": Score("How urgent?", ["can wait", "this week", "today"]),
        },
    )
    print(resp.nouls["billing"].noul, resp.choices["tone"].choice, resp.scores["urgency"].score)
    print(Gate().decide_all(resp))
"""

from .backends import (
    Backend,
    JevBackend,
    MockBackend,
    NimbleBackend,
    RetryPolicy,
    Rule,
    choice_answer,
    score_answer,
    uniform_answer,
)
from .batch import abatch, batch, failures, successes
from .errors import (
    JevAPIError,
    JevError,
    MissingAPIKeyError,
    RequestTooLargeError,
    UnsupportedQuestionError,
    ValidationError,
)
from .gate import Decision, Gate, confidence_of
from .trace import JsonlTrace, Traced, TraceRecord, no_preview, replay
from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Response,
    Score,
    ScoreAnswer,
    Usage,
    from_wire,
    questions_to_wire,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "Noul", "Choice", "Score", "Question",
    "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "Answer",
    "Response", "Usage", "from_wire", "questions_to_wire",
    # backends
    "Backend", "JevBackend", "MockBackend", "NimbleBackend", "RetryPolicy", "Rule",
    "choice_answer", "score_answer", "uniform_answer",
    # gate
    "Gate", "Decision", "confidence_of",
    # trace
    "JsonlTrace", "Traced", "TraceRecord", "replay", "no_preview",
    # batch
    "abatch", "batch", "successes", "failures",
    # errors
    "JevError", "JevAPIError", "MissingAPIKeyError", "RequestTooLargeError",
    "UnsupportedQuestionError", "ValidationError",
]
