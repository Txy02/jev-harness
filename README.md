# jev-harness

Typed decisions with [TypeSafe's Jev](https://typesafe.ai) System One model, in Python.

Jev is not an LLM. You send it a `state` and a map of typed questions, and it returns typed answers with calibrated probabilities. No text generation, nothing to parse. `jev-harness` wraps that contract with:

- **Typed questions and answers** — `Noul` (yes/no probability), `Choice` (one of up to 255 options), `Score` (2–10 ordered levels), validated locally before any request.
- **Pluggable backends** — `JevBackend` (HTTP, retries with backoff on 429/529), `MockBackend` (offline, scriptable, for tests), `NimbleBackend` (adapter for the open [Bespoke Nimble](https://github.com/bespokelabsai/nimble) scorer).
- **Confidence gating** — `Gate` turns any answer into `ACT / REVIEW / REJECT`, with per-question overrides.
- **Tracing** — `Traced(backend, JsonlTrace(path))` records every call, including failures and the real model version that answered.
- **Batching** — `batch()` / `abatch()` run the same questions over many states concurrently.
- **CLI** — `jev-harness ask` and `jev-harness replay`.

> Community project. Not affiliated with TypeSafe AI. Contains no model weights.

## Install

```bash
pip install -e .          # from this directory
export TYPESAFE_API_KEY=... # create one at https://console.typesafe.ai
```

Python ≥ 3.10. Only runtime dependency: `httpx`.

## 60-second example

```python
from jev_harness import JevBackend, Noul, Choice, Score, Gate

backend = JevBackend()  # reads TYPESAFE_API_KEY

resp = backend.evaluate(
    state={
        "ticket": "I was charged twice for order A-104. Please refund the duplicate.",
        "policy": "Duplicate charges are eligible for a refund.",
    },
    questions={
        "refund_requested": Noul("Does `ticket` request a refund?"),
        "policy_allows": Noul("Does `policy` support the refund requested in `ticket`?"),
        "department": Choice(
            "Which team should handle `ticket`?",
            {"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": "Pricing"},
        ),
        "urgency": Score("How urgent is `ticket`?", ["can wait", "this week", "today"]),
    },
)

print(resp.model)                                # e.g. jev-1.13.0
print(resp.nouls["refund_requested"].noul)       # 0.97
print(resp.choices["department"].choice)         # billing
print(resp.scores["urgency"].score)              # 1.6  (between "this week" and "today")

gate = Gate(act_at=0.8, reject_below=0.3)
print(gate.decide_all(resp))                     # {'refund_requested': ACT, ...}
```

Everything in one request is evaluated in parallel and independently. Ask every question you might need; extra questions cost only their tokens.

## Backends

```python
from jev_harness import JevBackend, MockBackend, NimbleBackend, RetryPolicy

# Real API. Retries 429/529 with exponential backoff, honours retry-after.
JevBackend(model="jev-1.13.0", retry=RetryPolicy(max_attempts=5), timeout=30)

# Offline. Unmatched questions get uniform answers; add rules or script calls.
mock = MockBackend().add_rule("refund_requested", NoulAnswer(0.95))
mock.script({"department": choice_answer("billing", ["billing", "technical", "sales"])})

# Local Nimble scorer (Choice -> enum, Noul -> boolean; Score unsupported).
NimbleBackend(scorer)  # scorer = nimble.scoring.parallel_scorer.ParallelScorer(...)
```

Every backend implements `evaluate(state, questions, *, model=None)` and `aevaluate(...)`.

`JevBackend` also applies a coarse local size guard (`max_request_bytes`, default 200 kB). It is not a tokenizer; the API's real limits are 64k tokens per request and 32k for state + the longest question.

## Gating

```python
from jev_harness import Gate, Decision

gate = Gate(act_at=0.8, reject_below=0.3, overrides={"refund_requested": (0.95, 0.5)})
match gate.decide(resp.choices["department"], "department"):
    case Decision.ACT:    route_ticket(resp.choices["department"].choice)
    case Decision.REVIEW: queue_for_human(resp)
    case Decision.REJECT: ask_clarifying_question()
```

Choice and Score use the API's `confidence`. Noul has no confidence field, so the gate uses `|p − 0.5| × 2` (a strong *no* is just as actionable as a strong *yes*).

## Tracing

```python
from jev_harness import Traced, JsonlTrace, replay, no_preview

backend = Traced(JevBackend(), JsonlTrace("runs/today.jsonl", redact=no_preview))
...
for rec in replay("runs/today.jsonl"):
    print(rec.ts, rec.model, rec.latency_ms, rec.answers)
```

Records include a state preview by default. Keep trace files private or pass `redact`.

## Batching

```python
from jev_harness import batch, successes, failures

results = batch(backend, tickets, questions, concurrency=8)
ok = successes(results)          # list[Response], order preserved
bad = failures(results)          # list[(index, exception)]
```

## CLI

```bash
jev-harness ask --state "My payouts have been failing for 3 days" \
  --noul "urgent=Does this convey urgency?" \
  --choice "team=Which team?::billing,technical,sales" \
  --score "anger=How angry?::calm,frustrated,furious" \
  --trace runs/cli.jsonl

jev-harness ask --mock --json --state @ticket.json --noul "Is it about billing?"
jev-harness replay runs/cli.jsonl --last 5
```

## Testing

```bash
pip install -e ".[dev]"
pytest                          # offline, uses MockBackend + respx
TYPESAFE_API_KEY=... pytest -m live -v   # real API smoke test
```

Live results are recorded in `docs/live-validation.md`.

## Notes and limits

- Jev's primary language is English; CJK input is accepted with lower accuracy.
- `jev-latest` moves when a new version ships. Log the response's `model` (the trace does) and pin a version once you have tuned thresholds.
- Rate limits are adjusting dynamically during launch. Tune `RetryPolicy` and `batch(concurrency=...)`.
- The Nimble adapter is tested against a fake scorer only; it has not been validated on a real Nimble checkpoint here.

## Related

- [TypeSafe docs](https://docs.typesafe.ai) · [API reference](https://docs.typesafe.ai/api) · [Patterns](https://docs.typesafe.ai/patterns)
- [`jevrag`](../jevrag) — Jev-powered reranking, routing and citation checking for RAG.
- [`jev-agent`](../jev-agent) — Jev decides, an LLM generates: a gated agent loop.

MIT.
