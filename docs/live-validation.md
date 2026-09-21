# Live validation

| Date | Model (from response) | input_tokens | output_tokens | latency_ms | Result |
|---|---|---|---|---|---|
| 2026-09-21 | `jev-1.13.0` | 452 | 74 | 799 | pass |

State: a cancelled-flight refund ticket plus the refund policy (see `tests/live/test_live_smoke.py`).

| Question | Answer | Gate(act_at=0.6) |
|---|---|---|
| `refund_requested` | `{"type": "noul", "noul": 0.99}` | act |
| `request_type` | `{"type": "choice", "choice": "refund", "probabilities": {"rebooking": 0.0, "information": 0.0, "refund": 1.0}, "confidence": 1.0}` | act |
| `frustration` | `{"type": "score", "score": 0.27, "legend": {"0": "Calm and neutral.", "1": "Concerned but civil.", "2": "Very angry or using strong language."}, "probabilities": {"0": 0.73, "1": 0.27, "2": 0.0}, "confidence": 0.6}` | act |

Raw output: `docs/live-last-run.json`. Full trace: `docs/live-trace.jsonl` (not committed).
