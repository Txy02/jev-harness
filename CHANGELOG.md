# Changelog

## 0.1.0 — 2026-09-21

Initial release.

- Typed `Noul` / `Choice` / `Score` questions and answers matching the System One wire format.
- `JevBackend` (httpx, sync + async, exponential backoff on 429/529, retry-after support, local size guard).
- `MockBackend` with rules, scripted answers and call recording.
- `NimbleBackend` adapter for Bespoke Nimble scorers (fake-scorer tested only).
- `Gate` confidence gating with per-question overrides.
- `JsonlTrace` / `Traced` / `replay` tracing.
- `batch` / `abatch` concurrent evaluation.
- `jev-harness ask|replay` CLI.
