"""Command line: `jev-harness ask ...` and `jev-harness replay ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .backends import JevBackend, MockBackend
from .errors import JevError
from .gate import Gate
from .trace import JsonlTrace, Traced, replay
from .types import Choice, Noul, Question, Score, StateType


def _load_state(raw: str) -> StateType:
    if raw.startswith("@"):
        text = Path(raw[1:]).read_text(encoding="utf-8")
        if raw.endswith(".json"):
            return json.loads(text)
        return text
    return raw


def _parse_questions(args: argparse.Namespace) -> dict[str, Question]:
    qs: dict[str, Question] = {}
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}{counter}"

    for spec in args.noul or []:
        qid, _, text = spec.partition("=")
        if not text:
            qid, text = next_id("noul"), spec
        qs[qid] = Noul(text)
    for spec in args.choice or []:
        head, sep, opts = spec.partition("::")
        if not sep:
            raise SystemExit("--choice needs the form [id=]question::opt1,opt2,...")
        qid, _, text = head.partition("=")
        if not text:
            qid, text = next_id("choice"), head
        qs[qid] = Choice(text, {o.strip(): None for o in opts.split(",") if o.strip()})
    for spec in args.score or []:
        head, sep, levels = spec.partition("::")
        if not sep:
            raise SystemExit("--score needs the form [id=]question::level1,level2,...")
        qid, _, text = head.partition("=")
        if not text:
            qid, text = next_id("score"), head
        qs[qid] = Score(text, [lv.strip() for lv in levels.split(",") if lv.strip()])
    if not qs:
        raise SystemExit("give at least one --noul / --choice / --score")
    return qs


def _print_table(resp: Any, gate: Gate) -> None:
    decisions = gate.decide_all(resp)
    print(f"model: {resp.model}   usage: {resp.usage.to_wire()}   latency: {resp.latency_ms:.0f}ms")
    for qid, ans in resp.answers.items():
        d = decisions[qid].value
        if ans.type == "noul":
            print(f"  {qid:<20} noul   {ans.noul:.3f}                      [{d}]")
        elif ans.type == "choice":
            print(f"  {qid:<20} choice {ans.choice!s:<20} conf={ans.confidence:.2f} [{d}]")
        else:
            print(
                f"  {qid:<20} score  {ans.score:.2f}                 "
                f"conf={ans.confidence:.2f} [{d}]"
            )


def cmd_ask(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    questions = _parse_questions(args)
    backend: Any = MockBackend() if args.mock else JevBackend(model=args.model)
    if args.trace:
        backend = Traced(backend, JsonlTrace(args.trace))
    try:
        resp = backend.evaluate(state, questions, model=args.model)
    except JevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(resp.to_wire(), ensure_ascii=False, indent=2))
    else:
        _print_table(resp, Gate(act_at=args.act_at, reject_below=args.reject_below))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    records = replay(args.path)
    if args.last:
        records = records[-args.last :]
    for rec in records:
        if args.json:
            print(rec.to_json())
        else:
            status = "ERR " if rec.error else "ok  "
            print(
                f"{status} {rec.ts} {rec.backend:<12} {rec.model:<12} "
                f"{rec.latency_ms:7.0f}ms q={','.join(rec.question_ids)} "
                f"digest={rec.state_digest}"
            )
            if rec.error:
                print(f"     {rec.error}")
    print(f"{len(records)} record(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev-harness", description="Ask Jev typed questions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="evaluate questions against a state")
    ask.add_argument("--state", required=True, help="text, or @file.txt / @file.json")
    ask.add_argument("--noul", action="append", help="[id=]question")
    ask.add_argument("--choice", action="append", help="[id=]question::opt1,opt2")
    ask.add_argument("--score", action="append", help="[id=]question::level1,level2")
    ask.add_argument("--model", default=None)
    ask.add_argument("--mock", action="store_true", help="use MockBackend (no API call)")
    ask.add_argument("--trace", default=None, help="append a JSONL trace record to this file")
    ask.add_argument("--json", action="store_true", help="print raw JSON response")
    ask.add_argument("--act-at", type=float, default=0.8)
    ask.add_argument("--reject-below", type=float, default=0.3)
    ask.set_defaults(fn=cmd_ask)

    rp = sub.add_parser("replay", help="print records from a JSONL trace")
    rp.add_argument("path")
    rp.add_argument("--last", type=int, default=0)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_replay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
