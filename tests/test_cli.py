import json

from jev_harness.cli import main
from jev_harness.trace import replay


def test_ask_mock_json(capsys):
    rc = main(
        [
            "ask",
            "--mock",
            "--json",
            "--state",
            "I was charged twice",
            "--noul",
            "billing=Is this about billing?",
            "--choice",
            "tone=Tone?::calm,angry",
            "--score",
            "How urgent?::low,high",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "mock-jev"
    assert set(out["answers"]) == {"billing", "tone", "score1"}
    assert out["answers"]["tone"]["type"] == "choice"


def test_ask_mock_table_and_trace(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    rc = main(
        ["ask", "--mock", "--state", "hi", "--noul", "Is it a greeting?", "--trace", str(trace)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "model: mock-jev" in out
    assert "noul1" in out and "[reject]" in out  # uniform 0.5 -> certainty 0 -> reject
    assert len(replay(trace)) == 1


def test_ask_state_from_json_file(tmp_path, capsys):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"msg": "hello"}), encoding="utf-8")
    rc = main(["ask", "--mock", "--json", "--state", f"@{f}", "--noul", "q"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["answers"]["noul1"]["noul"] == 0.5


def test_replay_prints_records(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    for _ in range(3):
        main(["ask", "--mock", "--state", "hi", "--noul", "q", "--trace", str(trace)])
    capsys.readouterr()
    rc = main(["replay", str(trace), "--last", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 record(s)" in out
    assert out.count("ok  ") == 2


def test_ask_requires_questions(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["ask", "--mock", "--state", "hi"])
