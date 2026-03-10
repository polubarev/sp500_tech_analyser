from __future__ import annotations

import json

from sp500_tech_analyser import cli


def test_cli_emits_logs_to_stderr_and_result_json_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "refresh_raw_snapshots",
        lambda config: {
            "downloaded": 2,
            "skipped": 3,
            "total_seen": 5,
            "bootstrapped_from_legacy_predictions": 0,
        },
    )

    exit_code = cli.main(["--log-level", "INFO", "refresh-raw"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Command started: refresh-raw" in captured.err
    assert "Command finished: refresh-raw" in captured.err
    assert json.loads(captured.out)["downloaded"] == 2
