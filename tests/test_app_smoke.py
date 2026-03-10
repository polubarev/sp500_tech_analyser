from __future__ import annotations

import importlib
import sys
from datetime import timedelta, timezone

import pandas as pd
import requests

from sp500_tech_analyser.pipeline import build_processed_artifacts
from tests.conftest import make_market_sessions, make_snapshot_payload, write_raw_snapshot


class StreamlitStub:
    def __init__(self):
        self.calls = []

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, value):
        self.calls.append(("title", value))

    def caption(self, value):
        self.calls.append(("caption", value))

    def subheader(self, value):
        self.calls.append(("subheader", value))

    def info(self, value):
        self.calls.append(("info", value))

    def warning(self, value):
        self.calls.append(("warning", value))

    def error(self, value):
        self.calls.append(("error", value))

    def dataframe(self, value, **kwargs):
        self.calls.append(("dataframe", len(value)))

    def write(self, value):
        self.calls.append(("write", value))

    def selectbox(self, label, options, key=None):
        self.calls.append(("selectbox", label))
        return options[0]

    def pyplot(self, figure):
        self.calls.append(("pyplot", figure is not None))


def _boom(*args, **kwargs):
    raise AssertionError("unexpected network activity")


def test_app_renders_from_processed_artifacts_without_network(app_config, monkeypatch):
    market_sessions = make_market_sessions(periods=180)
    snapshot_times = pd.bdate_range("2024-01-10", periods=26)
    for index, snapshot_at in enumerate(snapshot_times):
        payload = make_snapshot_payload(
            snapshot_at.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=13),
            short_score=index - 13,
            medium_score=index - 8,
            long_score=index - 3,
        )
        write_raw_snapshot(app_config.raw_dir / f"investtech_{index:03d}.json", payload)

    build_processed_artifacts(app_config, market_sessions=market_sessions)

    monkeypatch.setenv("SP500_TECH_DATA_ROOT", str(app_config.data_root))
    monkeypatch.setattr(requests, "get", _boom)

    sys.modules.pop("sp500_tech_analyser.dashboard", None)
    app = importlib.import_module("sp500_tech_analyser.dashboard")
    stub = StreamlitStub()
    monkeypatch.setattr(app, "st", stub)
    monkeypatch.setattr(app, "plot_strategy_curves", lambda *args, **kwargs: object())
    monkeypatch.setattr(app, "plot_calibration", lambda *args, **kwargs: object())
    monkeypatch.setattr(app, "plot_threshold_history", lambda *args, **kwargs: object())

    app.main()

    assert ("title", "Investtech Decision Dashboard") in stub.calls
    assert not any(call[0] == "error" for call in stub.calls)
    assert any(call[0] == "pyplot" for call in stub.calls)
