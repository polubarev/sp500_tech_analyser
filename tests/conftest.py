from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sp500_tech_analyser.config import AppConfig
from sp500_tech_analyser.storage import format_utc_timestamp


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        gcs_project="test-project",
        gcs_bucket="test-bucket",
        data_root=tmp_path / "data",
        benchmark_ticker="^GSPC",
        raw_gcs_prefix="data/raw/investtech",
        investtech_url="https://example.com/investtech",
        legacy_predictions_dir=tmp_path / "legacy_predictions",
    )


def make_snapshot_payload(
    snapshot_at: datetime,
    short_score: int,
    medium_score: int,
    long_score: int,
) -> dict:
    if snapshot_at.tzinfo is None:
        snapshot_at = snapshot_at.replace(tzinfo=timezone.utc)
    else:
        snapshot_at = snapshot_at.astimezone(timezone.utc)

    return {
        "datetime": format_utc_timestamp(snapshot_at),
        "short_term": {
            "analysis": "Short analysis",
            "special": "Short special",
            "conclusion": "Short conclusion",
            "recommendation": "Short rec",
            "score": short_score,
        },
        "medium_term": {
            "analysis": "Medium analysis",
            "special": "Medium special",
            "conclusion": "Medium conclusion",
            "recommendation": "Medium rec",
            "score": medium_score,
        },
        "long_term": {
            "analysis": "Long analysis",
            "special": None,
            "conclusion": "Long conclusion",
            "recommendation": "Long rec",
            "score": long_score,
        },
    }


def write_raw_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_market_sessions(start: str = "2024-01-01", periods: int = 200) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    closes = 100.0 + pd.Series(range(periods), dtype=float)
    return pd.DataFrame({"session_date": dates.date, "close": closes})
