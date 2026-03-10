from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from sp500_tech_analyser.constants import SIGNAL_MAPPINGS
from sp500_tech_analyser.evaluation import evaluate_signal_mapping
from sp500_tech_analyser.pipeline import build_processed_artifacts, build_signal_frame, normalize_raw_snapshots
from tests.conftest import make_market_sessions, make_snapshot_payload, write_raw_snapshot


def test_normalize_raw_snapshots_dedupes_and_converts_to_utc(app_config):
    first_payload = make_snapshot_payload(
        datetime(2024, 1, 10, 11, 0, tzinfo=timezone.utc),
        short_score=-10,
        medium_score=5,
        long_score=20,
    )
    duplicate_payload = {
        **make_snapshot_payload(
            datetime(2024, 1, 10, 11, 0, tzinfo=timezone.utc),
            short_score=42,
            medium_score=7,
            long_score=25,
        ),
        "datetime": "2024-01-10T13:00:00+02:00",
    }
    write_raw_snapshot(app_config.raw_dir / "investtech_dup_a.json", first_payload)
    write_raw_snapshot(app_config.raw_dir / "investtech_dup_b.json", duplicate_payload)

    snapshots = normalize_raw_snapshots(app_config)

    assert len(snapshots) == 3
    assert str(snapshots["snapshot_at"].dtype) == "datetime64[ns, UTC]"
    assert snapshots["snapshot_at"].nunique() == 1
    assert snapshots.loc[snapshots["term"] == "short", "score"].item() == 42


def test_build_signal_frame_only_uses_supported_signal_mappings(app_config):
    snapshot_times = [
        datetime(2024, 1, 10, 13, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 11, 13, 0, tzinfo=timezone.utc),
    ]
    for index, snapshot_at in enumerate(snapshot_times):
        payload = make_snapshot_payload(snapshot_at, short_score=10 + index, medium_score=20 + index, long_score=30 + index)
        write_raw_snapshot(app_config.raw_dir / f"investtech_{index}.json", payload)

    snapshots = normalize_raw_snapshots(app_config)
    signals = build_signal_frame(snapshots, make_market_sessions(periods=220))

    observed = set(zip(signals["signal_name"], signals["horizon_label"]))
    expected = {(mapping.signal_name, mapping.horizon_label) for mapping in SIGNAL_MAPPINGS}
    assert observed == expected
    assert "3w" not in set(signals["horizon_label"])


def test_walk_forward_threshold_selection_does_not_leak_future_information():
    rows = []
    base_time = pd.Timestamp("2024-01-01T13:00:00Z")
    for index in range(20):
        rows.append(
            {
                "snapshot_at": base_time + pd.Timedelta(days=index),
                "provider": "investtech",
                "term": "short",
                "signal_name": "short_score",
                "horizon_label": "1w",
                "score": 10,
                "forward_return": 0.02,
                "label_available": True,
            }
        )
    rows.append(
        {
            "snapshot_at": base_time + pd.Timedelta(days=20),
            "provider": "investtech",
            "term": "short",
            "signal_name": "short_score",
            "horizon_label": "1w",
            "score": 80,
            "forward_return": -0.80,
            "label_available": True,
        }
    )
    signals_df = pd.DataFrame(rows)

    evaluation_row, strategies_df, _, _ = evaluate_signal_mapping(
        signals_df=signals_df,
        mapping=next(mapping for mapping in SIGNAL_MAPPINGS if mapping.signal_name == "short_score" and mapping.horizon_label == "1w"),
        min_train_size=20,
    )

    assert evaluation_row["sample_count"] == 1
    assert strategies_df.iloc[0]["threshold"] == 0
    assert strategies_df.iloc[0]["position"] == 1


def test_build_processed_artifacts_warns_for_missing_long_labels_and_keeps_all_snapshots(app_config):
    market_sessions = make_market_sessions(periods=110)
    snapshot_times = pd.bdate_range("2024-01-10", periods=25)
    for index, snapshot_at in enumerate(snapshot_times):
        payload = make_snapshot_payload(
            snapshot_at.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=13),
            short_score=index - 12,
            medium_score=index - 6,
            long_score=index,
        )
        write_raw_snapshot(app_config.raw_dir / f"investtech_{index:03d}.json", payload)

    result = build_processed_artifacts(app_config, market_sessions=market_sessions)

    long_row = result["evaluation"][
        (result["evaluation"]["signal_name"] == "long_score")
        & (result["evaluation"]["horizon_label"] == "26w")
    ].iloc[0]
    assert result["summary"]["raw_snapshot_count"] == 25
    assert result["snapshots"]["snapshot_at"].nunique() == 25
    assert len(result["snapshots"]) == 75
    assert long_row["sample_count"] == 0
    assert any("long_score -> 26w" in warning for warning in result["summary"]["warnings"])


def test_dashboard_summary_guidance_keeps_horizon_label(app_config):
    market_sessions = make_market_sessions(periods=180)
    snapshot_times = pd.bdate_range("2024-01-10", periods=26)
    for index, snapshot_at in enumerate(snapshot_times):
        payload = make_snapshot_payload(
            snapshot_at.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=13),
            short_score=60,
            medium_score=40,
            long_score=20,
        )
        write_raw_snapshot(app_config.raw_dir / f"investtech_guidance_{index:03d}.json", payload)

    result = build_processed_artifacts(app_config, market_sessions=market_sessions)
    latest_short = next(
        row for row in result["summary"]["latest_signals"]
        if row["signal_name"] == "short_score" and row["horizon_label"] == "1w"
    )

    assert "1w signal" in latest_short["usage_guidance"]
    assert "None signal" not in result["summary"]["executive_message"]


def test_realistic_strategy_backtest_uses_non_overlapping_trades(app_config):
    market_sessions = make_market_sessions(periods=80)
    rows = []
    base_time = pd.Timestamp("2024-01-01T13:00:00Z")
    session_dates = market_sessions["session_date"].tolist()
    for index in range(24):
        base_session = session_dates[index]
        future_session = session_dates[index + 5]
        rows.append(
            {
                "snapshot_at": base_time + pd.Timedelta(days=index),
                "provider": "investtech",
                "term": "short",
                "signal_name": "short_score",
                "horizon_label": "1w",
                "score": 60,
                "recommendation": "Positive",
                "analysis": "n/a",
                "conclusion": "n/a",
                "special": None,
                "base_session_date": base_session,
                "future_session_date": future_session,
                "base_close": 100.0,
                "future_close": 105.0,
                "forward_return": 0.05,
                "label_available": True,
            }
        )
    signals_df = pd.DataFrame(rows)

    evaluation_row, strategies_df, _, _ = evaluate_signal_mapping(
        signals_df=signals_df,
        mapping=next(mapping for mapping in SIGNAL_MAPPINGS if mapping.signal_name == "short_score" and mapping.horizon_label == "1w"),
        market_sessions=market_sessions,
        min_train_size=20,
    )

    assert evaluation_row["sample_count"] == 4
    assert evaluation_row["executed_trade_count"] == 1
    assert len(strategies_df) == 1
    assert strategies_df.iloc[0]["trade_entry_date"] < strategies_df.iloc[0]["trade_exit_date"]
