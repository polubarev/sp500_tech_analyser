from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class SignalMapping:
    signal_name: str
    term: str
    horizon_label: str
    horizon_delta: timedelta


PROVIDER_NAME = "investtech"

RAW_TERM_KEYS = {
    "short": "short_term",
    "medium": "medium_term",
    "long": "long_term",
}

SIGNAL_MAPPINGS = (
    SignalMapping("short_score", "short", "1w", timedelta(weeks=1)),
    SignalMapping("short_score", "short", "2w", timedelta(weeks=2)),
    SignalMapping("short_score", "short", "4w", timedelta(weeks=4)),
    SignalMapping("medium_score", "medium", "13w", timedelta(weeks=13)),
    SignalMapping("long_score", "long", "26w", timedelta(weeks=26)),
)

CALIBRATION_BINS = [-100, -50, -10, 10, 50, 100]
CALIBRATION_BUCKETS = [
    "[-100, -50)",
    "[-50, -10)",
    "[-10, 10)",
    "[10, 50)",
    "[50, 100]",
]

SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "provider",
    "term",
    "score",
    "recommendation",
    "analysis",
    "conclusion",
    "special",
]

SIGNAL_COLUMNS = [
    "snapshot_at",
    "provider",
    "term",
    "signal_name",
    "horizon_label",
    "score",
    "recommendation",
    "analysis",
    "conclusion",
    "special",
    "base_session_date",
    "future_session_date",
    "base_close",
    "future_close",
    "forward_return",
    "label_available",
]

EVALUATION_COLUMNS = [
    "signal_name",
    "term",
    "horizon_label",
    "sample_count",
    "executed_trade_count",
    "labeled_observations",
    "total_observations",
    "coverage",
    "pearson_correlation",
    "directional_accuracy",
    "precision_up",
    "recall_up",
    "precision_down",
    "recall_down",
    "cumulative_strategy_return",
    "cumulative_buy_hold_return",
    "excess_return_vs_buy_hold",
    "max_drawdown",
    "latest_threshold",
    "verdict",
]

CALIBRATION_COLUMNS = [
    "signal_name",
    "term",
    "horizon_label",
    "bucket",
    "forward_return_average",
    "hit_rate",
    "sample_count",
]

STRATEGY_COLUMNS = [
    "snapshot_at",
    "provider",
    "term",
    "signal_name",
    "horizon_label",
    "score",
    "forward_return",
    "threshold",
    "position",
    "predicted_label",
    "true_label",
    "base_session_date",
    "future_session_date",
    "trade_entry_date",
    "trade_exit_date",
    "strategy_return",
    "benchmark_return",
    "cumulative_strategy_equity",
    "cumulative_buy_hold_equity",
]
