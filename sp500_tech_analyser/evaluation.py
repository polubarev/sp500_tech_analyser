from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd

from .constants import (
    CALIBRATION_BINS,
    CALIBRATION_BUCKETS,
    CALIBRATION_COLUMNS,
    EVALUATION_COLUMNS,
    STRATEGY_COLUMNS,
    SignalMapping,
)


def classify_return(value: float) -> str:
    if pd.isna(value):
        return "Flat"
    if value > 0:
        return "Up"
    if value < 0:
        return "Down"
    return "Flat"


def classify_position(value: int) -> str:
    if value > 0:
        return "Up"
    if value < 0:
        return "Down"
    return "Flat"


def determine_position(score: float, threshold: int) -> int:
    if pd.isna(score):
        return 0
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def fit_optimal_threshold(train_df: pd.DataFrame, thresholds: Iterable[int]) -> dict:
    diagnostics = []
    for threshold in thresholds:
        positions = train_df["score"].apply(lambda score: determine_position(score, threshold))
        strategy_returns = positions * train_df["forward_return"]
        cumulative_return = float((1.0 + strategy_returns).prod() - 1.0)
        active_mask = positions != 0
        directional_accuracy = float(
            (positions[active_mask].map(classify_position) == train_df.loc[active_mask, "true_label"]).mean()
        ) if active_mask.any() else 0.0
        diagnostics.append(
            {
                "threshold": threshold,
                "cumulative_return": cumulative_return,
                "coverage": float(active_mask.mean()),
                "directional_accuracy": directional_accuracy,
            }
        )

    best = max(
        diagnostics,
        key=lambda row: (
            round(row["cumulative_return"], 12),
            round(row["directional_accuracy"], 12),
            -row["threshold"],
        ),
    )
    return best


def _precision_recall(df: pd.DataFrame, label: str) -> tuple[float, float]:
    predicted = df["predicted_label"] == label
    actual = df["true_label"] == label
    true_positive = int((predicted & actual).sum())
    precision = true_positive / int(predicted.sum()) if predicted.sum() else 0.0
    recall = true_positive / int(actual.sum()) if actual.sum() else 0.0
    return float(precision), float(recall)


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def _safe_pearson_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) <= 1 or len(right) <= 1:
        return 0.0
    left_clean = pd.Series(left, dtype=float)
    right_clean = pd.Series(right, dtype=float)
    if left_clean.nunique(dropna=True) <= 1 or right_clean.nunique(dropna=True) <= 1:
        return 0.0
    correlation = left_clean.corr(right_clean)
    return float(correlation) if pd.notna(correlation) else 0.0


def _normalize_session_date(value) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _overlapping_strategy_metrics(oos_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    overlapping = oos_df.copy()
    if "base_session_date" not in overlapping.columns:
        overlapping["base_session_date"] = pd.NaT
    if "future_session_date" not in overlapping.columns:
        overlapping["future_session_date"] = pd.NaT
    overlapping["strategy_return"] = overlapping["position"] * overlapping["forward_return"]
    overlapping["benchmark_return"] = overlapping["forward_return"]
    overlapping["cumulative_strategy_equity"] = (1.0 + overlapping["strategy_return"]).cumprod()
    overlapping["cumulative_buy_hold_equity"] = (1.0 + overlapping["benchmark_return"]).cumprod()
    overlapping["trade_entry_date"] = pd.NaT
    overlapping["trade_exit_date"] = pd.NaT
    overlapping = overlapping[STRATEGY_COLUMNS]

    metrics = {
        "executed_trade_count": int((overlapping["position"] != 0).sum()),
        "cumulative_strategy_return": float(overlapping["cumulative_strategy_equity"].iloc[-1] - 1.0) if not overlapping.empty else 0.0,
        "cumulative_buy_hold_return": float(overlapping["cumulative_buy_hold_equity"].iloc[-1] - 1.0) if not overlapping.empty else 0.0,
        "max_drawdown": _max_drawdown(overlapping["cumulative_strategy_equity"]),
    }
    return overlapping, metrics


def _simulate_non_overlapping_trades(oos_df: pd.DataFrame, market_sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sessions = market_sessions.copy()
    sessions["session_date"] = pd.to_datetime(sessions["session_date"]).dt.date
    sessions = sessions.sort_values("session_date").drop_duplicates("session_date", keep="last").reset_index(drop=True)
    sessions["daily_return"] = sessions["close"].pct_change().fillna(0.0)
    sessions["session_index"] = sessions.index
    session_to_index = dict(zip(sessions["session_date"], sessions["session_index"]))

    candidates = oos_df.copy()
    candidates["base_session_date"] = candidates["base_session_date"].map(_normalize_session_date)
    candidates["future_session_date"] = candidates["future_session_date"].map(_normalize_session_date)
    candidates["base_index"] = candidates["base_session_date"].map(session_to_index)
    candidates["future_index"] = candidates["future_session_date"].map(session_to_index)
    candidates["entry_index"] = candidates["base_index"].apply(
        lambda value: int(value) + 1 if pd.notna(value) else None
    )
    candidates["tradable"] = False
    comparable = candidates["entry_index"].notna() & candidates["future_index"].notna()
    candidates.loc[comparable, "tradable"] = (
        candidates.loc[comparable, "entry_index"].astype(int)
        <= candidates.loc[comparable, "future_index"].astype(int)
    )

    tradable = candidates[candidates["tradable"]].copy().reset_index(drop=True)
    if tradable.empty:
        return pd.DataFrame(columns=STRATEGY_COLUMNS), {
            "executed_trade_count": 0,
            "cumulative_strategy_return": 0.0,
            "cumulative_buy_hold_return": 0.0,
            "max_drawdown": 0.0,
        }

    start_index = int(tradable["entry_index"].min())
    end_index = int(tradable["future_index"].max())
    backtest_sessions = sessions.iloc[start_index : end_index + 1].copy()
    backtest_sessions["strategy_daily_return"] = 0.0

    executed_rows = []
    active_until_index = -1
    for row in tradable.itertuples(index=False):
        entry_index = int(row.entry_index)
        exit_index = int(row.future_index)
        if row.position == 0 or entry_index <= active_until_index:
            continue

        window = sessions.iloc[entry_index : exit_index + 1]
        strategy_window_returns = row.position * window["daily_return"]
        strategy_return = float((1.0 + strategy_window_returns).prod() - 1.0)
        benchmark_return = float((1.0 + window["daily_return"]).prod() - 1.0)

        mask = (backtest_sessions["session_index"] >= entry_index) & (backtest_sessions["session_index"] <= exit_index)
        backtest_sessions.loc[mask, "strategy_daily_return"] = row.position * backtest_sessions.loc[mask, "daily_return"]

        executed_rows.append(
            {
                "snapshot_at": row.snapshot_at,
                "provider": row.provider,
                "term": row.term,
                "signal_name": row.signal_name,
                "horizon_label": row.horizon_label,
                "score": row.score,
                "forward_return": row.forward_return,
                "threshold": row.threshold,
                "position": row.position,
                "predicted_label": row.predicted_label,
                "true_label": row.true_label,
                "base_session_date": row.base_session_date,
                "future_session_date": row.future_session_date,
                "trade_entry_date": sessions.iloc[entry_index]["session_date"],
                "trade_exit_date": sessions.iloc[exit_index]["session_date"],
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "_trade_exit_index": exit_index,
            }
        )
        active_until_index = exit_index

    backtest_sessions["cumulative_strategy_equity"] = (1.0 + backtest_sessions["strategy_daily_return"]).cumprod()
    backtest_sessions["cumulative_buy_hold_equity"] = (1.0 + backtest_sessions["daily_return"]).cumprod()

    if not executed_rows:
        metrics = {
            "executed_trade_count": 0,
            "cumulative_strategy_return": float(backtest_sessions["cumulative_strategy_equity"].iloc[-1] - 1.0),
            "cumulative_buy_hold_return": float(backtest_sessions["cumulative_buy_hold_equity"].iloc[-1] - 1.0),
            "max_drawdown": _max_drawdown(backtest_sessions["cumulative_strategy_equity"]),
        }
        return pd.DataFrame(columns=STRATEGY_COLUMNS), metrics

    executed_df = pd.DataFrame(executed_rows)
    equity_lookup = backtest_sessions.set_index("session_index")[
        ["cumulative_strategy_equity", "cumulative_buy_hold_equity"]
    ]
    executed_df["cumulative_strategy_equity"] = executed_df["_trade_exit_index"].map(
        equity_lookup["cumulative_strategy_equity"]
    )
    executed_df["cumulative_buy_hold_equity"] = executed_df["_trade_exit_index"].map(
        equity_lookup["cumulative_buy_hold_equity"]
    )
    executed_df = executed_df.drop(columns=["_trade_exit_index"])
    executed_df = executed_df[STRATEGY_COLUMNS].sort_values("snapshot_at").reset_index(drop=True)

    metrics = {
        "executed_trade_count": int(len(executed_df)),
        "cumulative_strategy_return": float(backtest_sessions["cumulative_strategy_equity"].iloc[-1] - 1.0),
        "cumulative_buy_hold_return": float(backtest_sessions["cumulative_buy_hold_equity"].iloc[-1] - 1.0),
        "max_drawdown": _max_drawdown(backtest_sessions["cumulative_strategy_equity"]),
    }
    return executed_df, metrics


def determine_verdict(sample_count: int, directional_accuracy: float, strategy_return: float, buy_hold_return: float) -> str:
    performance_checks = int(directional_accuracy >= 0.55) + int(strategy_return > buy_hold_return)
    if sample_count >= 25 and performance_checks == 2:
        return "Reliable"
    if sample_count >= 15 and performance_checks == 1:
        return "Experimental"
    return "Unreliable"


def build_calibration_rows(oos_df: pd.DataFrame, mapping: SignalMapping) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {
            "bucket": pd.Categorical(CALIBRATION_BUCKETS, categories=CALIBRATION_BUCKETS, ordered=True),
            "forward_return_average": 0.0,
            "hit_rate": 0.0,
            "sample_count": 0,
        }
    )
    if not oos_df.empty:
        bucketed = oos_df.copy()
        bucketed["bucket"] = pd.cut(
            bucketed["score"],
            bins=CALIBRATION_BINS,
            labels=CALIBRATION_BUCKETS,
            include_lowest=True,
            right=True,
        )
        grouped = (
            bucketed.dropna(subset=["bucket"])
            .groupby("bucket", observed=False)
            .agg(
                forward_return_average=("forward_return", "mean"),
                hit_rate=("forward_return", lambda values: float((values > 0).mean())),
                sample_count=("forward_return", "count"),
            )
            .reset_index()
        )
        calibration = calibration.drop(columns=["forward_return_average", "hit_rate", "sample_count"]).merge(
            grouped,
            on="bucket",
            how="left",
        )
        calibration["forward_return_average"] = calibration["forward_return_average"].fillna(0.0)
        calibration["hit_rate"] = calibration["hit_rate"].fillna(0.0)
        calibration["sample_count"] = calibration["sample_count"].fillna(0).astype(int)

    calibration.insert(0, "horizon_label", mapping.horizon_label)
    calibration.insert(0, "term", mapping.term)
    calibration.insert(0, "signal_name", mapping.signal_name)
    return calibration[CALIBRATION_COLUMNS]


def evaluate_signal_mapping(
    signals_df: pd.DataFrame,
    mapping: SignalMapping,
    market_sessions: pd.DataFrame | None = None,
    min_train_size: int = 20,
    thresholds: Iterable[int] = range(0, 101, 5),
) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str]]:
    subset = (
        signals_df[
            (signals_df["signal_name"] == mapping.signal_name)
            & (signals_df["horizon_label"] == mapping.horizon_label)
        ]
        .sort_values("snapshot_at")
        .reset_index(drop=True)
    )
    total_observations = len(subset)
    labeled = subset[subset["label_available"]].copy().reset_index(drop=True)
    labeled["true_label"] = labeled["forward_return"].apply(classify_return)

    warnings: list[str] = []
    missing_labels = total_observations - len(labeled)
    if missing_labels:
        warnings.append(
            f"{mapping.signal_name} -> {mapping.horizon_label} is missing {missing_labels} forward labels."
        )

    if len(labeled) <= min_train_size:
        if len(labeled) == 0:
            warnings.append(f"{mapping.signal_name} -> {mapping.horizon_label} has no labeled observations yet.")
        else:
            warnings.append(
                f"{mapping.signal_name} -> {mapping.horizon_label} has only {len(labeled)} labeled observations; "
                f"{min_train_size + 1} are needed for walk-forward scoring."
            )
        evaluation_row = {
            "signal_name": mapping.signal_name,
            "term": mapping.term,
            "horizon_label": mapping.horizon_label,
            "sample_count": 0,
            "executed_trade_count": 0,
            "labeled_observations": int(len(labeled)),
            "total_observations": int(total_observations),
            "coverage": 0.0,
            "pearson_correlation": 0.0,
            "directional_accuracy": 0.0,
            "precision_up": 0.0,
            "recall_up": 0.0,
            "precision_down": 0.0,
            "recall_down": 0.0,
            "cumulative_strategy_return": 0.0,
            "cumulative_buy_hold_return": 0.0,
            "excess_return_vs_buy_hold": 0.0,
            "max_drawdown": 0.0,
            "latest_threshold": 0,
            "verdict": "Unreliable",
        }
        return (
            evaluation_row,
            pd.DataFrame(columns=STRATEGY_COLUMNS),
            build_calibration_rows(pd.DataFrame(columns=["score", "forward_return"]), mapping),
            warnings,
        )

    strategy_rows = []
    for position_index in range(min_train_size, len(labeled)):
        train_df = labeled.iloc[:position_index]
        test_row = labeled.iloc[position_index].copy()
        diagnostic = fit_optimal_threshold(train_df, thresholds=thresholds)
        threshold = int(diagnostic["threshold"])
        position = determine_position(float(test_row["score"]), threshold)
        strategy_return = float(position * test_row["forward_return"])
        benchmark_return = float(test_row["forward_return"])
        strategy_rows.append(
            {
                **test_row.to_dict(),
                "threshold": threshold,
                "position": position,
                "predicted_label": classify_position(position),
                "true_label": test_row["true_label"],
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
            }
        )

    oos_df = pd.DataFrame(strategy_rows)
    if market_sessions is None:
        strategies_df, strategy_metrics = _overlapping_strategy_metrics(oos_df)
    else:
        strategies_df, strategy_metrics = _simulate_non_overlapping_trades(oos_df, market_sessions)

    active_mask = oos_df["position"] != 0
    directional_accuracy = float(
        (oos_df.loc[active_mask, "predicted_label"] == oos_df.loc[active_mask, "true_label"]).mean()
    ) if active_mask.any() else 0.0
    precision_up, recall_up = _precision_recall(oos_df, "Up")
    precision_down, recall_down = _precision_recall(oos_df, "Down")
    cumulative_strategy_return = strategy_metrics["cumulative_strategy_return"]
    cumulative_buy_hold_return = strategy_metrics["cumulative_buy_hold_return"]
    verdict = determine_verdict(
        sample_count=len(oos_df),
        directional_accuracy=directional_accuracy,
        strategy_return=cumulative_strategy_return,
        buy_hold_return=cumulative_buy_hold_return,
    )

    evaluation_row = {
        "signal_name": mapping.signal_name,
        "term": mapping.term,
        "horizon_label": mapping.horizon_label,
        "sample_count": int(len(oos_df)),
        "executed_trade_count": int(strategy_metrics["executed_trade_count"]),
        "labeled_observations": int(len(labeled)),
        "total_observations": int(total_observations),
        "coverage": float(active_mask.mean()),
        "pearson_correlation": _safe_pearson_correlation(oos_df["score"], oos_df["forward_return"]),
        "directional_accuracy": directional_accuracy,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "precision_down": precision_down,
        "recall_down": recall_down,
        "cumulative_strategy_return": cumulative_strategy_return,
        "cumulative_buy_hold_return": cumulative_buy_hold_return,
        "excess_return_vs_buy_hold": float(cumulative_strategy_return - cumulative_buy_hold_return),
        "max_drawdown": strategy_metrics["max_drawdown"],
        "latest_threshold": int(oos_df["threshold"].iloc[-1]) if not oos_df.empty else 0,
        "verdict": verdict,
    }

    calibration_df = build_calibration_rows(oos_df, mapping)
    return evaluation_row, strategies_df, calibration_df, warnings
