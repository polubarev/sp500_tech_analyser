from __future__ import annotations

import pandas as pd


def plot_strategy_curves(strategies_df: pd.DataFrame, signal_name: str, horizon_label: str):
    import matplotlib.pyplot as plt

    subset = strategies_df[
        (strategies_df["signal_name"] == signal_name) & (strategies_df["horizon_label"] == horizon_label)
    ].copy()
    if subset.empty:
        return None

    x_axis = "trade_exit_date" if "trade_exit_date" in subset.columns and subset["trade_exit_date"].notna().any() else "snapshot_at"
    subset = subset.sort_values(x_axis)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(subset[x_axis], subset["cumulative_strategy_equity"], label="Strategy", linewidth=2)
    ax.plot(subset[x_axis], subset["cumulative_buy_hold_equity"], label="Buy & Hold", linewidth=2)
    ax.set_title(f"{signal_name} -> {horizon_label} walk-forward equity")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(calibration_df: pd.DataFrame, signal_name: str, horizon_label: str):
    import matplotlib.pyplot as plt

    subset = calibration_df[
        (calibration_df["signal_name"] == signal_name) & (calibration_df["horizon_label"] == horizon_label)
    ].copy()
    if subset.empty:
        return None

    fig, ax_left = plt.subplots(figsize=(10, 4.5))
    ax_right = ax_left.twinx()
    x = range(len(subset))
    ax_left.bar(x, subset["forward_return_average"], color="#33658A", alpha=0.75, label="Avg forward return")
    ax_right.plot(x, subset["hit_rate"], color="#C84630", marker="o", linewidth=2, label="Hit rate")
    ax_left.set_xticks(list(x), subset["bucket"], rotation=20, ha="right")
    ax_left.set_ylabel("Avg forward return")
    ax_right.set_ylabel("Hit rate")
    ax_left.set_title(f"{signal_name} -> {horizon_label} calibration")
    ax_left.grid(True, axis="y", alpha=0.3)

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")
    fig.tight_layout()
    return fig


def plot_threshold_history(strategies_df: pd.DataFrame, signal_name: str, horizon_label: str):
    import matplotlib.pyplot as plt

    subset = strategies_df[
        (strategies_df["signal_name"] == signal_name) & (strategies_df["horizon_label"] == horizon_label)
    ].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("snapshot_at")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.step(subset["snapshot_at"], subset["threshold"], where="post", linewidth=2, color="#2F4858")
    ax.set_title(f"{signal_name} -> {horizon_label} selected threshold")
    ax.set_ylabel("Threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
