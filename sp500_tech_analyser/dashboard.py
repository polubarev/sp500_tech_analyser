import pandas as pd

from .config import AppConfig
from .plotting import plot_calibration, plot_strategy_curves, plot_threshold_history
from .storage import load_dashboard_bundle

st = None


def _get_streamlit():
    global st
    if st is None:
        import streamlit as streamlit

        st = streamlit
    return st


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_decimal(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def _mapping_options(evaluation_df: pd.DataFrame) -> list[str]:
    if evaluation_df.empty:
        return []
    return [
        f"{row.signal_name} -> {row.horizon_label}"
        for row in evaluation_df.itertuples(index=False)
    ]


def _split_mapping(option: str) -> tuple[str, str]:
    signal_name, horizon_label = option.split(" -> ", maxsplit=1)
    return signal_name, horizon_label


def _render_latest_signals(summary: dict) -> None:
    streamlit = _get_streamlit()
    latest_df = pd.DataFrame(summary.get("latest_signals", []))
    if latest_df.empty:
        streamlit.info("No latest signal summary is available yet.")
        return

    display_df = latest_df.rename(
        columns={
            "signal_name": "Signal",
            "horizon_label": "Horizon",
            "latest_score": "Score",
            "latest_recommendation": "Recommendation",
            "verdict": "Verdict",
            "sample_count": "OOS Samples",
            "executed_trade_count": "Trades",
            "coverage": "Coverage",
            "directional_accuracy": "Accuracy",
            "excess_return_vs_buy_hold": "Excess Return",
            "usage_guidance": "How To Use It",
        }
    )[
        [
            "Signal",
            "Horizon",
            "Score",
            "Recommendation",
            "Verdict",
            "OOS Samples",
            "Trades",
            "Coverage",
            "Accuracy",
            "Excess Return",
            "How To Use It",
        ]
    ]
    display_df["Score"] = display_df["Score"].map(_format_decimal)
    display_df["Coverage"] = display_df["Coverage"].map(_format_percent)
    display_df["Accuracy"] = display_df["Accuracy"].map(_format_percent)
    display_df["Excess Return"] = display_df["Excess Return"].map(_format_percent)
    streamlit.dataframe(display_df, width="stretch", hide_index=True)


def _render_evaluation(evaluation_df: pd.DataFrame) -> None:
    streamlit = _get_streamlit()
    if evaluation_df.empty:
        streamlit.info("No evaluation artifact is available yet.")
        return

    display_df = evaluation_df.rename(
        columns={
            "signal_name": "Signal",
            "horizon_label": "Horizon",
            "sample_count": "OOS Samples",
            "executed_trade_count": "Trades",
            "coverage": "Coverage",
            "pearson_correlation": "Pearson",
            "directional_accuracy": "Accuracy",
            "precision_up": "Precision Up",
            "recall_up": "Recall Up",
            "precision_down": "Precision Down",
            "recall_down": "Recall Down",
            "cumulative_strategy_return": "Strategy Return",
            "excess_return_vs_buy_hold": "Excess Return",
            "max_drawdown": "Max Drawdown",
            "verdict": "Verdict",
        }
    )[
        [
            "Signal",
            "Horizon",
            "OOS Samples",
            "Trades",
            "Coverage",
            "Pearson",
            "Accuracy",
            "Precision Up",
            "Recall Up",
            "Precision Down",
            "Recall Down",
            "Strategy Return",
            "Excess Return",
            "Max Drawdown",
            "Verdict",
        ]
    ]
    percent_columns = [
        "Coverage",
        "Accuracy",
        "Precision Up",
        "Recall Up",
        "Precision Down",
        "Recall Down",
        "Strategy Return",
        "Excess Return",
        "Max Drawdown",
    ]
    for column in percent_columns:
        display_df[column] = display_df[column].map(_format_percent)
    display_df["Pearson"] = display_df["Pearson"].map(_format_decimal)
    streamlit.dataframe(display_df, width="stretch", hide_index=True)


def _render_calibration_table(calibration_df: pd.DataFrame, signal_name: str, horizon_label: str) -> None:
    streamlit = _get_streamlit()
    subset = calibration_df[
        (calibration_df["signal_name"] == signal_name) & (calibration_df["horizon_label"] == horizon_label)
    ].copy()
    if subset.empty:
        streamlit.info("No calibration rows are available for this mapping.")
        return

    subset = subset.rename(
        columns={
            "bucket": "Score Bucket",
            "forward_return_average": "Avg Forward Return",
            "hit_rate": "Hit Rate",
            "sample_count": "Samples",
        }
    )[["Score Bucket", "Avg Forward Return", "Hit Rate", "Samples"]]
    subset["Avg Forward Return"] = subset["Avg Forward Return"].map(_format_percent)
    subset["Hit Rate"] = subset["Hit Rate"].map(_format_percent)
    streamlit.dataframe(subset, width="stretch", hide_index=True)


def _render_diagnostics(summary: dict, snapshots_df: pd.DataFrame) -> None:
    streamlit = _get_streamlit()
    streamlit.write(f"Raw snapshots processed: {summary.get('raw_snapshot_count', 0)}")
    streamlit.write(f"Normalized rows: {summary.get('normalized_row_count', 0)}")

    warnings = summary.get("warnings", [])
    if warnings:
        for warning in warnings:
            streamlit.warning(warning)
    else:
        streamlit.info("No build warnings were recorded.")

    if not snapshots_df.empty:
        recent = snapshots_df.sort_values("snapshot_at", ascending=False).head(9).copy()
        recent["snapshot_at"] = recent["snapshot_at"].astype(str)
        streamlit.dataframe(recent, width="stretch", hide_index=True)


def main() -> None:
    streamlit = _get_streamlit()
    streamlit.set_page_config(page_title="Investtech Decision Dashboard", layout="wide")
    config = AppConfig.from_env()
    bundle = load_dashboard_bundle(config)

    if not bundle["summary"]:
        streamlit.error(
            "Processed artifacts are missing. Run `python -m sp500_tech_analyser.cli refresh-raw` "
            "and then `python -m sp500_tech_analyser.cli build`."
        )
        return

    summary = bundle["summary"]
    evaluation_df = bundle["evaluation"]
    calibration_df = bundle["calibration"]
    strategies_df = bundle["strategies"]
    snapshots_df = bundle["snapshots"]

    streamlit.title("Investtech Decision Dashboard")
    streamlit.caption(
        f"Latest snapshot: {summary.get('latest_snapshot_at') or 'n/a'} | "
        f"Benchmark: {summary.get('benchmark_ticker', config.benchmark_ticker)}"
    )

    streamlit.subheader("Executive Summary")
    streamlit.info(summary.get("executive_message", "No executive summary is available yet."))
    _render_latest_signals(summary)

    streamlit.subheader("Historical Validation")
    _render_evaluation(evaluation_df)
    validation_options = _mapping_options(evaluation_df)
    if validation_options:
        selected = streamlit.selectbox("Validation plot", validation_options, key="validation_mapping")
        signal_name, horizon_label = _split_mapping(selected)
        validation_figure = plot_strategy_curves(strategies_df, signal_name, horizon_label)
        if validation_figure is not None:
            streamlit.pyplot(validation_figure)

    streamlit.subheader("Calibration")
    calibration_options = _mapping_options(evaluation_df)
    if calibration_options:
        selected = streamlit.selectbox("Calibration view", calibration_options, key="calibration_mapping")
        signal_name, horizon_label = _split_mapping(selected)
        _render_calibration_table(calibration_df, signal_name, horizon_label)
        calibration_figure = plot_calibration(calibration_df, signal_name, horizon_label)
        if calibration_figure is not None:
            streamlit.pyplot(calibration_figure)
    else:
        streamlit.info("No calibration artifact is available yet.")

    streamlit.subheader("Diagnostics")
    _render_diagnostics(summary, snapshots_df)
    diagnostic_options = _mapping_options(evaluation_df)
    if diagnostic_options:
        selected = streamlit.selectbox("Threshold history", diagnostic_options, key="diagnostic_mapping")
        signal_name, horizon_label = _split_mapping(selected)
        threshold_figure = plot_threshold_history(strategies_df, signal_name, horizon_label)
        if threshold_figure is not None:
            streamlit.pyplot(threshold_figure)
