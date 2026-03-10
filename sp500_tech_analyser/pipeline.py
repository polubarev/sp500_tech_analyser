from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import pandas as pd

from .config import AppConfig
from .constants import (
    CALIBRATION_COLUMNS,
    EVALUATION_COLUMNS,
    RAW_TERM_KEYS,
    SIGNAL_COLUMNS,
    SIGNAL_MAPPINGS,
    SNAPSHOT_COLUMNS,
    STRATEGY_COLUMNS,
)
from .evaluation import evaluate_signal_mapping
from .market import fetch_benchmark_history, resolve_base_session, resolve_future_session
from .providers.investtech import InvesttechProvider
from .storage import (
    bootstrap_legacy_raw_history,
    ensure_directories,
    format_utc_timestamp,
    load_raw_snapshot_payloads,
    parse_utc_timestamp,
    processed_artifact_path,
    sync_raw_snapshots_from_gcs,
    upload_raw_snapshot_to_gcs,
    write_dataframe,
    write_json,
    write_local_raw_snapshot,
)

logger = logging.getLogger(__name__)


def normalize_raw_snapshots(config: AppConfig) -> pd.DataFrame:
    payloads = load_raw_snapshot_payloads(config)
    rows = []
    for source_path, payload in payloads:
        snapshot_at = parse_utc_timestamp(payload["datetime"])
        for term, raw_key in RAW_TERM_KEYS.items():
            block = payload.get(raw_key) or {}
            rows.append(
                {
                    "snapshot_at": snapshot_at,
                    "provider": config.provider_name,
                    "term": term,
                    "score": block.get("score"),
                    "recommendation": block.get("recommendation"),
                    "analysis": block.get("analysis"),
                    "conclusion": block.get("conclusion"),
                    "special": block.get("special"),
                    "_source_path": str(source_path),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    snapshots = pd.DataFrame(rows).sort_values(["snapshot_at", "provider", "term", "_source_path"])
    snapshots["snapshot_at"] = pd.to_datetime(snapshots["snapshot_at"], utc=True)
    snapshots = snapshots.drop_duplicates(subset=["snapshot_at", "provider", "term"], keep="last")
    normalized = snapshots[SNAPSHOT_COLUMNS].reset_index(drop=True)
    logger.info(
        "Normalized %s raw row(s) into %s snapshot row(s)",
        len(rows),
        len(normalized),
    )
    return normalized


def build_signal_frame(snapshots_df: pd.DataFrame, market_sessions: pd.DataFrame) -> pd.DataFrame:
    if snapshots_df.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    market_sessions = market_sessions.copy()
    market_sessions["session_date"] = pd.to_datetime(market_sessions["session_date"]).dt.date
    market_sessions = market_sessions.sort_values("session_date").drop_duplicates("session_date", keep="last")

    session_dates = market_sessions["session_date"].tolist()
    close_by_date = dict(zip(session_dates, market_sessions["close"]))

    rows = []
    for mapping in SIGNAL_MAPPINGS:
        term_rows = snapshots_df[snapshots_df["term"] == mapping.term].copy()
        for snapshot in term_rows.itertuples(index=False):
            snapshot_at = snapshot.snapshot_at
            if isinstance(snapshot_at, pd.Timestamp):
                if snapshot_at.tzinfo is None:
                    snapshot_at = snapshot_at.tz_localize("UTC").to_pydatetime()
                else:
                    snapshot_at = snapshot_at.to_pydatetime()
            else:
                if snapshot_at.tzinfo is None:
                    snapshot_at = snapshot_at.replace(tzinfo=timezone.utc)

            base_session_date = resolve_base_session(snapshot_at, session_dates)
            future_session_date = None
            base_close = None
            future_close = None
            forward_return = None
            if base_session_date is not None:
                target_date = base_session_date + mapping.horizon_delta
                future_session_date = resolve_future_session(base_session_date, target_date, session_dates)
                if future_session_date is not None:
                    base_close = float(close_by_date[base_session_date])
                    future_close = float(close_by_date[future_session_date])
                    forward_return = float((future_close - base_close) / base_close)

            rows.append(
                {
                    "snapshot_at": snapshot_at,
                    "provider": snapshot.provider,
                    "term": snapshot.term,
                    "signal_name": mapping.signal_name,
                    "horizon_label": mapping.horizon_label,
                    "score": snapshot.score,
                    "recommendation": snapshot.recommendation,
                    "analysis": snapshot.analysis,
                    "conclusion": snapshot.conclusion,
                    "special": snapshot.special,
                    "base_session_date": base_session_date,
                    "future_session_date": future_session_date,
                    "base_close": base_close,
                    "future_close": future_close,
                    "forward_return": forward_return,
                    "label_available": forward_return is not None,
                }
            )
    signals = pd.DataFrame(rows)
    if signals.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    normalized_signals = signals[SIGNAL_COLUMNS].sort_values(["snapshot_at", "signal_name", "horizon_label"]).reset_index(drop=True)
    logger.info("Built %s signal row(s) across %s mapping(s)", len(normalized_signals), len(SIGNAL_MAPPINGS))
    return normalized_signals


def _guidance_for_row(latest_row: dict, evaluation_row: dict, horizon_label: str) -> str:
    verdict = evaluation_row.get("verdict", "Unreliable")
    recommendation = latest_row.get("recommendation") or "No recommendation"
    horizon = horizon_label or latest_row.get("horizon_label") or evaluation_row.get("horizon_label") or "target horizon"
    if verdict == "Reliable":
        return f"Use the latest {horizon} signal as a primary directional input, but size risk conservatively."
    if verdict == "Experimental":
        return f"Use the latest {horizon} signal only with confirmation from other inputs; it is not robust enough alone."
    return f"Do not trade the latest {horizon} signal on its own; treat {recommendation.lower()} as descriptive only."


def build_dashboard_summary(
    config: AppConfig,
    snapshots_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    warnings: list[str],
) -> dict:
    latest_snapshot_at = None
    if not snapshots_df.empty:
        latest_snapshot_at = snapshots_df["snapshot_at"].max()

    latest_signals = []
    if not signals_df.empty:
        latest_lookup = (
            signals_df.sort_values("snapshot_at")
            .groupby(["signal_name", "horizon_label"], as_index=False)
            .tail(1)
            .set_index(["signal_name", "horizon_label"])
        )
        evaluation_lookup = evaluation_df.set_index(["signal_name", "horizon_label"]) if not evaluation_df.empty else {}
        for mapping in SIGNAL_MAPPINGS:
            key = (mapping.signal_name, mapping.horizon_label)
            if key not in latest_lookup.index:
                continue
            latest_row = latest_lookup.loc[key].to_dict()
            evaluation_row = (
                evaluation_lookup.loc[key].to_dict() if len(evaluation_lookup) and key in evaluation_lookup.index else {}
            )
            latest_signals.append(
                {
                    "signal_name": mapping.signal_name,
                    "term": mapping.term,
                    "horizon_label": mapping.horizon_label,
                    "latest_snapshot_at": format_utc_timestamp(pd.Timestamp(latest_row["snapshot_at"]).to_pydatetime()),
                    "latest_score": latest_row.get("score"),
                    "latest_recommendation": latest_row.get("recommendation"),
                    "latest_conclusion": latest_row.get("conclusion"),
                    "verdict": evaluation_row.get("verdict", "Unreliable"),
                    "sample_count": int(evaluation_row.get("sample_count", 0) or 0),
                    "executed_trade_count": int(evaluation_row.get("executed_trade_count", 0) or 0),
                    "coverage": float(evaluation_row.get("coverage", 0.0) or 0.0),
                    "directional_accuracy": float(evaluation_row.get("directional_accuracy", 0.0) or 0.0),
                    "cumulative_strategy_return": float(
                        evaluation_row.get("cumulative_strategy_return", 0.0) or 0.0
                    ),
                    "excess_return_vs_buy_hold": float(
                        evaluation_row.get("excess_return_vs_buy_hold", 0.0) or 0.0
                    ),
                    "usage_guidance": _guidance_for_row(latest_row, evaluation_row, mapping.horizon_label),
                }
            )

    executive_message = "No processed snapshots are available yet."
    if latest_signals:
        lead = max(latest_signals, key=lambda row: abs(row.get("latest_score") or 0))
        executive_message = (
            f"Latest Investtech signal: {lead['signal_name']} for {lead['horizon_label']} scored "
            f"{lead['latest_score']}, is rated {lead['verdict']}, and should be used as follows: {lead['usage_guidance']}"
        )

    return {
        "generated_at": format_utc_timestamp(datetime.now(timezone.utc)),
        "provider": config.provider_name,
        "benchmark_ticker": config.benchmark_ticker,
        "raw_snapshot_count": int(snapshots_df["snapshot_at"].nunique()) if not snapshots_df.empty else 0,
        "normalized_row_count": int(len(snapshots_df)),
        "latest_snapshot_at": format_utc_timestamp(latest_snapshot_at.to_pydatetime()) if latest_snapshot_at else None,
        "warnings": sorted(set(warnings)),
        "latest_signals": latest_signals,
        "executive_message": executive_message,
    }


def refresh_raw_snapshots(config: AppConfig | None = None) -> dict:
    config = config or AppConfig.from_env()
    ensure_directories(config)
    logger.info("Starting raw snapshot refresh into %s", config.raw_dir)
    bootstrapped = bootstrap_legacy_raw_history(config)
    synced = sync_raw_snapshots_from_gcs(config)
    synced["bootstrapped_from_legacy_predictions"] = bootstrapped
    logger.info(
        "Raw snapshot refresh complete: bootstrapped=%s downloaded=%s skipped=%s total_seen=%s",
        bootstrapped,
        synced["downloaded"],
        synced["skipped"],
        synced["total_seen"],
    )
    return synced


def capture_latest_snapshot(config: AppConfig | None = None, upload_to_gcs: bool = True) -> dict:
    config = config or AppConfig.from_env()
    ensure_directories(config)
    logger.info("Capturing latest %s snapshot from %s", config.provider_name, config.investtech_url)
    provider = InvesttechProvider(url=config.investtech_url)
    raw_snapshot = provider.build_raw_snapshot()
    local_path = write_local_raw_snapshot(config, raw_snapshot.provider, raw_snapshot.snapshot_at, raw_snapshot.payload)
    blob_name = None
    if upload_to_gcs:
        blob_name = upload_raw_snapshot_to_gcs(
            config=config,
            provider=raw_snapshot.provider,
            snapshot_at=raw_snapshot.snapshot_at,
            payload=raw_snapshot.payload,
        )
    logger.info(
        "Latest snapshot capture complete: snapshot_at=%s local_path=%s uploaded=%s",
        format_utc_timestamp(raw_snapshot.snapshot_at),
        local_path,
        bool(blob_name),
    )
    return {
        "provider": raw_snapshot.provider,
        "snapshot_at": format_utc_timestamp(raw_snapshot.snapshot_at),
        "local_path": str(local_path),
        "gcs_blob": blob_name,
        "payload": raw_snapshot.payload,
    }


def build_processed_artifacts(
    config: AppConfig | None = None,
    market_sessions: pd.DataFrame | None = None,
    min_train_size: int = 20,
) -> dict:
    config = config or AppConfig.from_env()
    ensure_directories(config)
    logger.info("Starting processed artifact build into %s", config.processed_dir)
    snapshots_df = normalize_raw_snapshots(config)
    if snapshots_df.empty:
        raise ValueError(f"No raw snapshots were found in {config.raw_dir}.")

    if market_sessions is None:
        start_date = snapshots_df["snapshot_at"].min().date()
        end_date = snapshots_df["snapshot_at"].max().date() + timedelta(weeks=27)
        logger.info(
            "Fetching benchmark history for %s from %s to %s",
            config.benchmark_ticker,
            start_date,
            end_date,
        )
        market_sessions = fetch_benchmark_history(config.benchmark_ticker, start_date, end_date)
    logger.info("Using %s benchmark session row(s)", len(market_sessions))

    signals_df = build_signal_frame(snapshots_df, market_sessions)
    evaluation_rows = []
    strategy_frames = []
    calibration_frames = []
    warnings: list[str] = []
    for mapping in SIGNAL_MAPPINGS:
        logger.info("Evaluating mapping %s -> %s", mapping.signal_name, mapping.horizon_label)
        evaluation_row, strategy_df, calibration_df, mapping_warnings = evaluate_signal_mapping(
            signals_df=signals_df,
            mapping=mapping,
            market_sessions=market_sessions,
            min_train_size=min_train_size,
        )
        evaluation_rows.append(evaluation_row)
        strategy_frames.append(strategy_df)
        calibration_frames.append(calibration_df)
        warnings.extend(mapping_warnings)
        logger.info(
            "Finished mapping %s -> %s: sample_count=%s verdict=%s",
            mapping.signal_name,
            mapping.horizon_label,
            evaluation_row["sample_count"],
            evaluation_row["verdict"],
        )
        for warning_message in mapping_warnings:
            logger.warning(warning_message)

    evaluation_df = pd.DataFrame(evaluation_rows, columns=EVALUATION_COLUMNS)
    non_empty_strategy_frames = [frame for frame in strategy_frames if not frame.empty]
    non_empty_calibration_frames = [frame for frame in calibration_frames if not frame.empty]
    strategies_df = (
        pd.concat(non_empty_strategy_frames, ignore_index=True)
        if non_empty_strategy_frames
        else pd.DataFrame(columns=STRATEGY_COLUMNS)
    )
    calibration_df = (
        pd.concat(non_empty_calibration_frames, ignore_index=True)
        if non_empty_calibration_frames
        else pd.DataFrame(columns=CALIBRATION_COLUMNS)
    )
    dashboard_summary = build_dashboard_summary(config, snapshots_df, signals_df, evaluation_df, warnings)

    write_dataframe(processed_artifact_path(config, "snapshots.csv"), snapshots_df)
    write_dataframe(processed_artifact_path(config, "signals.csv"), signals_df)
    write_dataframe(processed_artifact_path(config, "evaluation.csv"), evaluation_df)
    write_dataframe(processed_artifact_path(config, "calibration.csv"), calibration_df)
    write_dataframe(processed_artifact_path(config, "strategies.csv"), strategies_df)
    write_json(processed_artifact_path(config, "dashboard_summary.json"), dashboard_summary)
    logger.info(
        "Processed artifact build complete: snapshots=%s signals=%s evaluation=%s calibration=%s strategies=%s warnings=%s",
        len(snapshots_df),
        len(signals_df),
        len(evaluation_df),
        len(calibration_df),
        len(strategies_df),
        len(warnings),
    )

    return {
        "snapshots": snapshots_df,
        "signals": signals_df,
        "evaluation": evaluation_df,
        "calibration": calibration_df,
        "strategies": strategies_df,
        "summary": dashboard_summary,
    }
