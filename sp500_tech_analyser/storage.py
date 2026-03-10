from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import AppConfig

logger = logging.getLogger(__name__)


ARTIFACT_NAMES = (
    "snapshots.csv",
    "signals.csv",
    "evaluation.csv",
    "calibration.csv",
    "strategies.csv",
    "dashboard_summary.json",
)


def ensure_directories(config: AppConfig) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)


def parse_utc_timestamp(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def raw_snapshot_filename(provider: str, snapshot_at: datetime) -> str:
    return f"{provider}_{snapshot_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.json"


def bootstrap_legacy_raw_history(config: AppConfig) -> int:
    ensure_directories(config)
    if not config.legacy_predictions_dir.exists():
        logger.debug("Legacy predictions directory does not exist: %s", config.legacy_predictions_dir)
        return 0

    copied = 0
    for source_path in sorted(config.legacy_predictions_dir.glob("*.json")):
        target_path = config.raw_dir / source_path.name
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    if copied:
        logger.info("Bootstrapped %s legacy raw snapshot(s) into %s", copied, config.raw_dir)
    else:
        logger.debug("No legacy raw snapshots needed bootstrapping from %s", config.legacy_predictions_dir)
    return copied


def write_local_raw_snapshot(config: AppConfig, provider: str, snapshot_at: datetime, payload: dict) -> Path:
    ensure_directories(config)
    path = config.raw_dir / raw_snapshot_filename(provider, snapshot_at)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote raw snapshot to %s", path)
    return path


def upload_raw_snapshot_to_gcs(
    config: AppConfig,
    provider: str,
    snapshot_at: datetime,
    payload: dict,
    client=None,
) -> str:
    if client is None:
        from google.cloud import storage

        client = storage.Client(project=config.gcs_project)
    bucket = client.bucket(config.gcs_bucket)
    blob_name = f"{config.raw_gcs_prefix.rstrip('/')}/{raw_snapshot_filename(provider, snapshot_at)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(payload, ensure_ascii=False, indent=2), content_type="application/json")
    logger.info("Uploaded raw snapshot to gs://%s/%s", config.gcs_bucket, blob_name)
    return blob_name


def sync_raw_snapshots_from_gcs(config: AppConfig, client=None) -> dict:
    ensure_directories(config)
    if client is None:
        from google.cloud import storage

        client = storage.Client(project=config.gcs_project)
    bucket = client.bucket(config.gcs_bucket)
    logger.info(
        "Syncing raw snapshots from gs://%s into %s",
        config.gcs_bucket,
        config.raw_dir,
    )

    downloaded = 0
    skipped = 0
    seen = set()
    for prefix in (config.raw_gcs_prefix.rstrip("/"), "investtech_"):
        logger.debug("Listing blobs with prefix %s", prefix)
        for blob in bucket.list_blobs(prefix=prefix):
            if not blob.name.endswith(".json"):
                continue
            filename = Path(blob.name).name
            if filename in seen:
                continue
            seen.add(filename)
            destination = config.raw_dir / filename
            if destination.exists():
                skipped += 1
                continue
            blob.download_to_filename(destination)
            downloaded += 1
            logger.debug("Downloaded %s to %s", blob.name, destination)

    result = {"downloaded": downloaded, "skipped": skipped, "total_seen": len(seen)}
    logger.info(
        "Raw snapshot sync complete: downloaded=%s skipped=%s total_seen=%s",
        downloaded,
        skipped,
        len(seen),
    )
    return result


def load_raw_snapshot_payloads(config: AppConfig) -> list[tuple[Path, dict]]:
    ensure_directories(config)
    bootstrap_legacy_raw_history(config)
    payloads: list[tuple[Path, dict]] = []
    for path in sorted(config.raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payloads.append((path, json.load(handle)))
    logger.info("Loaded %s raw snapshot payload(s) from %s", len(payloads), config.raw_dir)
    return payloads


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s row(s) to %s", len(df), path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote JSON artifact to %s", path)


def processed_artifact_path(config: AppConfig, name: str) -> Path:
    if name not in ARTIFACT_NAMES:
        raise ValueError(f"Unknown artifact: {name}")
    return config.processed_dir / name


def load_dashboard_bundle(config: AppConfig) -> dict:
    ensure_directories(config)
    bundle = {}
    snapshots_path = processed_artifact_path(config, "snapshots.csv")
    signals_path = processed_artifact_path(config, "signals.csv")
    evaluation_path = processed_artifact_path(config, "evaluation.csv")
    calibration_path = processed_artifact_path(config, "calibration.csv")
    strategies_path = processed_artifact_path(config, "strategies.csv")
    summary_path = processed_artifact_path(config, "dashboard_summary.json")

    bundle["snapshots"] = (
        pd.read_csv(snapshots_path, parse_dates=["snapshot_at"]) if snapshots_path.exists() else pd.DataFrame()
    )
    bundle["signals"] = pd.read_csv(signals_path, parse_dates=["snapshot_at"]) if signals_path.exists() else pd.DataFrame()
    bundle["evaluation"] = pd.read_csv(evaluation_path) if evaluation_path.exists() else pd.DataFrame()
    bundle["calibration"] = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    bundle["strategies"] = (
        pd.read_csv(strategies_path, parse_dates=["snapshot_at", "trade_entry_date", "trade_exit_date"])
        if strategies_path.exists()
        else pd.DataFrame()
    )
    bundle["summary"] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    bundle["paths"] = {
        "snapshots": snapshots_path,
        "signals": signals_path,
        "evaluation": evaluation_path,
        "calibration": calibration_path,
        "strategies": strategies_path,
        "summary": summary_path,
    }
    return bundle
