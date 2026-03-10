from __future__ import annotations

import argparse
import json
import logging

from .config import AppConfig
from .logging_utils import configure_logging
from .pipeline import build_processed_artifacts, refresh_raw_snapshots

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investtech raw sync and processed artifact builder.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to SP500_TECH_LOG_LEVEL or INFO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("refresh-raw", help="Sync raw Investtech snapshots into data/raw/investtech/.")
    subparsers.add_parser("build", help="Rebuild processed dashboard artifacts into data/processed/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = AppConfig.from_env()
    logger.info("Command started: %s", args.command)

    if args.command == "refresh-raw":
        result = refresh_raw_snapshots(config)
    elif args.command == "build":
        build_result = build_processed_artifacts(config)
        result = {"written": str(config.processed_dir), "summary": build_result["summary"]}
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    logger.info("Command finished: %s", args.command)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
