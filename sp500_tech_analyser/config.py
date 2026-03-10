from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import PROVIDER_NAME


@dataclass(frozen=True)
class AppConfig:
    gcs_project: str
    gcs_bucket: str
    data_root: Path
    benchmark_ticker: str
    raw_gcs_prefix: str
    investtech_url: str
    provider_name: str = PROVIDER_NAME
    legacy_predictions_dir: Path = Path("predictions")

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_root = Path(os.getenv("SP500_TECH_DATA_ROOT", "data"))
        return cls(
            gcs_project=os.getenv("SP500_TECH_PROJECT", "tg-bot-sso"),
            gcs_bucket=os.getenv("SP500_TECH_BUCKET", "sp-500-tech-analysis"),
            data_root=data_root,
            benchmark_ticker=os.getenv("SP500_TECH_BENCHMARK", "^GSPC"),
            raw_gcs_prefix=os.getenv("SP500_TECH_RAW_GCS_PREFIX", "data/raw/investtech"),
            investtech_url=os.getenv(
                "SP500_TECH_INVESTTECH_URL",
                "https://www.investtech.com/main/market.php?CompanyID=10400521&product=241",
            ),
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / self.provider_name

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"
