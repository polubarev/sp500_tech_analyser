from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class RawSnapshot:
    provider: str
    snapshot_at: datetime
    payload: dict


class SnapshotProvider(Protocol):
    name: str

    def fetch_html(self) -> str:
        """Return the provider HTML for the latest snapshot."""

    def build_raw_snapshot(self, html: str | None = None, snapshot_at: datetime | None = None) -> RawSnapshot:
        """Build one raw snapshot payload from provider HTML."""
