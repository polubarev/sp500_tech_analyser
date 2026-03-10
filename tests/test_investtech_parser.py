from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sp500_tech_analyser.providers.investtech import parse_investtech_html


def test_parse_investtech_html_extracts_expected_fields():
    html = Path("tests/fixtures/investtech/normal.html").read_text(encoding="utf-8")
    snapshot = parse_investtech_html(html, snapshot_at=datetime(2024, 1, 10, 13, 0, tzinfo=timezone.utc))

    short_term = snapshot.payload["short_term"]
    medium_term = snapshot.payload["medium_term"]
    long_term = snapshot.payload["long_term"]

    assert snapshot.payload["datetime"] == "2024-01-10T13:00:00Z"
    assert short_term["analysis"] == "Short analysis paragraph."
    assert short_term["special"] == "Short RSI commentary."
    assert short_term["conclusion"] == "Short conclusion text."
    assert short_term["recommendation"] == "Weak Negative"
    assert short_term["score"] == -29
    assert medium_term["special"] == "Medium support resistance note."
    assert long_term["special"] is None


def test_parse_investtech_html_handles_missing_special_block():
    html = Path("tests/fixtures/investtech/missing_special.html").read_text(encoding="utf-8")
    snapshot = parse_investtech_html(html, snapshot_at=datetime(2024, 1, 10, 13, 0, tzinfo=timezone.utc))

    assert snapshot.payload["short_term"]["analysis"] == "Short analysis paragraph without special block."
    assert snapshot.payload["short_term"]["special"] is None
    assert snapshot.payload["short_term"]["conclusion"] == "Short conclusion text."
