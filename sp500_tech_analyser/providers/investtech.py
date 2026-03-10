from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape

import requests
from bs4 import BeautifulSoup, Comment

from ..constants import PROVIDER_NAME, RAW_TERM_KEYS
from ..storage import format_utc_timestamp
from .base import RawSnapshot

TERM_LABELS = {
    "short": "Short term",
    "medium": "Medium term",
    "long": "Long term",
}

SPECIAL_MARKERS = {
    "short": ("RSI start", "RSI end"),
    "medium": ("SR start", "SR end"),
}

CONCLUSION_MARKERS = ("techConclusionStart", "techConclusionEnd")


def _clean_text(fragment: str | None) -> str | None:
    if not fragment:
        return None
    soup = BeautifulSoup(fragment, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text or None


def _extract_comment_block(html_fragment: str, marker_start: str, marker_end: str) -> str | None:
    pattern = rf"<!--\s*{re.escape(marker_start)}\s*-->(.*?)<!--\s*{re.escape(marker_end)}\s*-->"
    match = re.search(pattern, html_fragment, re.S)
    return _clean_text(match.group(1)) if match else None


def _strip_comment_blocks(html_fragment: str, term: str) -> str:
    patterns = [
        rf"<!--\s*{CONCLUSION_MARKERS[0]}\s*-->(.*?)<!--\s*{CONCLUSION_MARKERS[1]}\s*-->",
    ]
    if term in SPECIAL_MARKERS:
        marker_start, marker_end = SPECIAL_MARKERS[term]
        patterns.append(rf"<!--\s*{re.escape(marker_start)}\s*-->(.*?)<!--\s*{re.escape(marker_end)}\s*-->")
    stripped = html_fragment
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.S)
    return stripped


def parse_term_block(container: BeautifulSoup, term: str) -> dict | None:
    term_name = TERM_LABELS[term]
    for div in container.find_all("div", class_="cr_oneColWith20pctMargins"):
        heading = div.find("h2")
        if not heading or heading.get_text(strip=True) != term_name:
            continue

        html_fragment = str(div)
        analysis_soup = BeautifulSoup(_strip_comment_blocks(html_fragment, term), "html.parser")
        for tag in analysis_soup.find_all(["h2", "h3"]):
            tag.decompose()
        for comment in analysis_soup.find_all(string=lambda value: isinstance(value, Comment)):
            comment.extract()
        analysis = _clean_text(str(analysis_soup))

        recommendation = None
        score = None
        heading_block = div.find("h3")
        if heading_block:
            score_match = re.search(r"Score:\s*([-+]?\d+)", heading_block.get_text(" ", strip=True))
            score = int(score_match.group(1)) if score_match else None
            recommendation_span = heading_block.find("span", id=re.compile(r".*CommentaryEvaluation"))
            if recommendation_span:
                recommendation = _clean_text(str(recommendation_span))

        special = None
        if term in SPECIAL_MARKERS:
            marker_start, marker_end = SPECIAL_MARKERS[term]
            special = _extract_comment_block(html_fragment, marker_start, marker_end)
        conclusion = _extract_comment_block(html_fragment, *CONCLUSION_MARKERS)

        return {
            "analysis": analysis,
            "special": special,
            "conclusion": conclusion,
            "recommendation": recommendation,
            "score": score,
        }
    return None


def parse_investtech_html(html: str, snapshot_at: datetime | None = None) -> RawSnapshot:
    snapshot_at = snapshot_at or datetime.now(timezone.utc)
    if snapshot_at.tzinfo is None:
        snapshot_at = snapshot_at.replace(tzinfo=timezone.utc)
    else:
        snapshot_at = snapshot_at.astimezone(timezone.utc)

    soup = BeautifulSoup(html, "html.parser")
    payload = {
        "datetime": format_utc_timestamp(snapshot_at),
        RAW_TERM_KEYS["short"]: parse_term_block(soup, "short"),
        RAW_TERM_KEYS["medium"]: parse_term_block(soup, "medium"),
        RAW_TERM_KEYS["long"]: parse_term_block(soup, "long"),
    }
    return RawSnapshot(provider=PROVIDER_NAME, snapshot_at=snapshot_at, payload=payload)


@dataclass
class InvesttechProvider:
    url: str
    timeout_seconds: int = 30
    name: str = PROVIDER_NAME

    def fetch_html(self) -> str:
        response = requests.get(self.url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def build_raw_snapshot(self, html: str | None = None, snapshot_at: datetime | None = None) -> RawSnapshot:
        html = html if html is not None else self.fetch_html()
        return parse_investtech_html(html, snapshot_at=snapshot_at)
