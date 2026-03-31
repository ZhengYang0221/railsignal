"""
patch_schedule.py — HSR patch version schedule.

Tries to scrape version dates from the Honkai: Star Rail wiki on first call,
falls back to a hardcoded schedule if scraping fails (bot detection, network, etc.).
Results are cached in-process for the lifetime of the server.
"""

import logging
import re
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger("railsignal.patch_schedule")

WIKI_URL = "https://honkai-star-rail.fandom.com/wiki/Version"

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Hardcoded schedule covering all confirmed patches through early 2026.
# Versions 3.5+ are estimated on the standard ~42-day cadence.
# Wiki scraping will override this with accurate dates when available.
_FALLBACK: list[dict] = [
    {"version": "1.0", "start": date(2023, 4, 26), "end": date(2023, 6, 6)},
    {"version": "1.1", "start": date(2023, 6, 7),  "end": date(2023, 7, 18)},
    {"version": "1.2", "start": date(2023, 7, 19), "end": date(2023, 8, 29)},
    {"version": "1.3", "start": date(2023, 8, 30), "end": date(2023, 10, 10)},
    {"version": "1.4", "start": date(2023, 10, 11),"end": date(2023, 11, 14)},
    {"version": "1.5", "start": date(2023, 11, 15),"end": date(2023, 12, 26)},
    {"version": "1.6", "start": date(2023, 12, 27),"end": date(2024, 2, 5)},
    {"version": "2.0", "start": date(2024, 2, 6),  "end": date(2024, 3, 26)},
    {"version": "2.1", "start": date(2024, 3, 27), "end": date(2024, 5, 7)},
    {"version": "2.2", "start": date(2024, 5, 8),  "end": date(2024, 6, 18)},
    {"version": "2.3", "start": date(2024, 6, 19), "end": date(2024, 7, 30)},
    {"version": "2.4", "start": date(2024, 7, 31), "end": date(2024, 9, 10)},
    {"version": "2.5", "start": date(2024, 9, 11), "end": date(2024, 10, 22)},
    {"version": "2.6", "start": date(2024, 10, 23),"end": date(2024, 12, 3)},
    {"version": "2.7", "start": date(2024, 12, 4), "end": date(2025, 1, 14)},
    {"version": "3.0", "start": date(2025, 1, 15), "end": date(2025, 2, 25)},
    {"version": "3.1", "start": date(2025, 2, 26), "end": date(2025, 4, 8)},
    {"version": "3.2", "start": date(2025, 4, 9),  "end": date(2025, 5, 20)},
    {"version": "3.3", "start": date(2025, 5, 21), "end": date(2025, 7, 1)},
    {"version": "3.4", "start": date(2025, 7, 2),  "end": date(2025, 8, 12)},
    {"version": "3.5", "start": date(2025, 8, 13), "end": date(2025, 9, 23)},
    {"version": "3.6", "start": date(2025, 9, 24), "end": date(2025, 11, 4)},
    {"version": "3.7", "start": date(2025, 11, 5), "end": date(2025, 12, 16)},
    {"version": "3.8", "start": date(2025, 12, 17),"end": date(2026, 1, 27)},
    {"version": "4.0", "start": date(2026, 1, 28), "end": date(2026, 3, 10)},
    {"version": "4.1", "start": date(2026, 3, 11), "end": date(2026, 4, 21)},
]

_cache: list[dict] | None = None


def _parse_date(s: str) -> Optional[date]:
    """Parse 'Month DD, YYYY' or 'Month DD YYYY' into a date object."""
    m = re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', s.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if not month:
        return None
    try:
        return date(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def _scrape_wiki() -> list[dict]:
    """
    Scrape version dates from the HSR wiki. Returns [] on any failure.
    Strips HTML tags, then searches for version-number + date-pair patterns.
    """
    try:
        resp = requests.get(
            WIKI_URL, timeout=10,
            headers={"User-Agent": "RailSignal/1.0 (portfolio project)"},
        )
        resp.raise_for_status()

        # Strip HTML tags and collapse whitespace for easy regex matching
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)

        # Look for: a version number (e.g. 2.6), then two dates within 300 chars
        pattern = re.compile(
            r"\b(\d+\.\d+)\b"
            r".{0,300}?"
            r"(\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+\d{4})"
            r".{0,150}?"
            r"(\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+\d{4})",
            re.IGNORECASE,
        )

        schedule: list[dict] = []
        seen: set[str] = set()

        for m in pattern.finditer(text[:100_000]):
            ver = m.group(1)
            if ver in seen:
                continue
            start = _parse_date(m.group(2))
            end = _parse_date(m.group(3))
            if start and end and end > start:
                schedule.append({"version": ver, "start": start, "end": end})
                seen.add(ver)

        if len(schedule) >= 5:
            logger.info("Scraped %d patch versions from wiki.", len(schedule))
            return sorted(schedule, key=lambda x: x["start"])

        logger.warning("Wiki scrape returned too few results (%d); using fallback.", len(schedule))
    except Exception as exc:
        logger.warning("Wiki scrape failed: %s — using fallback schedule.", exc)

    return []


def get_patch_schedule() -> list[dict]:
    """Return the cached patch schedule (fetches on first call)."""
    global _cache
    if _cache is None:
        scraped = _scrape_wiki()
        _cache = scraped if scraped else _FALLBACK
        logger.info("Patch schedule ready: %d versions.", len(_cache))
    return _cache


def version_for_date(d: date) -> Optional[str]:
    """Return the version tag whose window contains the given date, or None."""
    for entry in get_patch_schedule():
        if entry["start"] <= d <= entry["end"]:
            return entry["version"]
    return None


def versions_for_range(start: date, end: date) -> list[str]:
    """Return all version tags whose windows overlap the given date range."""
    return [
        e["version"] for e in get_patch_schedule()
        if e["start"] <= end and e["end"] >= start
    ]
