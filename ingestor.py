"""
ingestor.py — Service 1: Reddit scraper, cleaner, embedder, classifier, DB writer.

Run directly:
    python ingestor.py
"""

import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import db
from brain import batch_embed
from ml_engine import classify_post, load_models
from patch_schedule import get_patch_schedule

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [INGESTOR] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("railsignal.ingestor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBREDDIT = os.getenv("REDDIT_SUBREDDIT", "HonkaiStarRail")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "railsignal/1.0")
INGEST_LIMIT = int(os.getenv("REDDIT_INGEST_LIMIT", 500))
REQUEST_DELAY = 2  # seconds between paginated requests

_vader = SentimentIntensityAnalyzer()

BOT_USERNAMES = {"AutoModerator", "reddit", "BotDefense", "anti-evil-operations"}

VERSION_PATTERNS = [
    r"\bv?(\d+\.\d+)\b",           # v2.1 or 2.1
    r"\bpatch\s+(\d+\.\d+)\b",     # patch 2.1
    r"\[(\d+\.\d+)\]",             # [2.1]
    r"(\d+\.\d+)\s+banner",        # 2.1 banner
    r"(\d+\.\d+)\s+update",        # 2.1 update
]

# ---------------------------------------------------------------------------
# Reddit fetch helpers
# ---------------------------------------------------------------------------

def fetch_feed(
    subreddit: str,
    feed: str,
    limit: int,
    time_filter: Optional[str] = None,
) -> list[dict]:
    """
    Paginate Reddit public JSON for a given feed (hot / new / top).
    time_filter applies to the top feed: "day", "week", "month", "year", "all".
    Returns a flat list of raw post dicts.
    """
    base_url = f"https://www.reddit.com/r/{subreddit}/{feed}.json"
    headers = {"User-Agent": USER_AGENT}
    posts: list[dict] = []
    after: Optional[str] = None
    page = 0

    while len(posts) < limit:
        params: dict = {"limit": 100}
        if after:
            params["after"] = after
        if time_filter:
            params["t"] = time_filter

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Reddit request failed (page %d): %s", page, exc)
            break

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            posts.append(child["data"])

        after = data.get("after")
        if not after:
            break

        page += 1
        logger.info("Fetched feed=%s page=%d total=%d", feed, page, len(posts))
        time.sleep(REQUEST_DELAY)

    return posts[:limit]


def fetch_search(
    subreddit: str,
    query: str,
    limit: int = 300,
    sort: str = "top",
    time_filter: str = "year",
) -> list[dict]:
    """
    Search a subreddit for posts matching query.
    Returns posts with their original created_utc timestamps, enabling
    historical data collection regardless of hot/new feed recency.
    """
    base_url = f"https://www.reddit.com/r/{subreddit}/search.json"
    headers = {"User-Agent": USER_AGENT}
    posts: list[dict] = []
    after: Optional[str] = None
    page = 0

    while len(posts) < limit:
        params: dict = {
            "q": query,
            "sort": sort,
            "t": time_filter,
            "restrict_sr": "1",
            "limit": 100,
        }
        if after:
            params["after"] = after

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Search request failed (page %d, q=%r): %s", page, query, exc)
            break

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            posts.append(child["data"])

        after = data.get("after")
        if not after:
            break

        page += 1
        logger.info("Search q=%r page=%d total=%d", query, page, len(posts))
        time.sleep(REQUEST_DELAY)

    return posts[:limit]


def _dedup(posts: list[dict]) -> list[dict]:
    """Deduplicate a list of raw Reddit post dicts by post id."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in posts:
        pid = p.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def fetch_posts(subreddit: str, total_limit: int) -> list[dict]:
    """
    Scrape hot + new feeds and deduplicate.
    Covers the last 1-2 weeks of activity for a busy subreddit.
    """
    per_feed = total_limit // 2
    raw = fetch_feed(subreddit, "hot", per_feed) + fetch_feed(subreddit, "new", per_feed)
    deduped = _dedup(raw)
    logger.info("Fetched %d unique posts from r/%s (hot+new)", len(deduped), subreddit)
    return deduped[:total_limit]


def fetch_history(subreddit: str) -> list[dict]:
    """
    Collect historical posts across multiple patch windows by combining:
      1. Top posts from the past year  (original created_utc timestamps preserved)
      2. Per-version search results    (finds posts that mention each patch by number)

    This allows the Version Compare tab to show data across patches even
    when hot/new feeds only surface the most recent content.
    """
    all_posts: list[dict] = []

    # 1. Top posts from the past year — high-quality posts with original dates
    logger.info("Fetching top/year posts ...")
    all_posts.extend(fetch_feed(subreddit, "top", 500, time_filter="year"))

    # 2. Version-specific searches — use last 8 patches from the schedule
    history_versions = [e["version"] for e in get_patch_schedule()[-8:]]
    logger.info("History versions to search: %s", history_versions)
    for version in history_versions:
        logger.info("Searching for version %s posts ...", version)
        all_posts.extend(fetch_search(subreddit, version, limit=100, time_filter="year"))
        time.sleep(REQUEST_DELAY)

    deduped = _dedup(all_posts)
    logger.info("History fetch complete: %d unique posts", len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Normalize unicode, collapse whitespace, remove null bytes."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_version_tag(text: str) -> Optional[str]:
    """
    Extract the first game version tag found in text.
    Returns e.g. '2.1', '2.2', or None.
    """
    lower = text.lower()
    for pattern in VERSION_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return match.group(1)
    return None


def is_bot(author: str) -> bool:
    """Return True if the author is a known bot account."""
    return author in BOT_USERNAMES or author.lower().endswith("bot")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO reddit_posts (
    post_id, title, body, full_text, author,
    upvotes, upvote_ratio, num_comments,
    created_utc, version_tag, post_type, sentiment_score, embedding
)
VALUES (
    %(post_id)s, %(title)s, %(body)s, %(full_text)s, %(author)s,
    %(upvotes)s, %(upvote_ratio)s, %(num_comments)s,
    %(created_utc)s, %(version_tag)s, %(post_type)s, %(sentiment_score)s,
    %(embedding)s::vector
)
ON CONFLICT (post_id) DO UPDATE SET
    upvotes         = EXCLUDED.upvotes,
    upvote_ratio    = EXCLUDED.upvote_ratio,
    num_comments    = EXCLUDED.num_comments,
    post_type       = EXCLUDED.post_type,
    sentiment_score = EXCLUDED.sentiment_score,
    embedding       = EXCLUDED.embedding,
    scraped_at      = NOW();
"""

LOG_SQL = """
INSERT INTO ingestion_log
    (posts_scraped, posts_embedded, posts_classified, status, error_msg)
VALUES
    (%(posts_scraped)s, %(posts_embedded)s, %(posts_classified)s,
     %(status)s, %(error_msg)s);
"""


def _upsert_post(record: dict) -> None:
    """Upsert a single processed post record into reddit_posts."""
    with db.db_cursor(commit=True) as cur:
        cur.execute(UPSERT_SQL, record)


def _write_log(stats: dict) -> None:
    """Write a run summary row to ingestion_log."""
    with db.db_cursor(commit=True) as cur:
        cur.execute(LOG_SQL, stats)


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def run_ingestion(mode: str = "standard") -> dict:
    """
    Full pipeline: fetch → clean → embed → classify → upsert.

    Args:
        mode: "standard" — scrapes hot + new feeds (last 1-2 weeks).
              "history"  — scrapes top/year + version searches (last 12 months).
    Returns a summary stats dict.
    """
    t_start = time.time()
    load_models()

    stats = {
        "posts_scraped": 0,
        "posts_embedded": 0,
        "posts_classified": 0,
        "status": "failed",
        "error_msg": None,
        "mode": mode,
    }

    try:
        # 1. Fetch raw posts
        if mode == "history":
            logger.info("Historical ingestion mode: top/year + version searches.")
            raw_posts = fetch_history(SUBREDDIT)
        else:
            raw_posts = fetch_posts(SUBREDDIT, INGEST_LIMIT)
        stats["posts_scraped"] = len(raw_posts)

        # 2. Clean and filter
        cleaned: list[dict] = []
        for p in raw_posts:
            author = p.get("author", "")
            if is_bot(author):
                continue

            title = clean_text(p.get("title", ""))
            body = clean_text(p.get("selftext", ""))
            full_text = f"{title} {body}".strip()

            if not full_text:
                continue

            created_utc = datetime.fromtimestamp(
                p.get("created_utc", 0), tz=timezone.utc
            ).replace(tzinfo=None)

            cleaned.append(
                {
                    "post_id": p["id"],
                    "title": title,
                    "body": body,
                    "full_text": full_text,
                    "author": author,
                    "upvotes": int(p.get("ups", 0)),
                    "upvote_ratio": float(p.get("upvote_ratio", 0.0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "created_utc": created_utc,
                    "version_tag": extract_version_tag(full_text),
                }
            )

        logger.info("After cleaning: %d posts remain", len(cleaned))

        # 3. Skip posts that already have embeddings in the DB
        existing_ids: set[str] = set()
        if cleaned:
            rows = db.execute_query(
                "SELECT post_id FROM reddit_posts WHERE post_id = ANY(%s) AND embedding IS NOT NULL",
                ([r["post_id"] for r in cleaned],),
            )
            existing_ids = {row["post_id"] for row in rows}

        new_posts  = [r for r in cleaned if r["post_id"] not in existing_ids]
        seen_posts = [r for r in cleaned if r["post_id"] in existing_ids]
        logger.info(
            "%d posts already embedded (skipping), %d new posts to embed",
            len(seen_posts), len(new_posts),
        )

        # 4. Embed only new posts
        texts = [r["full_text"] for r in new_posts]
        embeddings = batch_embed(texts) if texts else []
        stats["posts_embedded"] = len(embeddings)

        # 5. Classify + score + upsert new posts (with fresh embeddings)
        for record, emb in zip(new_posts, embeddings):
            post_type, _ = classify_post(record["full_text"])
            sentiment_score = _vader.polarity_scores(record["full_text"])["compound"]

            record["post_type"] = post_type
            record["sentiment_score"] = sentiment_score
            record["embedding"] = f"[{','.join(str(x) for x in emb)}]"
            _upsert_post(record)

        # Update already-seen posts (upvotes/classification may have changed).
        # Never re-fetch or re-send the embedding — update only metadata columns.
        if seen_posts:
            UPDATE_META_SQL = """
                UPDATE reddit_posts SET
                    upvotes         = %(upvotes)s,
                    upvote_ratio    = %(upvote_ratio)s,
                    num_comments    = %(num_comments)s,
                    post_type       = %(post_type)s,
                    sentiment_score = %(sentiment_score)s,
                    scraped_at      = NOW()
                WHERE post_id = %(post_id)s
            """
            for record in seen_posts:
                post_type, _ = classify_post(record["full_text"])
                record["post_type"] = post_type
                record["sentiment_score"] = _vader.polarity_scores(record["full_text"])["compound"]
                with db.db_cursor(commit=True) as cur:
                    cur.execute(UPDATE_META_SQL, record)

        stats["posts_classified"] = len(cleaned)
        stats["status"] = "success"
        logger.info(
            "Ingestion complete: %d scraped, %d embedded, %d classified in %.1fs",
            stats["posts_scraped"],
            stats["posts_embedded"],
            stats["posts_classified"],
            time.time() - t_start,
        )

    except Exception as exc:
        stats["status"] = "partial" if stats["posts_scraped"] > 0 else "failed"
        stats["error_msg"] = str(exc)
        logger.error("Ingestion error: %s", exc, exc_info=True)

    finally:
        _write_log(stats)

    stats["duration_seconds"] = round(time.time() - t_start, 1)
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_ingestion()
    print(result)
