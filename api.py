"""
api.py — Service 5: FastAPI REST backend.

Start with:
    uvicorn api:app --reload --port 8000
"""

import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import db
from brain import semantic_search
from ingestor import run_ingestion
from ml_engine import load_models
from patch_schedule import get_patch_schedule, versions_for_range
from rag_engine import run_rag_query

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Background job registry (in-process, single-server)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [API] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("railsignal.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RailSignal API",
    description="Gaming community intelligence — bilingual (EN/ZH) insights from r/HonkaiStarRail",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    """Load ML models and patch schedule once at startup."""
    logger.info("Loading ML models ...")
    load_models()
    logger.info("Models ready.")
    logger.info("Loading patch schedule ...")
    get_patch_schedule()
    logger.info("Patch schedule ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RagRequest(BaseModel):
    question: str
    mode: str = "summary"  # "summary" (The Bridge) or "qa" (Analyst tab)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Health check — confirms DB connectivity and model readiness."""
    db_ok = db.health_check()
    from ml_engine import _classifier, _regressor
    models_loaded = _classifier is not None and _regressor is not None
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "connected" if db_ok else "disconnected",
        "models": "loaded" if models_loaded else "heuristic_mode",
    }


@app.post("/ingest")
def ingest() -> dict:
    """
    Scrape hot + new feeds (last 1-2 weeks of activity).
    Blocks until complete (~2–5 min).
    """
    logger.info("Standard ingestion triggered via API.")
    stats = run_ingestion(mode="standard")
    if stats.get("status") == "failed":
        raise HTTPException(status_code=500, detail=stats.get("error_msg", "Ingestion failed"))
    return {
        "status": stats["status"],
        "posts_scraped": stats.get("posts_scraped", 0),
        "posts_embedded": stats.get("posts_embedded", 0),
        "posts_classified": stats.get("posts_classified", 0),
        "duration_seconds": stats.get("duration_seconds", 0),
    }


def _run_history_job(job_id: str) -> None:
    """Worker function executed in a background thread."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        stats = run_ingestion(mode="history")
        with _jobs_lock:
            _jobs[job_id].update({"status": "completed", "result": stats})
        logger.info("History job %s completed: %d posts", job_id, stats.get("posts_scraped", 0))
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({"status": "failed", "error": str(exc)})
        logger.error("History job %s failed: %s", job_id, exc)


@app.post("/ingest-history")
def ingest_history() -> dict:
    """
    Start a background historical scrape (top/year + per-version searches).
    Returns immediately with a job_id — poll /ingest-status/{job_id} for progress.
    Only one history job can run at a time.
    """
    with _jobs_lock:
        for job in _jobs.values():
            if job["status"] in ("pending", "running"):
                logger.info("History job already running: %s", job["job_id"])
                return {"status": "already_running", "job_id": job["job_id"]}

        job_id = uuid.uuid4().hex[:8]
        _jobs[job_id] = {"job_id": job_id, "status": "pending"}

    thread = threading.Thread(target=_run_history_job, args=(job_id,), daemon=True)
    thread.start()
    logger.info("History job %s started in background.", job_id)
    return {"status": "started", "job_id": job_id}


@app.get("/ingest-status/{job_id}")
def ingest_status(job_id: str) -> dict:
    """Poll the status of a background ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job


@app.get("/sentiment-trend")
def sentiment_trend(
    days: int = Query(7, ge=1, le=90),
    version_tag: Optional[str] = Query(None),
) -> dict:
    """
    Daily average sentiment score and post volume over the last N days.
    Optional version_tag filter.
    """
    since = datetime.utcnow() - timedelta(days=days)

    base_sql = """
        SELECT
            DATE(created_utc)       AS date,
            AVG(sentiment_score)    AS avg_sentiment,
            COUNT(*)                AS post_count
        FROM reddit_posts
        WHERE created_utc >= %s
          AND sentiment_score IS NOT NULL
    """
    params: list = [since]

    if version_tag:
        base_sql += " AND version_tag = %s"
        params.append(version_tag)

    base_sql += " GROUP BY DATE(created_utc) ORDER BY date ASC"

    rows = db.execute_query(base_sql, tuple(params))
    trend = [
        {
            "date": str(row["date"]),
            "avg_sentiment": round(float(row["avg_sentiment"] or 0), 4),
            "post_count": int(row["post_count"]),
        }
        for row in rows
    ]
    return {"trend": trend}


@app.post("/rag-query")
def rag_query(body: RagRequest) -> dict:
    """
    Core RAG endpoint. Accepts English or Mandarin question.
    mode="summary" returns a bullet-point executive summary (used by The Bridge).
    mode="qa" returns a direct stakeholder answer (used by the Analyst tab).
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    if body.mode not in ("summary", "qa"):
        raise HTTPException(status_code=422, detail="mode must be 'summary' or 'qa'.")

    logger.info("RAG query [mode=%s]: %r", body.mode, body.question[:80])
    try:
        result = run_rag_query(body.question, mode=body.mode)
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "english_summary": result["english_summary"],
        "chinese_summary": result["chinese_summary"],
        "defects_found": result["defects_found"],
        "source_posts": result["source_posts"],
        "retrieved_count": result["retrieved_count"],
    }


@app.get("/version-compare")
def version_compare(
    version_a: str = Query(...),
    version_b: str = Query(...),
) -> dict:
    """
    Compare two patch versions by avg sentiment, post count, and defect rates.
    """
    sql = """
        SELECT
            version_tag,
            AVG(sentiment_score)                                AS avg_sentiment,
            COUNT(*)                                            AS post_count,
            SUM(CASE WHEN post_type != 'positive_feedback'
                          AND post_type != 'general_discussion'
                     THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS defect_rate,
            MODE() WITHIN GROUP (ORDER BY post_type)            AS top_defect
        FROM reddit_posts
        WHERE version_tag = ANY(%s)
          AND sentiment_score IS NOT NULL
        GROUP BY version_tag
        ORDER BY version_tag;
    """
    rows = db.execute_query(sql, ([version_a, version_b],))

    comparison = [
        {
            "version": row["version_tag"],
            "avg_sentiment": round(float(row["avg_sentiment"] or 0), 4),
            "post_count": int(row["post_count"]),
            "top_defect": row["top_defect"] or "none",
            "defect_rate": round(float(row["defect_rate"] or 0), 4),
        }
        for row in rows
    ]
    return {"comparison": comparison}


@app.get("/date-range-compare")
def date_range_compare(
    start_a: date = Query(..., description="Period A start date (YYYY-MM-DD)"),
    end_a:   date = Query(..., description="Period A end date (YYYY-MM-DD)"),
    start_b: date = Query(..., description="Period B start date (YYYY-MM-DD)"),
    end_b:   date = Query(..., description="Period B end date (YYYY-MM-DD)"),
) -> dict:
    """
    Compare two arbitrary date ranges by avg sentiment, post volume, and defect rate.
    Auto-labels each period with any overlapping HSR patch version(s).
    """
    sql = """
        SELECT
            AVG(sentiment_score)                                          AS avg_sentiment,
            COUNT(*)                                                      AS post_count,
            SUM(CASE WHEN post_type NOT IN ('positive_feedback', 'general_discussion')
                     THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0)     AS defect_rate,
            MODE() WITHIN GROUP (ORDER BY post_type)                      AS top_defect
        FROM reddit_posts
        WHERE created_utc >= %s
          AND created_utc <= %s
          AND sentiment_score IS NOT NULL
    """

    def _fetch(start: date, end: date) -> dict:
        # include end day fully (up to midnight of next day)
        end_dt = datetime.combine(end, datetime.max.time())
        start_dt = datetime.combine(start, datetime.min.time())
        rows = db.execute_query(sql, (start_dt, end_dt))
        row = rows[0] if rows else {}
        return {
            "avg_sentiment": round(float(row.get("avg_sentiment") or 0), 4),
            "post_count": int(row.get("post_count") or 0),
            "defect_rate": round(float(row.get("defect_rate") or 0), 4),
            "top_defect": row.get("top_defect") or "none",
        }

    data_a = _fetch(start_a, end_a)
    data_b = _fetch(start_b, end_b)

    versions_a = versions_for_range(start_a, end_a)
    versions_b = versions_for_range(start_b, end_b)

    label_a = f"v{', v'.join(versions_a)}" if versions_a else "Custom Period"
    label_b = f"v{', v'.join(versions_b)}" if versions_b else "Custom Period"

    comparison = [
        {
            "label": label_a,
            "start": str(start_a),
            "end": str(end_a),
            **data_a,
        },
        {
            "label": label_b,
            "start": str(start_b),
            "end": str(end_b),
            **data_b,
        },
    ]
    return {"comparison": comparison}


@app.get("/top-posts-by-type")
def top_posts_by_type(days: int = Query(7, ge=1, le=90)) -> dict:
    """
    Returns the highest-upvoted post for each non-positive post type
    in the last N days. Used to surface real examples in Community Health.
    """
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.execute_query(
        """
        SELECT DISTINCT ON (post_type)
            post_id,
            title,
            author,
            full_text,
            upvotes,
            post_type,
            created_utc
        FROM reddit_posts
        WHERE post_type NOT IN ('positive_feedback', 'general_discussion')
          AND created_utc >= %s
        ORDER BY post_type, upvotes DESC
        """,
        (since,),
    )
    posts = []
    for row in rows:
        r = dict(row)
        if r.get("created_utc") and not isinstance(r["created_utc"], str):
            r["created_utc"] = r["created_utc"].isoformat()
        posts.append(r)
    return {"posts": posts}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.45, ge=0.0, le=1.0),
) -> dict:
    """
    Semantic search over Reddit posts. Supports English and Mandarin queries.
    Powers 'The Bridge' feature in the Streamlit dashboard.
    Results with similarity below min_similarity are filtered out.
    """
    logger.info("Semantic search: %r top_k=%d min_sim=%.2f", q[:60], top_k, min_similarity)
    try:
        results = semantic_search(q, top_k=top_k, min_similarity=min_similarity)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"query": q, "results": results}


@app.get("/versions")
def list_versions() -> dict:
    """
    Return all distinct version tags in the DB with post counts.
    Used to populate the Version Compare dropdowns.
    """
    rows = db.execute_query(
        """
        SELECT version_tag, COUNT(*) AS post_count
        FROM reddit_posts
        WHERE version_tag IS NOT NULL
        GROUP BY version_tag
        ORDER BY version_tag
        """
    )
    return {
        "versions": [
            {"version": r["version_tag"], "post_count": int(r["post_count"])}
            for r in rows
        ]
    }


@app.get("/patch-schedule")
def patch_schedule() -> dict:
    """
    Return the full HSR patch schedule (version, start, end dates).
    Used by the dashboard to auto-label date ranges and populate quick-select buttons.
    """
    schedule = get_patch_schedule()
    return {
        "schedule": [
            {
                "version": e["version"],
                "start": str(e["start"]),
                "end": str(e["end"]),
            }
            for e in schedule
        ]
    }


@app.get("/post-type-distribution")
def post_type_distribution(days: int = Query(30, ge=1, le=365)) -> dict:
    """Breakdown of post_type counts over the last N days."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.execute_query(
        """
        SELECT post_type, COUNT(*) AS count
        FROM reddit_posts
        WHERE created_utc >= %s AND post_type IS NOT NULL
        GROUP BY post_type ORDER BY count DESC
        """,
        (since,),
    )
    return {
        "distribution": [
            {"post_type": r["post_type"], "count": int(r["count"])} for r in rows
        ]
    }
