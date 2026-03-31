"""
db.py — Shared PostgreSQL connection pool and query helpers.
All services import from here; no raw psycopg2 elsewhere.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger("railsignal.db")

# ---------------------------------------------------------------------------
# Connection pool (initialised lazily on first use)
# ---------------------------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dbname=os.getenv("POSTGRES_DB", "railsignal"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
        )
        logger.info("DB connection pool created (min=2, max=10).")
    return _pool


@contextmanager
def db_cursor(
    commit: bool = False,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    Context manager yielding a DictCursor from the connection pool.
    Rolls back automatically on exception; commits if commit=True.
    Returns the connection to the pool on exit.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
            if commit:
                conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def execute_query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT query and return all rows as a list of dicts."""
    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def execute_write(sql: str, params: tuple = ()) -> None:
    """Run an INSERT / UPDATE / DELETE query and commit."""
    with db_cursor(commit=True) as cur:
        cur.execute(sql, params)


def execute_many(sql: str, params_list: list[tuple]) -> None:
    """Batch-execute a write query (e.g. bulk insert)."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def health_check() -> bool:
    """Return True if the DB is reachable, False otherwise."""
    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False
