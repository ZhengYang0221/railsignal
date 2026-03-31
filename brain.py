"""
brain.py — Service 3: OpenAI embedding generation and pgvector semantic search.

Supports cross-lingual queries: a Mandarin query will retrieve English posts
because OpenAI's embedding space is multilingual.
"""

import logging
import os
import time
from typing import Optional

import openai
from dotenv import load_dotenv

import db

load_dotenv(override=True)

logger = logging.getLogger("railsignal.brain")

openai.api_key = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 500  # OpenAI accepts up to 2048; 500 cuts round-trips significantly
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles on each retry)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI Embeddings API with exponential backoff retry.
    Returns a list of 1536-dim float vectors.
    """
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError as exc:
            logger.warning("Rate limit hit (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
        except openai.APIError as exc:
            logger.warning("OpenAI API error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"OpenAI embedding failed after {MAX_RETRIES} attempts.")


def get_embedding(text: str) -> list[float]:
    """Embed a single text string and return a 1536-dim vector."""
    return _embed_with_retry([text])[0]


def batch_embed(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of texts in chunks of BATCH_SIZE.
    Returns vectors in the same order as input.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        logger.info(
            "Embedding batch %d/%d (%d texts) ...",
            i // BATCH_SIZE + 1,
            -(-len(texts) // BATCH_SIZE),
            len(chunk),
        )
        embeddings = _embed_with_retry(chunk)
        all_embeddings.extend(embeddings)
    return all_embeddings


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

SEARCH_SQL = """
SELECT
    post_id,
    title,
    author,
    full_text,
    upvotes,
    post_type,
    version_tag,
    sentiment_score,
    created_utc,
    1 - (embedding <=> %s::vector) AS similarity
FROM reddit_posts
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def semantic_search(query: str, top_k: int = 10, min_similarity: float = 0.0) -> list[dict]:
    """
    Embed query (any language) and run pgvector cosine similarity search.
    Returns up to top_k posts sorted by descending similarity.
    Cross-lingual: Mandarin query → English results.

    Args:
        query: Search query in any language.
        top_k: Maximum number of results to return.
        min_similarity: Minimum cosine similarity threshold (0.0–1.0).
                        Results below this score are discarded. A value of
                        0.45–0.55 is recommended to filter truly irrelevant posts.
    """
    query_vector = get_embedding(query)
    vector_str = f"[{','.join(str(x) for x in query_vector)}]"

    # Fetch more candidates than top_k so filtering doesn't leave us empty-handed
    fetch_k = top_k * 3 if min_similarity > 0 else top_k
    rows = db.execute_query(SEARCH_SQL, (vector_str, vector_str, fetch_k))

    results = []
    for row in rows:
        row = dict(row)
        if row.get("created_utc") and not isinstance(row["created_utc"], str):
            row["created_utc"] = row["created_utc"].isoformat()
        row["similarity"] = round(float(row.get("similarity", 0)), 4)
        if row["similarity"] >= min_similarity:
            results.append(row)

    # Respect the original top_k cap after filtering
    results = results[:top_k]

    logger.info(
        "Semantic search for %r: %d/%d results above sim=%.2f (top=%.3f)",
        query[:50],
        len(results),
        fetch_k,
        min_similarity,
        results[0]["similarity"] if results else 0,
    )
    return results
