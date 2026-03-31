"""
rag_engine.py — Service 4: RAG chain for bilingual stakeholder Q&A.

Accepts questions in English or Mandarin.
Returns dual-language executive summaries with cited upvote data.
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import openai
from dotenv import load_dotenv

from brain import semantic_search

load_dotenv(override=True)

logger = logging.getLogger("railsignal.rag_engine")

GPT_MODEL = "gpt-5.4"
MAX_RETRIES = 3
RETRY_DELAY = 2
RECENCY_DAYS = 30
MIN_UPVOTES = 5

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Used by The Bridge tab: bilingual bullet-point executive summary
SYSTEM_PROMPT = """\
You are a live-ops intelligence analyst for a AAA mobile game.
You will receive community feedback from Reddit and a stakeholder question.

Your response MUST contain exactly two sections with these exact headers:

[ENGLISH SUMMARY]
- 3-5 bullet point executive summary
- Identify any "Sentiment Defects": bugs, balance complaints, or gacha frustration
- Cite upvote counts for credibility (e.g., "892 upvotes")

[中文摘要]
- Same content in simplified Mandarin Chinese for Chinese-speaking stakeholders
- Maintain professional tone suitable for internal reporting
- Use bullet points (以 • 开头)
"""

# Used by the Analyst Q&A tab: direct, confident answers to stakeholder questions
QA_SYSTEM_PROMPT = """\
You are a live-ops intelligence analyst for a AAA mobile game answering a direct question from a business stakeholder.
You have access to recent community feedback scraped from Reddit.

Answer the question directly and confidently. Ground every claim in the data provided.
Reference specific evidence: upvote counts, recurring themes, post frequency.

Your response MUST contain exactly two sections with these exact headers:

[ENGLISH SUMMARY]
A direct, confident answer in 3-5 sentences. No vague hedging.
Lead with the key finding, then support it with data from the posts.
Flag any actionable issues the team should address.

[中文摘要]
Same answer in simplified Mandarin Chinese.
保持专业、直接的语气，适合内部汇报。
"""

USER_TEMPLATE = """\
Community data:
{context}

Stakeholder question: {question}
"""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(posts: list[dict]) -> str:
    """Format retrieved posts into a numbered context block for the prompt."""
    lines: list[str] = []
    for i, post in enumerate(posts, 1):
        lines.append(
            f"[Post {i}] upvotes={post.get('upvotes', 0)} "
            f"type={post.get('post_type', 'unknown')} "
            f"similarity={post.get('similarity', 0):.2f}\n"
            f"{post.get('full_text', '')[:400]}"
        )
    return "\n\n".join(lines)


def extract_defects(summary: str) -> list[str]:
    """
    Parse the GPT output to extract mentioned defect categories.
    Returns a list of matched defect labels.
    """
    defect_map = {
        "bug_report": ["bug", "crash", "broken", "glitch", "error", "fix"],
        "balance_complaint": ["balance", "op", "overpowered", "nerf", "buff", "unbalanced"],
        "gacha_frustration": ["gacha", "pity", "50/50", "f2p", "whale", "jades"],
    }
    summary_lower = summary.lower()
    found: list[str] = []
    for defect, keywords in defect_map.items():
        if any(kw in summary_lower for kw in keywords):
            found.append(defect)
    return found


# ---------------------------------------------------------------------------
# GPT call with retry
# ---------------------------------------------------------------------------

def _call_gpt(messages: list[dict]) -> str:
    """Call gpt-5.4 with exponential backoff retry. Returns the text response."""
    delay = RETRY_DELAY
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=GPT_MODEL,
                input=messages,
                temperature=0.3,
                max_output_tokens=1000,
            )
            return response.output_text
        except openai.RateLimitError as exc:
            logger.warning("GPT rate limit (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
        except openai.APIError as exc:
            logger.warning("GPT API error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"gpt-5.4 call failed after {MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Section parser
# ---------------------------------------------------------------------------

def _parse_sections(raw: str) -> tuple[str, str]:
    """
    Split GPT output into (english_summary, chinese_summary).
    Falls back gracefully if section headers are missing.
    """
    eng_match = re.search(
        r"\[ENGLISH SUMMARY\](.*?)(?=\[中文摘要\]|$)", raw, re.DOTALL | re.IGNORECASE
    )
    zh_match = re.search(r"\[中文摘要\](.*?)$", raw, re.DOTALL)

    english = eng_match.group(1).strip() if eng_match else raw.strip()
    chinese = zh_match.group(1).strip() if zh_match else ""
    return english, chinese


# ---------------------------------------------------------------------------
# Public RAG entry point
# ---------------------------------------------------------------------------

def run_rag_query(question: str, mode: str = "summary") -> dict:
    """
    Full RAG pipeline:
    1. Semantic search for relevant posts
    2. Filter by recency and upvote threshold
    3. Build prompt and call GPT with the appropriate system prompt
    4. Parse bilingual output

    Args:
        question: The stakeholder question (English or Mandarin).
        mode: "summary" for bullet-point executive summary (The Bridge),
              "qa" for direct answer style (Analyst tab).

    Returns:
        {
            english_summary: str,
            chinese_summary: str,
            defects_found: list[str],
            source_posts: list[dict],
            retrieved_count: int
        }
    """
    # 1. Retrieve
    candidates = semantic_search(question, top_k=20)

    # 2. Filter by recency (prefer last 30 days) and upvotes
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RECENCY_DAYS)
    filtered: list[dict] = []
    for post in candidates:
        created = post.get("created_utc", "")
        if created:
            try:
                dt = datetime.fromisoformat(created)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    continue
            except ValueError:
                pass
        if post.get("upvotes", 0) >= MIN_UPVOTES:
            filtered.append(post)

    # If strict filtering yields too few, fall back to top candidates
    source_posts = filtered[:10] if len(filtered) >= 3 else candidates[:10]

    # 3. Build prompt — choose system prompt based on mode
    context = format_context(source_posts)
    system = QA_SYSTEM_PROMPT if mode == "qa" else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]

    # 4. Call GPT
    raw_output = _call_gpt(messages)
    logger.info("RAG query answered. Output length: %d chars", len(raw_output))

    english_summary, chinese_summary = _parse_sections(raw_output)
    defects = extract_defects(raw_output)

    # Trim source posts for response payload.
    # Sort by upvotes descending so that the posts GPT is most likely to have
    # cited prominently (high-engagement posts) appear first in the source list.
    slim_sources = sorted(
        [
            {
                "post_id": p.get("post_id"),
                "text": (p.get("full_text") or "")[:300],
                "upvotes": p.get("upvotes") or 0,
                "similarity": p.get("similarity"),
            }
            for p in source_posts
        ],
        key=lambda x: x["upvotes"],
        reverse=True,
    )

    return {
        "english_summary": english_summary,
        "chinese_summary": chinese_summary,
        "defects_found": defects,
        "source_posts": slim_sources,
        "retrieved_count": len(source_posts),
    }
