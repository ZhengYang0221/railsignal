"""
app.py — Service 6: Streamlit stakeholder dashboard.

Start with:
    streamlit run app.py

All data is fetched from the FastAPI backend at localhost:8000.
No direct DB or ML imports.
"""

import os
import time

import httpx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

API_BASE = os.getenv("API_BASE", f"http://localhost:{os.getenv('API_PORT', 8000)}")
_CACHE_TTL = 300  # seconds before cached API data is considered stale

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RailSignal | 社区智能平台",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch(path: str, params: dict | None = None) -> dict | None:
    """Raw HTTP GET — no Streamlit side-effects. Returns None on any error."""
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_get(path: str, params: dict | None = None) -> dict | None:
    """GET with Streamlit error display on failure."""
    result = _fetch(path, params)
    if result is None:
        st.error(f"API error [{path}]: could not reach backend.")
    return result


def api_post(path: str, payload: dict) -> dict | None:
    """POST request to the FastAPI backend."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error [{path}]: {exc}")
        return None


def cached_get(key: str, path: str, params: dict | None = None, ttl: int = _CACHE_TTL) -> dict | None:
    """
    Cached GET using session_state. Re-fetches only when TTL has expired.
    Avoids redundant API calls on every Streamlit rerun (e.g. button clicks).
    """
    now = time.time()
    ts_key = f"{key}__ts"
    if key not in st.session_state or now - st.session_state.get(ts_key, 0) > ttl:
        data = _fetch(path, params)
        if data is not None:
            st.session_state[key] = data
            st.session_state[ts_key] = now
    return st.session_state.get(key)


def reddit_url(post_id: str) -> str:
    return f"https://www.reddit.com/r/HonkaiStarRail/comments/{post_id}/"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POST_TYPE_COLORS = {
    "bug_report":         "#EF4444",
    "balance_complaint":  "#F97316",
    "gacha_frustration":  "#A855F7",
    "positive_feedback":  "#22C55E",
    "general_discussion": "#3B82F6",
}

POST_TYPE_LABELS = {
    "bug_report":         "🐛 Bug Report",
    "balance_complaint":  "⚖️ Balance",
    "gacha_frustration":  "💎 Gacha",
    "positive_feedback":  "✅ Positive",
    "general_discussion": "💬 General",
}

NEGATIVE_TYPES = {"bug_report", "balance_complaint", "gacha_frustration"}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0">
        <span style="font-size:2rem">🚂</span>
        <div>
            <h1 style="margin:0;font-size:1.8rem;font-weight:700">RailSignal</h1>
            <p style="margin:0;color:#6B7280;font-size:0.9rem">
                Multilingual Gaming Intelligence Platform &nbsp;|&nbsp; 多语言游戏社区智能平台
            </p>
        </div>
    </div>
    <hr style="margin:12px 0 20px 0;border-color:#E5E7EB"/>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Global: history job status banner (shown on all tabs while job runs)
# ---------------------------------------------------------------------------
if "history_job_id" in st.session_state:
    job_id = st.session_state["history_job_id"]
    job_data = _fetch(f"/ingest-status/{job_id}")

    if job_data is None:
        # API unreachable — don't clear, user can retry
        st.warning("⏳ Historical scrape started but API status is unreachable. Check back shortly.")
    elif job_data.get("status") in ("pending", "running"):
        st.info(
            f"📚 Historical scrape in progress (job `{job_id}`) — "
            "this takes 10–20 min. Page auto-refreshes every 15 s."
        )
        time.sleep(15)
        st.rerun()
    elif job_data.get("status") == "completed":
        result = job_data.get("result", {})
        st.success(
            f"📚 Historical scrape complete — "
            f"{result.get('posts_scraped', 0):,} posts scraped, "
            f"{result.get('posts_embedded', 0):,} embedded "
            f"({result.get('duration_seconds', 0):.0f}s). "
            "Run **Version Compare** now."
        )
        del st.session_state["history_job_id"]
        for k in ["trend_data", "dist_data", "top_posts_data", "versions_data", "schedule_data"]:
            st.session_state.pop(k, None)
            st.session_state.pop(f"{k}__ts", None)
    elif job_data.get("status") == "failed":
        st.error(f"📚 Historical scrape failed: {job_data.get('error', 'unknown error')}")
        del st.session_state["history_job_id"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_home, tab_bridge, tab_versions, tab_qa = st.tabs(
    ["📊 Community Health", "🌉 The Bridge", "📅 Version Compare", "💼 Ask the Analyst"]
)


# ===========================================================================
# TAB 1 — Community Health
# ===========================================================================
with tab_home:
    col_header, col_btn = st.columns([4, 2])
    with col_header:
        st.subheader("Community Health")
        st.caption("7-day sentiment, post volume, and top community issues from r/HonkaiStarRail")
    with col_btn:
        st.write("")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🔄 Refresh", use_container_width=True, help="Scrapes the latest hot + new posts (~2–3 min)"):
                for k in ["trend_data", "dist_data", "top_posts_data", "weekly_digest"]:
                    st.session_state.pop(k, None)
                    st.session_state.pop(f"{k}__ts", None)
                with st.spinner("Ingesting latest posts ..."):
                    result = api_post("/ingest", {})
                    if result:
                        st.success(
                            f"{result.get('posts_scraped', 0)} posts "
                            f"in {result.get('duration_seconds', 0):.0f}s"
                        )
        with bcol2:
            history_running = "history_job_id" in st.session_state
            if st.button(
                "📚 History" if not history_running else "⏳ Running...",
                use_container_width=True,
                disabled=history_running,
                help="Scrapes top/year + per-version searches to populate multiple patch windows. Runs in the background (~10–20 min).",
            ):
                result = api_post("/ingest-history", {})
                if result and result.get("job_id"):
                    st.session_state["history_job_id"] = result["job_id"]
                    st.rerun()

    # Cached API calls — only re-fetched when TTL expires or cache is cleared
    trend_data     = cached_get("trend_data",      "/sentiment-trend",      {"days": 7})
    dist_data      = cached_get("dist_data",       "/post-type-distribution", {"days": 7})
    top_posts_data = cached_get("top_posts_data",  "/top-posts-by-type",    {"days": 7})

    if not (trend_data and trend_data.get("trend")):
        st.info("No data yet. Click **Refresh Data** to ingest posts.")
        st.stop()

    trend = trend_data["trend"]
    total_posts = sum(d["post_count"] for d in trend)
    avg_sentiment = (
        sum(d["avg_sentiment"] * d["post_count"] for d in trend) / total_posts
        if total_posts > 0 else 0
    )
    latest_sentiment = trend[-1]["avg_sentiment"] if trend else 0
    sentiment_delta  = latest_sentiment - trend[-2]["avg_sentiment"] if len(trend) >= 2 else 0

    defect_count = total_count = 0
    if dist_data:
        for item in dist_data.get("distribution", []):
            total_count += item["count"]
            if item["post_type"] in NEGATIVE_TYPES:
                defect_count += item["count"]
    defect_rate = (defect_count / total_count * 100) if total_count > 0 else 0

    top_issue_label = "—"
    top_issue_quote = ""
    if top_posts_data and top_posts_data.get("posts"):
        best = max(top_posts_data["posts"], key=lambda p: p.get("upvotes", 0))
        top_issue_label = POST_TYPE_LABELS.get(best["post_type"], best["post_type"])
        top_issue_quote = (best.get("title") or best.get("full_text") or "")[:80]

    # --- KPI cards ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts (7d)", f"{total_posts:,}")
    m2.metric("Avg Sentiment", f"{avg_sentiment:.2f}", f"{sentiment_delta:+.2f}")
    m3.metric("Defect Rate", f"{defect_rate:.1f}%")
    with m4:
        st.metric("Trending Issue", top_issue_label)
        if top_issue_quote:
            st.caption(f'"{top_issue_quote}..."')

    st.write("")

    # --- AI Weekly Digest (cached in session_state, persists across reruns) ---
    with st.expander("🤖 AI Weekly Digest — What's driving sentiment this week?", expanded=True):
        if "weekly_digest" not in st.session_state:
            with st.spinner("Generating AI insight ..."):
                digest = api_post("/rag-query", {
                    "question": (
                        "What are the top 3 community issues this week? "
                        "What is driving positive and negative sentiment? "
                        "What should the live-ops team prioritise?"
                    ),
                    "mode": "qa",
                })
                st.session_state["weekly_digest"] = digest

        digest = st.session_state.get("weekly_digest")
        if digest:
            dcol_en, dcol_zh = st.columns(2)
            with dcol_en:
                st.markdown("**English**")
                st.markdown(digest.get("english_summary", "—"))
            with dcol_zh:
                st.markdown("**中文**")
                st.markdown(digest.get("chinese_summary", "—"))

            # Source posts as reference links
            sources = digest.get("source_posts", [])
            if sources:
                st.markdown("**Referenced posts:**")
                for s in sources:
                    pid = s.get("post_id", "")
                    snippet = (s.get("text") or "")[:80]
                    upvotes = s.get("upvotes", 0)
                    if pid:
                        st.markdown(
                            f"- [⬆️ {upvotes:,} &nbsp; {snippet}...]({reddit_url(pid)})"
                        )
        else:
            st.warning("Could not generate digest. Is the API running?")

    st.write("")

    # --- Sentiment + Volume dual-axis chart ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=[d["date"] for d in trend],
            y=[d["post_count"] for d in trend],
            name="Post Volume",
            marker_color="rgba(99,102,241,0.15)",
            hovertemplate="%{y} posts<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=[d["date"] for d in trend],
            y=[d["avg_sentiment"] for d in trend],
            name="Avg Sentiment",
            mode="lines+markers",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="%{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.update_layout(
        title="Daily Sentiment Score & Post Volume (7 days)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=40, b=20, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Sentiment Score", range=[-1, 1], secondary_y=False)
    fig.update_yaxes(title_text="Post Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Post type distribution + Top issues side by side ---
    chart_col, issues_col = st.columns([1, 1])

    with chart_col:
        if dist_data and dist_data.get("distribution"):
            dist = dist_data["distribution"]
            labels = [POST_TYPE_LABELS.get(d["post_type"], d["post_type"]) for d in dist]
            counts = [d["count"] for d in dist]
            colors = [POST_TYPE_COLORS.get(d["post_type"], "#9CA3AF") for d in dist]
            fig_dist = go.Figure(go.Bar(
                x=labels, y=counts,
                marker_color=colors,
                text=counts, textposition="outside",
            ))
            fig_dist.update_layout(
                title="Post Type Distribution (7 days)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(t=40, b=20, l=0, r=0),
                yaxis_title="Post Count",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    with issues_col:
        st.markdown("**Top Post per Complaint Category**")
        st.caption("Highest-upvoted post from each negative category this week")
        if top_posts_data and top_posts_data.get("posts"):
            for post in top_posts_data["posts"]:
                ptype   = post.get("post_type", "")
                label   = POST_TYPE_LABELS.get(ptype, ptype)
                color   = POST_TYPE_COLORS.get(ptype, "#9CA3AF")
                title   = post.get("title") or (post.get("full_text") or "")[:80]
                snippet = (post.get("full_text") or "")[:200]
                pid     = post.get("post_id", "")
                with st.expander(f"{label} — {post.get('upvotes', 0):,} upvotes"):
                    st.markdown(
                        f"<div style='border-left:3px solid {color};padding-left:10px'>"
                        f"<strong>{title}</strong><br/>"
                        f"<span style='font-size:0.85rem;color:#6B7280'>{snippet}...</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if pid:
                        st.markdown(f"[🔗 View on Reddit]({reddit_url(pid)})")
        else:
            st.info("No complaint posts found in the last 7 days.")


# ===========================================================================
# TAB 2 — The Bridge (Semantic Search + RAG)
# ===========================================================================
with tab_bridge:
    st.subheader("🌉 The Bridge — Cross-Lingual Semantic Search")
    st.caption(
        "Search r/HonkaiStarRail in English or Mandarin. "
        "Results include the original Reddit post and a Mandarin executive summary."
    )

    query = st.text_input(
        "Search query",
        placeholder="Type in English or 中文 ... e.g. '黄泉角色强度' or 'Acheron team comp'",
        label_visibility="collapsed",
    )

    if query.strip():
        with st.spinner("Searching and generating insights ..."):
            search_result = api_get("/search", {"q": query, "top_k": 10, "min_similarity": 0.45})
            rag_result    = api_post("/rag-query", {"question": query, "mode": "summary"})

        # --- Executive summary ---
        if rag_result:
            col_en, col_zh = st.columns(2)
            with col_en:
                st.markdown("**📋 Executive Summary (English)**")
                st.markdown(rag_result.get("english_summary", "—"))
            with col_zh:
                st.markdown("**📋 执行摘要（中文）**")
                st.markdown(rag_result.get("chinese_summary", "—"))

            defects = rag_result.get("defects_found", [])
            if defects:
                badges = " ".join(f"`{POST_TYPE_LABELS.get(d, d)}`" for d in defects)
                st.markdown(f"**🚨 Sentiment Defects Detected:** {badges}")

            # Source post links for the executive summary
            sources = rag_result.get("source_posts", [])
            if sources:
                with st.expander("📎 Source posts used in this summary"):
                    for s in sources:
                        pid     = s.get("post_id", "")
                        snippet = (s.get("text") or "")[:100]
                        upvotes = s.get("upvotes", 0)
                        sim     = s.get("similarity", 0)
                        if pid:
                            st.markdown(
                                f"- [⬆️ {upvotes:,} &nbsp; 🎯 {sim:.2f} &nbsp; {snippet}...]"
                                f"({reddit_url(pid)})"
                            )

        st.divider()

        # --- Search results ---
        if search_result and search_result.get("results"):
            results = search_result["results"]
            st.markdown(f"**Found {len(results)} relevant posts:**")
            for post in results:
                post_type  = post.get("post_type", "general_discussion")
                type_badge = POST_TYPE_LABELS.get(post_type, post_type)
                sim        = post.get("similarity", 0)
                pid        = post.get("post_id", "")
                title      = post.get("title") or ""
                author     = post.get("author") or "unknown"
                created    = (post.get("created_utc") or "")[:10]
                snippet    = (post.get("full_text") or "")[:350]

                st.markdown(
                    f"**{type_badge}** &nbsp; ⬆️ {post.get('upvotes', 0):,} "
                    f"&nbsp; 🎯 {sim:.2f}"
                )
                if title:
                    st.markdown(f"**{title}**")
                st.markdown(
                    f"<div style='background:#F3F4F6;padding:10px;border-radius:8px;"
                    f"font-size:0.85rem;color:#374151'>{snippet}...</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"👤 u/{author} &nbsp; 📅 {created}")
                if pid:
                    st.markdown(f"[🔗 View on Reddit]({reddit_url(pid)})")
                st.write("")
        elif search_result is not None:
            st.info(
                "No sufficiently relevant posts found for this query. "
                "Try different keywords or run an ingestion to add more data."
            )


# ===========================================================================
# TAB 3 — Version Compare
# ===========================================================================
with tab_versions:
    st.subheader("📅 Version Compare")
    st.caption(
        "Compare community sentiment across two HSR patches. "
        "Posts are assigned to a patch based on when they were created — "
        "if a post's date falls within a patch window, it belongs to that patch."
    )

    # Use the full patch schedule as dropdown options (not DB version_tags).
    # This way every known patch is selectable regardless of what text posts contain.
    schedule_data = cached_get("schedule_data", "/patch-schedule", ttl=3600)
    schedule_list: list[dict] = schedule_data.get("schedule", []) if schedule_data else []

    if not schedule_list:
        st.info("Could not load patch schedule. Is the API running?")
    else:
        # Sort newest-first for the dropdown
        schedule_sorted = sorted(schedule_list, key=lambda x: x["start"], reverse=True)

        def _ver_label(p: dict) -> str:
            return f"v{p['version']}  —  {p['start']} → {p['end']}"

        options     = [_ver_label(p) for p in schedule_sorted]
        patch_keys  = [p["version"] for p in schedule_sorted]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("Version A", options, index=1, key="ver_sel_a")
            patch_a = schedule_sorted[options.index(sel_a)]
        with col_b:
            sel_b = st.selectbox("Version B", options, index=0, key="ver_sel_b")
            patch_b = schedule_sorted[options.index(sel_b)]

        st.write("")
        if st.button("Compare Versions →", use_container_width=False):
            if patch_a["version"] == patch_b["version"]:
                st.warning("Please select two different versions.")
            else:
                with st.spinner("Comparing ..."):
                    compare_data = api_get("/date-range-compare", {
                        "start_a": patch_a["start"],
                        "end_a":   patch_a["end"],
                        "start_b": patch_b["start"],
                        "end_b":   patch_b["end"],
                    })

                if compare_data and compare_data.get("comparison"):
                    comparison = compare_data["comparison"]

                    # Relabel with version names for clarity
                    if len(comparison) == 2:
                        comparison[0]["label"] = f"v{patch_a['version']}"
                        comparison[1]["label"] = f"v{patch_b['version']}"

                    st.markdown("### Comparison Summary")
                    cols = st.columns(len(comparison))
                    for col, v in zip(cols, comparison):
                        with col:
                            st.markdown(f"#### {v['label']}")
                            st.caption(f"📅 {v['start']} → {v['end']}")
                            st.metric("Avg Sentiment", f"{v['avg_sentiment']:.2f}")
                            st.metric("Posts", f"{v['post_count']:,}")
                            st.metric("Defect Rate", f"{v['defect_rate']*100:.1f}%")
                            st.metric(
                                "Top Defect",
                                POST_TYPE_LABELS.get(v["top_defect"], v["top_defect"]),
                            )

                    if len(comparison) == 2:
                        a, b = comparison[0], comparison[1]
                        sent_delta   = b["avg_sentiment"] - a["avg_sentiment"]
                        defect_delta = b["defect_rate"]   - a["defect_rate"]
                        count_delta  = b["post_count"]    - a["post_count"]

                        st.divider()
                        st.markdown(f"### Δ Change: {a['label']} → {b['label']}")
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Sentiment",   f"{b['avg_sentiment']:.2f}", f"{sent_delta:+.2f}", delta_color="normal")
                        d2.metric("Defect Rate", f"{b['defect_rate']*100:.1f}%", f"{defect_delta*100:+.1f}%", delta_color="inverse")
                        d3.metric("Post Volume", f"{b['post_count']:,}", f"{count_delta:+,}")

                        # Bar chart
                        fig_compare = go.Figure()
                        for v in comparison:
                            fig_compare.add_trace(go.Bar(
                                name=v["label"],
                                x=["Avg Sentiment", "Defect Rate (×10)"],
                                y=[v["avg_sentiment"], v["defect_rate"] * 10],
                                text=[f"{v['avg_sentiment']:.2f}", f"{v['defect_rate']*100:.1f}%"],
                                textposition="outside",
                            ))
                        fig_compare.update_layout(
                            barmode="group",
                            title=f"{a['label']} vs {b['label']} — Sentiment & Defect Rate",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=350,
                            margin=dict(t=40, b=20, l=0, r=0),
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)

                        if a["post_count"] == 0 or b["post_count"] == 0:
                            st.warning(
                                "One or both periods have 0 posts. This means the ingestor "
                                "was not running during that patch window, so no posts from "
                                "that period are in the database. "
                                "Posts are matched by creation date — only dates when the "
                                "ingestor was actively scraping will have data."
                            )
                        else:
                            # AI insight
                            st.divider()
                            with st.spinner("Generating AI comparison insight ..."):
                                insight = api_post("/rag-query", {
                                    "question": (
                                        f"Compare community sentiment and major issues between "
                                        f"version {patch_a['version']} and version {patch_b['version']}. "
                                        f"What changed and what should the team focus on?"
                                    ),
                                    "mode": "qa",
                                })
                            if insight:
                                icol_en, icol_zh = st.columns(2)
                                with icol_en:
                                    st.markdown("**🤖 AI Analysis (English)**")
                                    st.markdown(insight.get("english_summary", "—"))
                                with icol_zh:
                                    st.markdown("**🤖 AI 分析（中文）**")
                                    st.markdown(insight.get("chinese_summary", "—"))
                else:
                    st.warning("Could not retrieve comparison data. Check that the API is running.")


# ===========================================================================
# TAB 4 — Ask the Analyst (Stakeholder Q&A)
# ===========================================================================
with tab_qa:
    st.subheader("💼 Ask the Analyst")
    st.caption(
        "Ask any question about community sentiment, player feedback, or live-ops health. "
        "Answers are grounded in scraped Reddit data and returned in English and Mandarin."
    )

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    # Handle suggestion button clicks — set session_state["qa_input"] BEFORE rendering
    # the text_input so the widget picks up the value without needing value= override.
    SUGGESTIONS = [
        "What are players complaining about most this week?",
        "How was the reception for the latest character?",
        "Are bug reports increasing or decreasing?",
        "What features are players requesting?",
        "What is causing the most gacha frustration right now?",
    ]

    st.markdown("**Suggested questions:**")
    sug_cols = st.columns(len(SUGGESTIONS))
    for i, sug in enumerate(SUGGESTIONS):
        with sug_cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                # Pre-fill the input AND mark for auto-submit on next render
                st.session_state["qa_input"] = sug
                st.session_state["qa_auto_submit"] = True

    # Pop the auto-submit flag BEFORE rendering the rest of the UI
    auto_submit = st.session_state.pop("qa_auto_submit", False)

    st.write("")

    # Text input — key-driven, no value= override (avoids the "reset on rerun" bug)
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is driving negative sentiment this week?",
        label_visibility="collapsed",
        key="qa_input",
    )

    col_ask, col_clear = st.columns([5, 1])
    with col_ask:
        ask_clicked = st.button("Ask →", use_container_width=True, type="primary")
    with col_clear:
        if st.button("Clear history", use_container_width=True):
            st.session_state["qa_history"] = []
            st.rerun()

    if (ask_clicked or auto_submit) and question.strip():
        with st.spinner("Analysing community data ..."):
            result = api_post("/rag-query", {"question": question.strip(), "mode": "qa"})
        if result:
            st.session_state["qa_history"].insert(0, {
                "question": question.strip(),
                "english":  result.get("english_summary", ""),
                "chinese":  result.get("chinese_summary", ""),
                "defects":  result.get("defects_found", []),
                "sources":  result.get("source_posts", []),
            })

    # --- Conversation history ---
    for i, entry in enumerate(st.session_state["qa_history"]):
        st.markdown(
            f"<div style='background:#F9FAFB;border-radius:8px;padding:12px 16px;"
            f"border-left:4px solid #6366F1;margin-bottom:8px'>"
            f"<strong>Q:</strong> {entry['question']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        ans_en, ans_zh = st.columns(2)
        with ans_en:
            st.markdown("**Answer (English)**")
            st.markdown(entry["english"] or "—")
        with ans_zh:
            st.markdown("**回答（中文）**")
            st.markdown(entry["chinese"] or "—")

        if entry.get("defects"):
            badges = " ".join(f"`{POST_TYPE_LABELS.get(d, d)}`" for d in entry["defects"])
            st.markdown(f"🚨 **Issues flagged:** {badges}")

        # Source links
        sources = entry.get("sources", [])
        if sources:
            with st.expander("📎 Source posts"):
                for s in sources:
                    pid     = s.get("post_id", "")
                    snippet = (s.get("text") or "")[:100]
                    upvotes = s.get("upvotes", 0)
                    if pid:
                        st.markdown(
                            f"- [⬆️ {upvotes:,} &nbsp; {snippet}...]({reddit_url(pid)})"
                        )

        if i < len(st.session_state["qa_history"]) - 1:
            st.divider()

    if not st.session_state["qa_history"]:
        st.info("Ask a question above to get started.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<hr style='border-color:#E5E7EB;margin-top:40px'/>"
    "<p style='text-align:center;color:#9CA3AF;font-size:0.8rem'>"
    "RailSignal v2.0 · Powered by OpenAI gpt-5.4 + pgvector · "
    "Data: r/HonkaiStarRail"
    "</p>",
    unsafe_allow_html=True,
)
