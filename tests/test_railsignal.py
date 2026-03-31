"""
tests/test_railsignal.py

Full test suite for RailSignal MVP.
Run with:  pytest tests/ -v

External dependencies (Reddit, OpenAI, PostgreSQL) are fully mocked
so these tests run without any credentials or running services.
"""

import importlib
import json
import sys
import types
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared mock DB rows
# ---------------------------------------------------------------------------
MOCK_POSTS = [
    {
        "post_id": "abc123",
        "full_text": "The new patch 2.3 has a huge bug with Acheron crash on startup",
        "title": "Bug in 2.3",
        "body": "Crash on startup",
        "upvotes": 500,
        "upvote_ratio": 0.95,
        "num_comments": 80,
        "version_tag": "2.3",
        "post_type": "bug_report",
        "sentiment_score": -0.4,
        "similarity": 0.91,
        "created_utc": "2026-03-10T12:00:00",
    },
    {
        "post_id": "def456",
        "full_text": "Lost my 50/50 again, gacha is so predatory, need more jades",
        "title": "Gacha is pain",
        "body": "",
        "upvotes": 342,
        "upvote_ratio": 0.88,
        "num_comments": 55,
        "version_tag": "2.2",
        "post_type": "gacha_frustration",
        "sentiment_score": -0.6,
        "similarity": 0.85,
        "created_utc": "2026-03-08T09:00:00",
    },
    {
        "post_id": "ghi789",
        "full_text": "Amazing story in 2.3! Love the new characters, best patch ever",
        "title": "Best patch",
        "body": "",
        "upvotes": 892,
        "upvote_ratio": 0.97,
        "num_comments": 120,
        "version_tag": "2.3",
        "post_type": "positive_feedback",
        "sentiment_score": 0.9,
        "similarity": 0.80,
        "created_utc": "2026-03-11T15:00:00",
    },
]


# ===========================================================================
# 1. db.py — connection helpers (pure logic, no real DB needed)
# ===========================================================================

class TestDbHelpers:
    """Test db.py utility functions using a mocked connection pool."""

    def setup_method(self):
        # Reset the module-level pool singleton before every test so each
        # test gets a clean slate and pool-creation code is exercised fresh.
        import db
        db._pool = None

    def teardown_method(self):
        import db
        db._pool = None

    def _mock_pool(self, fetchall_return=None):
        """Build a mock pool + connection + cursor and return (pool, cursor)."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_cursor = MagicMock()
        if fetchall_return is not None:
            mock_cursor.fetchall.return_value = fetchall_return
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_pool, mock_cursor

    def test_health_check_success(self):
        import db
        mock_pool, _ = self._mock_pool()
        with patch("db._get_pool", return_value=mock_pool):
            assert db.health_check() is True

    def test_health_check_failure(self):
        import db
        with patch("db._get_pool", side_effect=Exception("connection refused")):
            assert db.health_check() is False

    def test_execute_query_returns_list(self):
        import db
        mock_pool, _ = self._mock_pool(fetchall_return=[{"id": 1}, {"id": 2}])
        with patch("db._get_pool", return_value=mock_pool):
            result = db.execute_query("SELECT 1")
        assert isinstance(result, list)


# ===========================================================================
# 2. ingestor.py — pure functions (no network, no DB)
# ===========================================================================

class TestIngestorPureFunctions:
    """Tests for stateless helper functions in ingestor.py."""

    def setup_method(self):
        # Patch external deps before importing ingestor
        self.patches = [
            patch("brain.get_embedding", return_value=[0.1] * 1536),
            patch("brain.batch_embed", return_value=[[0.1] * 1536]),
            patch("ml_engine.load_models"),
            patch("ml_engine.classify_post", return_value=("bug_report", 0.9)),
            patch("db.db_cursor"),
        ]
        for p in self.patches:
            p.start()
        import ingestor
        self.ingestor = ingestor

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    def test_clean_text_strips_whitespace(self):
        result = self.ingestor.clean_text("  hello   world  ")
        assert result == "hello world"

    def test_clean_text_normalises_unicode(self):
        result = self.ingestor.clean_text("caf\u00e9")
        assert "caf" in result

    def test_clean_text_removes_null_bytes(self):
        result = self.ingestor.clean_text("hello\x00world")
        assert "\x00" not in result

    def test_clean_text_empty_string(self):
        assert self.ingestor.clean_text("") == ""

    def test_extract_version_tag_bracket_format(self):
        assert self.ingestor.extract_version_tag("[2.1] patch notes") == "2.1"

    def test_extract_version_tag_v_prefix(self):
        assert self.ingestor.extract_version_tag("v2.2 is out now") == "2.2"

    def test_extract_version_tag_patch_word(self):
        assert self.ingestor.extract_version_tag("patch 2.3 broke everything") == "2.3"

    def test_extract_version_tag_banner(self):
        assert self.ingestor.extract_version_tag("2.1 banner is live") == "2.1"

    def test_extract_version_tag_none(self):
        assert self.ingestor.extract_version_tag("no version here") is None

    def test_is_bot_automoderator(self):
        assert self.ingestor.is_bot("AutoModerator") is True

    def test_is_bot_bot_suffix(self):
        assert self.ingestor.is_bot("SomeRandomBot") is True

    def test_is_bot_real_user(self):
        assert self.ingestor.is_bot("regular_user_123") is False


# ===========================================================================
# 3. ml_engine.py — labelling heuristics + model train/infer
# ===========================================================================

class TestMlEngine:
    """Tests for ML classification heuristics and model pipelines."""

    def setup_method(self):
        self.patches = [patch("db.execute_query", return_value=[])]
        for p in self.patches:
            p.start()
        import ml_engine
        self.ml = ml_engine

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    # --- Seed labelling ---
    def test_seed_label_bug(self):
        assert self.ml.seed_label("there is a huge bug and crash in the game") == "bug_report"

    def test_seed_label_gacha(self):
        assert self.ml.seed_label("lost my 50/50 again and no more jades") == "gacha_frustration"

    def test_seed_label_balance(self):
        assert self.ml.seed_label("this character is completely overpowered and unfair") == "balance_complaint"

    def test_seed_label_positive(self):
        assert self.ml.seed_label("this game is amazing and I love everything") == "positive_feedback"

    def test_seed_label_general(self):
        assert self.ml.seed_label("just started the game today") == "general_discussion"

    # --- Fallback classify when no model loaded ---
    def test_classify_post_no_model_returns_heuristic(self):
        self.ml._classifier = None
        label, conf = self.ml.classify_post("game crashed bug broken")
        assert label == "bug_report"
        assert 0 <= conf <= 1

    # --- Fallback score when no model loaded ---
    def test_score_engagement_no_model_returns_zero(self):
        self.ml._regressor = None
        self.ml._reg_scaler = None
        score = self.ml.score_engagement({
            "full_text": "test", "num_comments": 10,
            "upvote_ratio": 0.9, "post_length": 50, "has_version_tag": 0
        })
        assert score == 0.0

    # --- Full train → infer cycle ---
    def test_train_classifier_and_predict(self):
        import pandas as pd
        data = {
            "full_text": [
                "game crash bug broken error",
                "lost 50/50 pity whale jades",
                "amazing love great perfect fun",
                "op overpowered nerf buff weak",
                "just playing today started",
            ],
            "post_type": [
                "bug_report", "gacha_frustration",
                "positive_feedback", "balance_complaint", "general_discussion",
            ],
        }
        df = pd.DataFrame(data)
        pipeline = self.ml.train_classifier(df)
        result = pipeline.predict(["the game keeps crashing with an error"])
        assert result[0] in [
            "bug_report", "balance_complaint", "gacha_frustration",
            "positive_feedback", "general_discussion",
        ]

    def test_train_regressor_returns_pipeline_and_scaler(self):
        import pandas as pd
        data = {
            "full_text": ["post one text", "post two text", "post three"],
            "upvotes": [100, 50, 300],
            "upvote_ratio": [0.9, 0.7, 0.95],
            "num_comments": [10, 5, 30],
            "post_length": [100, 50, 150],
            "has_version_tag": [1, 0, 1],
        }
        df = pd.DataFrame(data)
        pipeline, scaler = self.ml.train_regressor(df)
        assert "tfidf" in pipeline
        assert "regressor" in pipeline
        assert scaler is not None


# ===========================================================================
# 4. brain.py — embedding + semantic search
# ===========================================================================

class TestBrain:
    """Tests for embedding calls and semantic search."""

    def setup_method(self):
        self.mock_embedding = [0.01] * 1536

    @patch("brain.openai.OpenAI")
    def test_get_embedding_returns_vector(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_item = MagicMock()
        mock_item.embedding = self.mock_embedding
        mock_client.embeddings.create.return_value.data = [mock_item]

        import brain
        result = brain.get_embedding("test text")
        assert len(result) == 1536
        assert isinstance(result[0], float)

    @patch("brain.openai.OpenAI")
    def test_batch_embed_chunks_correctly(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        def make_response(texts):
            items = [MagicMock(embedding=self.mock_embedding) for _ in texts]
            resp = MagicMock()
            resp.data = items
            return resp

        mock_client.embeddings.create.side_effect = lambda model, input: make_response(input)

        import brain
        # 150 texts — fits in 1 batch (BATCH_SIZE=500)
        texts = ["text"] * 150
        results = brain.batch_embed(texts)
        assert len(results) == 150
        assert mock_client.embeddings.create.call_count == 1

    @patch("brain.db.execute_query")
    @patch("brain.get_embedding")
    def test_semantic_search_returns_results(self, mock_embed, mock_query):
        mock_embed.return_value = self.mock_embedding
        mock_query.return_value = [
            {**p, "similarity": p["similarity"]}
            for p in MOCK_POSTS
        ]
        import brain
        results = brain.semantic_search("Acheron bug", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert "similarity" in results[0]

    @patch("brain.openai.OpenAI")
    def test_get_embedding_retries_on_rate_limit(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        import openai as oai
        mock_item = MagicMock()
        mock_item.embedding = self.mock_embedding
        good_resp = MagicMock()
        good_resp.data = [mock_item]

        mock_client.embeddings.create.side_effect = [
            oai.RateLimitError("rate limit", response=MagicMock(), body={}),
            good_resp,
        ]

        import brain
        with patch("brain.time.sleep"):  # skip real sleep
            result = brain.get_embedding("test")
        assert len(result) == 1536
        assert mock_client.embeddings.create.call_count == 2


# ===========================================================================
# 5. rag_engine.py — context formatting + defect extraction + parsing
# ===========================================================================

class TestRagEngine:
    """Tests for RAG helper functions and the full query pipeline."""

    def setup_method(self):
        import rag_engine
        self.rag = rag_engine

    def test_format_context_includes_upvotes(self):
        ctx = self.rag.format_context(MOCK_POSTS[:2])
        assert "500" in ctx
        assert "342" in ctx

    def test_format_context_truncates_long_text(self):
        long_post = {**MOCK_POSTS[0], "full_text": "x" * 1000}
        ctx = self.rag.format_context([long_post])
        assert len(ctx) < 1200  # 400 char cap + metadata

    def test_extract_defects_bug(self):
        defects = self.rag.extract_defects("There is a major crash bug in this version.")
        assert "bug_report" in defects

    def test_extract_defects_gacha(self):
        defects = self.rag.extract_defects("Players frustrated by gacha and 50/50 losses.")
        assert "gacha_frustration" in defects

    def test_extract_defects_balance(self):
        defects = self.rag.extract_defects("Character is way overpowered, needs nerf.")
        assert "balance_complaint" in defects

    def test_extract_defects_multiple(self):
        defects = self.rag.extract_defects("Bug with gacha and balance is broken too.")
        assert len(defects) >= 2

    def test_extract_defects_none(self):
        defects = self.rag.extract_defects("The weather is nice today.")
        assert defects == []

    def test_parse_sections_both_present(self):
        raw = (
            "[ENGLISH SUMMARY]\n- Good content\n- More points\n"
            "[中文摘要]\n• 中文内容\n• 更多要点"
        )
        eng, zh = self.rag._parse_sections(raw)
        assert "Good content" in eng
        assert "中文内容" in zh

    def test_parse_sections_fallback_no_headers(self):
        raw = "Just some plain text with no sections."
        eng, zh = self.rag._parse_sections(raw)
        assert "plain text" in eng
        assert zh == ""

    @patch("rag_engine.semantic_search")
    @patch("rag_engine._call_gpt")
    def test_run_rag_query_returns_correct_keys(self, mock_gpt, mock_search):
        mock_search.return_value = MOCK_POSTS
        mock_gpt.return_value = (
            "[ENGLISH SUMMARY]\n- Post 2.3 has bugs\n- Players frustrated\n"
            "[中文摘要]\n• 2.3版本存在漏洞\n• 玩家感到沮丧"
        )
        result = self.rag.run_rag_query("What bugs exist in 2.3?")
        assert "english_summary" in result
        assert "chinese_summary" in result
        assert "defects_found" in result
        assert "source_posts" in result
        assert "retrieved_count" in result

    @patch("rag_engine.semantic_search")
    @patch("rag_engine._call_gpt")
    def test_run_rag_query_mandarin_input(self, mock_gpt, mock_search):
        mock_search.return_value = MOCK_POSTS
        mock_gpt.return_value = (
            "[ENGLISH SUMMARY]\n- Bug confirmed\n[中文摘要]\n• 确认漏洞"
        )
        result = self.rag.run_rag_query("黄泉角色强度评价")
        assert result["english_summary"] != ""

    @patch("rag_engine.semantic_search")
    @patch("rag_engine._call_gpt")
    def test_run_rag_query_trims_source_posts(self, mock_gpt, mock_search):
        mock_search.return_value = MOCK_POSTS * 5  # 15 candidates
        mock_gpt.return_value = "[ENGLISH SUMMARY]\n- ok\n[中文摘要]\n• ok"
        result = self.rag.run_rag_query("anything")
        assert len(result["source_posts"]) <= 10

    @patch("rag_engine._call_gpt", side_effect=RuntimeError("GPT failed"))
    @patch("rag_engine.semantic_search", return_value=MOCK_POSTS)
    def test_run_rag_query_raises_on_gpt_failure(self, _, mock_gpt):
        with pytest.raises(RuntimeError, match="GPT failed"):
            self.rag.run_rag_query("what happens when GPT fails?")


# ===========================================================================
# 6. api.py — FastAPI endpoint contracts
# ===========================================================================

class TestApi:
    """Tests for all FastAPI endpoints using TestClient with mocked services."""

    def setup_method(self):
        # Must patch before importing api to prevent real startup side-effects
        self.patches = [
            patch("ml_engine.load_models"),
            patch("ml_engine._classifier", MagicMock()),
            patch("ml_engine._regressor", MagicMock()),
            patch("db.health_check", return_value=True),
            patch("db.execute_query", return_value=[]),
        ]
        for p in self.patches:
            p.start()

        from fastapi.testclient import TestClient
        import api
        self.client = TestClient(api.app)

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    def test_health_returns_ok(self):
        with patch("api.db.health_check", return_value=True):
            resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["db"] in ("connected", "disconnected")
        assert "status" in data

    def test_sentiment_trend_default_params(self):
        with patch("api.db.execute_query", return_value=[
            {"date": "2026-03-10", "avg_sentiment": 0.5, "post_count": 42}
        ]):
            resp = self.client.get("/sentiment-trend")
        assert resp.status_code == 200
        assert "trend" in resp.json()

    def test_sentiment_trend_with_version_filter(self):
        with patch("api.db.execute_query", return_value=[]):
            resp = self.client.get("/sentiment-trend?days=14&version_tag=2.3")
        assert resp.status_code == 200

    def test_sentiment_trend_invalid_days(self):
        resp = self.client.get("/sentiment-trend?days=0")
        assert resp.status_code == 422

    def test_rag_query_valid_english(self):
        mock_result = {
            "english_summary": "Players frustrated by bugs.",
            "chinese_summary": "玩家对漏洞感到沮丧。",
            "defects_found": ["bug_report"],
            "source_posts": [],
            "retrieved_count": 5,
        }
        with patch("api.run_rag_query", return_value=mock_result):
            resp = self.client.post("/rag-query", json={"question": "What bugs exist?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "english_summary" in data
        assert "chinese_summary" in data

    def test_rag_query_valid_mandarin(self):
        mock_result = {
            "english_summary": "...",
            "chinese_summary": "...",
            "defects_found": [],
            "source_posts": [],
            "retrieved_count": 3,
        }
        with patch("api.run_rag_query", return_value=mock_result):
            resp = self.client.post("/rag-query", json={"question": "黄泉角色强度"})
        assert resp.status_code == 200

    def test_rag_query_empty_question(self):
        resp = self.client.post("/rag-query", json={"question": "   "})
        assert resp.status_code == 422

    def test_rag_query_missing_field(self):
        resp = self.client.post("/rag-query", json={})
        assert resp.status_code == 422

    def test_version_compare_returns_comparison(self):
        mock_rows = [
            {
                "version_tag": "2.1",
                "avg_sentiment": 0.71,
                "post_count": 203,
                "defect_rate": 0.23,
                "top_defect": "gacha_frustration",
            },
            {
                "version_tag": "2.2",
                "avg_sentiment": 0.65,
                "post_count": 178,
                "defect_rate": 0.31,
                "top_defect": "balance_complaint",
            },
        ]
        with patch("api.db.execute_query", return_value=mock_rows):
            resp = self.client.get("/version-compare?version_a=2.1&version_b=2.2")
        assert resp.status_code == 200
        data = resp.json()
        assert "comparison" in data
        assert len(data["comparison"]) == 2

    def test_version_compare_missing_params(self):
        resp = self.client.get("/version-compare?version_a=2.1")
        assert resp.status_code == 422

    def test_search_returns_results(self):
        with patch("api.semantic_search", return_value=MOCK_POSTS):
            resp = self.client.get("/search?q=Acheron+bug")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["query"] == "Acheron bug"

    def test_search_empty_query(self):
        resp = self.client.get("/search?q=")
        assert resp.status_code == 422

    def test_search_top_k_limit(self):
        resp = self.client.get("/search?q=test&top_k=200")
        assert resp.status_code == 422  # max is 50

    def test_versions_endpoint(self):
        with patch("api.db.execute_query", return_value=[
            {"version_tag": "2.1", "post_count": 100},
            {"version_tag": "2.2", "post_count": 200},
        ]):
            resp = self.client.get("/versions")
        assert resp.status_code == 200
        versions = resp.json()["versions"]
        assert len(versions) == 2
        assert versions[0]["version"] == "2.1"
        assert versions[0]["post_count"] == 100

    def test_post_type_distribution_endpoint(self):
        with patch("api.db.execute_query", return_value=[
            {"post_type": "bug_report", "count": 50},
            {"post_type": "positive_feedback", "count": 120},
        ]):
            resp = self.client.get("/post-type-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["distribution"]) == 2

    def test_ingest_endpoint_success(self):
        mock_stats = {
            "status": "success",
            "posts_scraped": 487,
            "posts_embedded": 487,
            "posts_classified": 487,
            "duration_seconds": 142.3,
        }
        with patch("api.run_ingestion", return_value=mock_stats):
            resp = self.client.post("/ingest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["posts_scraped"] == 487

    def test_ingest_endpoint_failure(self):
        mock_stats = {
            "status": "failed",
            "posts_scraped": 0,
            "posts_embedded": 0,
            "posts_classified": 0,
            "error_msg": "Reddit unreachable",
            "duration_seconds": 0,
        }
        with patch("api.run_ingestion", return_value=mock_stats):
            resp = self.client.post("/ingest")
        assert resp.status_code == 500
