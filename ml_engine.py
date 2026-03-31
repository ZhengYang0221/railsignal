"""
ml_engine.py — Service 2: sklearn post-type classifier and engagement score regressor.

Models are trained on ingested Reddit data and persisted as .pkl files.
Called by ingestor.py (training + inference) and api.py (inference only).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import db

logger = logging.getLogger("railsignal.ml_engine")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
REGRESSOR_PATH = MODELS_DIR / "regressor.pkl"
LABELS_CSV = Path("data/training_labels.csv")

MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Global model state (loaded once at API startup / ingestion start)
# ---------------------------------------------------------------------------
_classifier: Optional[Pipeline] = None
_regressor: Optional[Pipeline] = None
_reg_scaler: Optional[MinMaxScaler] = None

# ---------------------------------------------------------------------------
# Keyword-based seed labelling heuristics
# ---------------------------------------------------------------------------
KEYWORD_RULES: dict[str, list[str]] = {
    "bug_report": [
        "bug", "crash", "broken", "glitch", "error", "freeze",
        "not working", "fix", "issue", "problem",
    ],
    "balance_complaint": [
        "op", "overpowered", "broken", "nerf", "buff", "weak",
        "useless", "too strong", "unfair", "unbalanced",
    ],
    "gacha_frustration": [
        "pity", "lost 50/50", "no pulls", "jades", "f2p",
        "whale", "p2w", "expensive", "predatory", "rigged",
    ],
    "positive_feedback": [
        "amazing", "love", "great", "best", "perfect",
        "beautiful", "fun", "enjoy", "good", "nice",
    ],
}


def seed_label(text: str) -> str:
    """Assign a heuristic label based on keyword matches."""
    text_lower = text.lower()
    scores = {label: 0 for label in KEYWORD_RULES}
    for label, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1
    top = max(scores, key=scores.get)  # type: ignore[arg-type]
    return top if scores[top] > 0 else "general_discussion"


# ---------------------------------------------------------------------------
# Training data helpers
# ---------------------------------------------------------------------------

def _load_training_data() -> pd.DataFrame:
    """
    Load labeled training data from DB. If a CSV of curated labels exists,
    merge it on top for supervised examples.
    """
    rows = db.execute_query(
        "SELECT full_text, upvotes, upvote_ratio, num_comments, version_tag "
        "FROM reddit_posts WHERE full_text IS NOT NULL LIMIT 5000"
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["has_version_tag"] = df["version_tag"].notna().astype(int)
    df["post_length"] = df["full_text"].str.len()

    # Seed labels
    df["post_type"] = df["full_text"].apply(seed_label)

    # Override with curated labels if available
    if LABELS_CSV.exists():
        curated = pd.read_csv(LABELS_CSV)
        if "full_text" in curated.columns and "post_type" in curated.columns:
            merge = df.merge(
                curated[["full_text", "post_type"]],
                on="full_text",
                how="left",
                suffixes=("_seed", "_curated"),
            )
            df["post_type"] = merge["post_type_curated"].fillna(merge["post_type_seed"])

    return df


# ---------------------------------------------------------------------------
# Model A — Post Type Classifier
# ---------------------------------------------------------------------------

def train_classifier(df: pd.DataFrame) -> Pipeline:
    """Train a TF-IDF + Logistic Regression classifier on post_type labels."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            df["full_text"], df["post_type"], test_size=0.2, random_state=42, stratify=df["post_type"]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info("Classifier trained on %d samples (80/20 split).\n%s", len(df), report)
    else:
        pipeline.fit(df["full_text"], df["post_type"])
        logger.info("Classifier trained on %d samples (too few for split).", len(df))

    return pipeline


# ---------------------------------------------------------------------------
# Model B — Engagement Score Regressor
# ---------------------------------------------------------------------------

def train_regressor(df: pd.DataFrame) -> tuple[Pipeline, MinMaxScaler]:
    """
    Train a Random Forest Regressor predicting log-normalized upvotes.
    Returns the fitted pipeline and a MinMaxScaler for output normalization.
    """
    df = df.copy()
    df["log_upvotes"] = np.log1p(df["upvotes"].clip(lower=0))

    tfidf = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
    text_features = tfidf.fit_transform(df["full_text"]).toarray()

    meta_features = df[["num_comments", "upvote_ratio", "post_length", "has_version_tag"]].fillna(0).values
    X = np.hstack([text_features, meta_features])
    y = df["log_upvotes"].values

    regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    regressor.fit(X, y)

    # Build a MinMaxScaler on predictions to normalise output to [-1, 1]
    preds = regressor.predict(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(preds.reshape(-1, 1))

    # Wrap tfidf + regressor in a simple namespace for pickling convenience
    pipeline = {"tfidf": tfidf, "regressor": regressor}

    logger.info("Regressor trained on %d samples.", len(df))
    return pipeline, scaler


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def classify_post(text: str) -> tuple[str, float]:
    """
    Predict the post_type label and confidence for a given text.
    Falls back to heuristic seed_label if model not loaded.
    """
    global _classifier
    if _classifier is None:
        label = seed_label(text)
        return label, 0.5

    proba = _classifier.predict_proba([text])[0]
    label_idx = int(np.argmax(proba))
    label = _classifier.classes_[label_idx]
    confidence = float(proba[label_idx])
    return label, confidence


def score_engagement(features: dict) -> float:
    """
    Predict a normalized engagement score in [-1.0, 1.0] for a post.
    features dict must contain: full_text, num_comments, upvote_ratio,
    post_length, has_version_tag.
    Falls back to 0.0 if model not loaded.
    """
    global _regressor, _reg_scaler
    if _regressor is None or _reg_scaler is None:
        return 0.0

    tfidf = _regressor["tfidf"]
    regressor = _regressor["regressor"]

    text_feat = tfidf.transform([features["full_text"]]).toarray()
    meta_feat = np.array(
        [[
            features.get("num_comments", 0),
            features.get("upvote_ratio", 0.0),
            features.get("post_length", 0),
            features.get("has_version_tag", 0),
        ]]
    )
    X = np.hstack([text_feat, meta_feat])
    raw_pred = regressor.predict(X)
    normalized = _reg_scaler.transform(raw_pred.reshape(-1, 1))[0][0]
    return float(normalized)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(path: str = str(MODELS_DIR)) -> None:
    """Pickle the trained classifier and regressor to disk."""
    global _classifier, _regressor, _reg_scaler
    models_path = Path(path)
    with open(models_path / "classifier.pkl", "wb") as f:
        pickle.dump(_classifier, f)
    with open(models_path / "regressor.pkl", "wb") as f:
        pickle.dump({"pipeline": _regressor, "scaler": _reg_scaler}, f)
    logger.info("Models saved to %s", path)


def load_models(path: str = str(MODELS_DIR)) -> None:
    """
    Load persisted models from disk. If none exist, train from DB data.
    """
    global _classifier, _regressor, _reg_scaler
    clf_path = Path(path) / "classifier.pkl"
    reg_path = Path(path) / "regressor.pkl"

    if clf_path.exists() and reg_path.exists():
        with open(clf_path, "rb") as f:
            _classifier = pickle.load(f)
        with open(reg_path, "rb") as f:
            payload = pickle.load(f)
            _regressor = payload["pipeline"]
            _reg_scaler = payload["scaler"]
        logger.info("Models loaded from %s", path)
        return

    logger.info("No persisted models found — training from DB data ...")
    df = _load_training_data()
    if df.empty:
        logger.warning("No training data in DB — models will use heuristics only.")
        return

    _classifier = train_classifier(df)
    _regressor, _reg_scaler = train_regressor(df)
    save_models(path)
