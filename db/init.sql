-- RailSignal Database Initialization
-- Run once after installing PostgreSQL + pgvector:
--   psql -U <your_user> -d <your_db> -f db/init.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main posts table
CREATE TABLE IF NOT EXISTS reddit_posts (
    id              SERIAL PRIMARY KEY,
    post_id         VARCHAR(20) UNIQUE NOT NULL,
    title           TEXT,
    body            TEXT,
    full_text       TEXT NOT NULL,
    author          VARCHAR(100),
    upvotes         INTEGER DEFAULT 0,
    upvote_ratio    FLOAT DEFAULT 0.0,
    num_comments    INTEGER DEFAULT 0,
    created_utc     TIMESTAMP NOT NULL,
    scraped_at      TIMESTAMP DEFAULT NOW(),
    version_tag     VARCHAR(20),
    post_type       VARCHAR(30),
    sentiment_score FLOAT,
    embedding       VECTOR(1536)
);

-- Ingestion audit log
CREATE TABLE IF NOT EXISTS ingestion_log (
    id                  SERIAL PRIMARY KEY,
    run_at              TIMESTAMP DEFAULT NOW(),
    posts_scraped       INTEGER,
    posts_embedded      INTEGER,
    posts_classified    INTEGER,
    status              VARCHAR(20),
    error_msg           TEXT
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_created_utc  ON reddit_posts(created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_version_tag  ON reddit_posts(version_tag);
CREATE INDEX IF NOT EXISTS idx_post_type    ON reddit_posts(post_type);
CREATE INDEX IF NOT EXISTS idx_embedding    ON reddit_posts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
