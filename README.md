# RailSignal

**Multilingual Gaming Intelligence & Yield Platform**

[![CI](https://github.com/ZhengYang0221/railsignal/actions/workflows/test.yml/badge.svg)](https://github.com/ZhengYang0221/railsignal/actions/workflows/test.yml)

Ingests English Reddit posts from r/HonkaiStarRail and surfaces bilingual (EN/ZH)
actionable insights for game live-ops stakeholders via a Streamlit dashboard.

---

## Walkthrough Screenshots


><img width="1789" height="799" alt="image" src="https://github.com/user-attachments/assets/0c876cfe-de46-4e6c-b9a9-2ddd6ff7da86" />
> <img width="1833" height="722" alt="image" src="https://github.com/user-attachments/assets/3dcb3de5-2058-4f74-8655-1b45008faba6" />
The Community Health tab surfaces real-time KPIs pulled from r/HonkaiStarRail. Below the metrics, an AI Weekly Digest powered by GPT-5.4 synthesises the week's top community signals
into a side-by-side bilingual brief — English on the left for global stakeholders, Mandarin on the right for the devs — with referenced source posts linked below.
<br>
<br>

The lower half shows a dual-axis chart overlaying daily sentiment (line) against post volume (bars), an ML-classified post type breakdown (TF-IDF + Logistic Regression),
and a Top Post per Complaint Category panel with direct Reddit links for raw community signal.
<br>
<br>

> <img width="1842" height="786" alt="image" src="https://github.com/user-attachments/assets/c9f503a2-6c5e-4d35-b9ef-c94ac52d87f3" />
The Bridge implements semantic search over the full Reddit corpus.
The search triggers an OpenAI embedding query against pgvector, retrieves the relevant posts, and pipes them through a
RAG chain to generate a dual-language executive summary.
Both languages' summaries are produced in a single call, ensuring consistency. Detected sentiment
defects such as Bug Report are automatically tagged below, and source posts used to ground the answer are accessible at the bottom.

<br>
<br>

> <img width="1834" height="837" alt="image" src="https://github.com/user-attachments/assets/51fdcc85-4e0a-44f0-b561-d510ced89661" />
<img width="1813" height="375" alt="image" src="https://github.com/user-attachments/assets/2b301197-6c4d-47bb-9205-d82c49fbb18f" />

The Version Compare tab allows live-ops teams to benchmark community health across
patch cycles. Comparing versions reveals
meaningful data such as average sentiment, defect rate, and post volume across different patches.
The delta indicators and grouped bar charts make the version
health story scannable at a glance, without requiring stakeholders to query raw data. The bottom part includes an AI analysis of the comparison too.

<br>
<br>

> <img width="1828" height="893" alt="image" src="https://github.com/user-attachments/assets/6627e7c2-cd06-48ee-8e3e-e833e5b82711" />
A stakeholder-facing RAG chatbot grounded entirely in scraped Reddit data, every claim traceable to a source post. Suggested question chips lower the barrier for non-technical users.
Returns structured bilingual answers citing specific upvote counts (6,065 ↑, 4,657 ↑, 2,455 ↑), with flagged issues and direct Reddit source links for full auditability.






---

## Stack

| Layer | Technology |
|---|---|
| Data ingestion | Reddit public JSON API + `requests` |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector DB | PostgreSQL 15 + `pgvector` |
| Sentiment | VADER (`vaderSentiment`) — compound score in [-1, 1] |
| ML | `scikit-learn` — TF-IDF + Logistic Regression (post classifier) |
| RAG | Retrieval-augmented `gpt-5.4` synthesis (bilingual EN/ZH) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |

---

## Quick Start (Docker)

The easiest way to run the full stack:

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and POSTGRES_PASSWORD in .env

docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

Then trigger an initial data ingest from the dashboard (Community Health tab → **🔄 Refresh**).

---

## Manual Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd railsignal
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY and Postgres credentials
```

### 3. Create the database

```bash
createdb railsignal
psql -U <your_user> -d railsignal -f db/init.sql
```

### 4. Run each service in a separate terminal

```bash
# Terminal 1 — seed data (run once, then on-demand via dashboard)
# ML models are trained automatically after the first ingest if no .pkl files exist.
python ingestor.py

# Terminal 2 — start the API
uvicorn api:app --reload --port 8000

# Terminal 3 — start the dashboard
streamlit run app.py
```

Dashboard: http://localhost:8501
API docs:  http://localhost:8000/docs

---

## Architecture

```
Reddit JSON API
      ↓
ingestor.py  ──→  reddit_posts (PostgreSQL + pgvector)
                      ↑
ml_engine.py ─────────┤  (classify post type)
brain.py     ─────────┤  (embed + vector search)
rag_engine.py ────────┤  (gpt-5.4 RAG synthesis)
                      ↓
api.py (FastAPI :8000)
      ↓
app.py (Streamlit :8501)
```

---

## Dashboard Features

| Tab | Feature |
|---|---|
| Community Health | 7-day sentiment trend, KPI cards, post type distribution, AI weekly digest |
| The Bridge | Cross-lingual semantic search (EN ↔ ZH) + bilingual RAG summary |
| Version Compare | Patch A vs Patch B — avg sentiment, post volume, defect rate |
| Ask the Analyst | Direct Q&A mode — stakeholder-ready bilingual answers |

---

## File Structure

```
railsignal/
├── ingestor.py          # Service 1: scraper + pipeline
├── ml_engine.py         # Service 2: post-type classifier
├── brain.py             # Service 3: embeddings + vector search
├── rag_engine.py        # Service 4: RAG chain (gpt-5.4)
├── api.py               # Service 5: FastAPI backend
├── app.py               # Service 6: Streamlit dashboard
├── db.py                # Shared DB connection pool
├── patch_schedule.py    # HSR patch date lookup
├── db/init.sql          # DB schema + pgvector indexes (run once)
├── models/              # Trained .pkl files (auto-generated, git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Known Limitations

| Area | Limitation | Future Fix |
|---|---|---|
| Post classifier | Labels are keyword-seeded pseudo-labels, not human-verified. The model achieves ~100% on holdout because it re-learns the same keyword rules it was trained on — not real generalisation. | Manually label ~500 posts in `data/training_labels.csv` and retrain, or use GPT-assisted labelling. |
| Class imbalance | `bug_report` (6 posts) and `gacha_frustration` (3 posts) are severely under-represented — the model effectively ignores these classes. | Collect more data via targeted subreddit searches; apply SMOTE or adjust class weights. |
| Sentiment as engagement | `sentiment_score` uses VADER compound on post text. VADER is designed for social media text and works well here, but it cannot capture sarcasm or gaming-community slang reliably. | Fine-tune a domain-specific sentiment model on HSR community data. |
| Reddit API scope | Hot + new feeds cover only the last 1–2 weeks. History mode extends coverage but is capped by Reddit's search API limits. | Integrate Pushshift or a Reddit dump for deeper historical data. |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB + model status check |
| POST | `/ingest` | Trigger standard Reddit scraping run |
| POST | `/ingest-history` | Background historical scrape (top/year + version searches) |
| GET | `/ingest-status/{job_id}` | Poll background job status |
| GET | `/sentiment-trend` | Daily sentiment over last N days |
| POST | `/rag-query` | Bilingual Q&A — EN or ZH input, EN+ZH output |
| GET | `/version-compare` | Patch satisfaction comparison by version tag |
| GET | `/date-range-compare` | Satisfaction comparison by custom date ranges |
| GET | `/search` | Semantic post search (cross-lingual) |
| GET | `/versions` | Available version tags with post counts |
| GET | `/patch-schedule` | HSR patch calendar |
| GET | `/top-posts-by-type` | Highest-upvoted post per complaint category |
| GET | `/post-type-distribution` | Post type breakdown over last N days |
