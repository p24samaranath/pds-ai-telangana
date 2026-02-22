# 🌾 PDS AI Optimization System
## AI-Powered Fair Price Shop (FPS) Optimization & Fraud Detection

A production-grade **multi-agent AI system** for India's Public Distribution System (PDS),
built for Telangana. Simultaneously tackles supply-demand mismatches, fraudulent transactions,
and poor geographic accessibility.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                 ORCHESTRATOR AGENT                   │
│          (Claude LLM + LangGraph-style routing)      │
└───────┬──────────────┬──────────────┬────────────────┘
        │              │              │              │
   ┌────▼────┐   ┌─────▼─────┐ ┌─────▼─────┐ ┌────▼────┐
   │ DEMAND  │   │  FRAUD    │ │   GEO     │ │REPORTING│
   │FORECAST │   │DETECTION  │ │OPTIMIZER  │ │  AGENT  │
   │ AGENT   │   │  AGENT    │ │  AGENT    │ │         │
   └─────────┘   └───────────┘ └───────────┘ └─────────┘
   LSTM+Prophet  DBSCAN+IF    K-Means+Voronoi  Claude NL
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)
- Anthropic API key (get from [console.anthropic.com](https://console.anthropic.com))

### 1. Clone & Setup
```bash
git clone <repo>
cd pds-optimization
./setup.sh
```

### 2. Configure API Key
```bash
# Edit .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Start Backend
```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Start Frontend
```bash
cd frontend
npm run dev
# Opens at http://localhost:3000
```

### 5. Access
| Service     | URL                              |
|-------------|----------------------------------|
| Dashboard   | http://localhost:3000            |
| API Docs    | http://localhost:8000/docs       |
| ReDoc       | http://localhost:8000/redoc      |

---

## 🤖 Agents

### Orchestrator Agent
Central coordinator built on Claude (claude-sonnet-4-6). Routes tasks to sub-agents,
maintains shared memory of decisions, generates executive summaries.

**Workflows:**
- `MONTHLY_BATCH` — Full pipeline: fraud → forecast → geo → report
- `REALTIME_FRAUD` — Stream mode fraud check with critical alert escalation
- `NL_QUERY` — Natural language Q&A for officials
- `GEO_CHANGE` — Re-analysis triggered by shop network changes

### Demand Forecast Agent
**Models:** LSTM (PyTorch) + Facebook Prophet + Ensemble
- 90-day demand lookahead per FPS shop per commodity
- Auto-retrains when MAPE > 15%
- Seasonal encoding (festivals, harvest cycles)

### Fraud Detection Agent
**Models:** Rule Engine + Isolation Forest + DBSCAN + Graph Fraud Ring Detector
- **Rule Engine:** Duplicate cards, after-hours transactions, month-end bulk fraud, low biometric rate
- **Isolation Forest:** Point anomalies across 7 transaction features
- **DBSCAN:** Coordinated fraud cluster detection
- **Graph Fraud Ring Detector:** Bipartite card-shop transaction graph + community detection (NetworkX) to surface organised fraud rings; scores rings by multi-shop card ratio, biometric miss rate, graph density, and PageRank hub centrality

### Geospatial Optimizer Agent
**Models:** K-Means + Voronoi Tessellation
- Identifies beneficiaries > 5 km from nearest active FPS
- Recommends new FPS locations using centroid analysis
- Computes district accessibility scores (0-1)
- Flags underperforming shops for consolidation

### Reporting Agent
- Dashboard metrics aggregation
- AI-generated executive summaries (Claude)
- **RAG chatbot** — TF-IDF vector store over indexed agent outputs; retrieves top-6 relevant chunks per query before calling Claude
- **Multi-turn conversation** — per-session history (up to 10 turns); session management via `/api/v1/agents/chat`
- Role-based views for different stakeholders

---

## 📁 Project Structure

```
.
├── backend/
│   ├── app/               # FastAPI app, config, constants
│   ├── agents/            # 5 autonomous agents
│   ├── ml_models/         # LSTM, Prophet, Isolation Forest, DBSCAN, Geo
│   ├── database/          # SQLAlchemy models, Pydantic schemas
│   ├── routes/            # REST API endpoints
│   ├── services/          # Data ingestion, feature engineering
│   └── data/              # Raw, processed, model files
├── frontend/
│   └── src/
│       ├── pages/         # Dashboard, Fraud, Forecasts, Map, AI Query
│       ├── components/    # Reusable UI components
│       └── services/      # API client
├── docker/                # Docker + nginx configs
└── setup.sh               # One-command setup
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/agents/status` | Orchestrator status |
| POST | `/api/v1/agents/run/full` | Run full pipeline |
| POST | `/api/v1/agents/query` | NL query |
| POST | `/api/v1/forecasts/` | Generate demand forecasts |
| GET | `/api/v1/fraud/alerts` | Get fraud alerts |
| GET | `/api/v1/fraud/alerts/critical` | Critical alerts only |
| POST | `/api/v1/fraud/score-transaction` | Score single transaction |
| GET | `/api/v1/geo/analysis` | Full geospatial analysis |
| GET | `/api/v1/geo/shops` | FPS shops as GeoJSON |
| GET | `/api/v1/geo/underserved-zones` | Underserved beneficiary zones |
| GET | `/api/v1/geo/recommendations` | New FPS location recommendations |
| GET | `/api/v1/dashboard/metrics` | Dashboard KPIs |
| POST | `/api/v1/agents/chat` | Multi-turn RAG chatbot |
| GET | `/api/v1/agents/chat/sessions` | List active chat sessions |
| DELETE | `/api/v1/agents/chat/sessions/{id}` | Clear a session |
| GET | `/api/v1/agents/chat/rag/stats` | RAG store statistics |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Agent Framework | Custom orchestrator (LangGraph-compatible) |
| LLM | Claude (claude-sonnet-4-6) via Anthropic SDK |
| ML Models | PyTorch (LSTM), Prophet, scikit-learn |
| Geospatial | SciPy, NumPy, GeoPandas |
| API | FastAPI + Pydantic v2 |
| Database | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy |
| Frontend | React 18 + Leaflet.js + Chart.js |
| Caching | Redis |
| Containerisation | Docker + nginx |

---

## 📊 Key Metrics

- **Fraud Detection:** Precision, Recall, F1-score; Mean Time to Detect (< 1s)
- **Demand Forecasting:** MAPE per commodity per shop; auto-retrain at MAPE > 15%
- **Geospatial:** % beneficiaries within 3 km; district accessibility score (0-1)
- **System:** API latency; pipeline duration; uptime

---

## 🔐 Responsible AI

- **Privacy:** Aadhaar data tokenised; compliant with DPDP Act 2023
- **Explainability:** Every fraud alert includes human-readable explanation + anomaly score
- **Human-in-the-Loop:** Critical actions (dealer suspension, card blocking) require officer confirmation
- **Fairness:** Fraud models designed to minimise false positives for tribal/rural beneficiaries
- **Auditability:** Every agent decision logged with timestamps for regulatory audit

---

## 📈 Phased Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 — Foundation | Data pipeline, baseline models, basic dashboard | ✅ Complete |
| 2 — Core Agents | DBSCAN + IF fraud, geospatial optimizer, LangGraph | ✅ Complete |
| 3 — Advanced | Graph fraud rings (NetworkX), RAG chatbot, auto-retraining | ✅ Complete |
| 4 — Scale | Multi-state expansion, fairness audits, DPDP compliance | 📋 Planned |

**Phase 3 progress:**
- ✅ Graph Fraud Ring Detector — bipartite card-shop graph + community detection + PageRank scoring (`ml_models/fraud_detection/graph_fraud_detector.py`)
- ✅ Auto-retraining — MAPE-triggered LSTM/Prophet retraining in `DemandForecastAgent`
- ✅ RAG chatbot — TF-IDF in-memory vector store (`services/rag_store.py`), top-k retrieval grounding every Claude call, multi-turn session history (`agents/reporting_agent.py`)

---

## 📄 License

MIT License — Built for public welfare. Free to use, modify, and deploy.
