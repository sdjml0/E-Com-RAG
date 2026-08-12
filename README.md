# 🏗️ Production-Grade Multimodal E-Commerce RAG Microservice

A high-performance, cost-effective, easy-to-maintain **Multimodal E-Commerce RAG Microservice** built with Python 3.14+, FastAPI, Pydantic v2, and Qdrant Named Multi-Vectors.

Designed strictly according to the architecture specification in [`RAG.md`](./RAG.md) and integration requirements in [`ECOM_AGENT_INTEGRATION.md`](./ECOM_AGENT_INTEGRATION.md).

---

## ⚡ Key Technical Features

1. **Named Multi-Vector Schema**: Single point documents containing dual vectors (`text_vector` 384d/1536d + `image_vector` 512d/768d) and metadata payloads.
2. **Single-Stage Payload Filtering**: Hard filters on `brand` (keyword match), `category_path` (taxonomy tree match), and `price` (float range) executed directly inside Qdrant HNSW index traversal.
3. **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists from text, visual, and keyword retrievers ($k=60.0$).
4. **Price Elasticity Soft Penalty**: Dynamically dampens product scores based on price deviation from `target_price`:
   $$\text{Final\_Score}(d) = RRF\_Score(d) \times \max\left(0.5, 1.0 - 0.3 \times \frac{|\text{Price}_d - T|}{T}\right)$$
5. **Real-Time Telemetry Event Stream (SSE)**: Publishes live microservice health, moving data logs, pipeline stage latencies, and error events to connected client dashboards via `/api/v1/events/stream`.
6. **Sub-Second Streaming RAG Response**: Multimodal recommendation synthesis via Gemini 2.0 Flash (`google-genai`) with fallback template generation.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Microservice
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/healthz/readiness`
- Telemetry SSE Stream: `http://localhost:8000/api/v1/events/stream`

### 3. Seed Sample E-Commerce Catalog
```bash
python scripts/seed_demo_data.py
```

### 4. Run Latency & SLA Evaluation Benchmark
```bash
python scripts/eval_pipeline.py
```

### 5. Run Unit & Integration Test Suite
```bash
pytest -v tests/
```

---

## 📂 Project Structure

```
/Users/saad/Desktop/RAG/
├── RAG.md                          # Architecture specification
├── ECOM_AGENT_INTEGRATION.md       # Integration guide for E-COM agents
├── pyproject.toml                  # Packaging & dependency configuration
├── requirements.txt                # Requirements file
├── README.md                       # Project manual
├── app/
│   ├── config.py                   # Pydantic Settings (Qdrant, Embedders, LLM)
│   ├── schemas.py                  # Pydantic v2 data models for API & telemetry
│   ├── telemetry/
│   │   └── event_bus.py            # Pub/Sub SSE event bus for telemetry
│   ├── embeddings/
│   │   ├── base.py                 # Abstract base class for embedders
│   │   ├── text_embedder.py        # SentenceTransformers & fast text embedder
│   │   └── vision_embedder.py      # PIL & visual feature embedder
│   ├── db/
│   │   └── vector_db.py            # Qdrant client wrapper & named multi-vectors
│   ├── search/
│   │   ├── rrf.py                  # RRF & price soft elasticity penalty engine
│   │   └── hybrid_searcher.py      # Multi-vector search with Qdrant payload filters
│   ├── llm/
│   │   └── rag_generator.py        # Multimodal Gemini LLM & streaming engine
│   ├── cache/
│   │   └── cache_service.py        # LRU & query context cache
│   └── main.py                     # FastAPI application entrypoint
├── scripts/
│   ├── seed_demo_data.py           # Seed script for e-commerce catalog
│   └── eval_pipeline.py            # Latency SLA evaluation script
└── tests/
    ├── test_schemas.py             # Schema unit tests
    ├── test_search.py              # Search & RRF scoring tests
    └── test_api.py                 # Endpoint integration tests
```

---

## 🔌 E-COM Agent API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/healthz/liveness` | `GET` | K8s liveness probe. |
| `/healthz/readiness` | `GET` | K8s readiness probe & vector DB status. |
| `/api/v1/health` | `GET` | Detailed telemetry, vector count, P95 latency. |
| `/api/v1/products/ingest` | `POST` | Ingest/upsert single product into multi-vector index. |
| `/api/v1/products/batch-ingest` | `POST` | Ingest batch of products. |
| `/api/v1/search` | `POST` | Multi-vector search with payload pre-filters and RRF. |
| `/api/v1/rag/generate` | `POST` | Multimodal RAG recommendation synthesis. |
| `/api/v1/rag/stream` | `POST` | SSE token streaming for sub-second LLM responses. |
| `/api/v1/events/stream` | `GET` | SSE pipeline telemetry broadcast (`health_update`, `moving_data`, `error_event`). |
