# 🏗️ Production Multimodal E-Commerce RAG Microservice

A high-performance, cost-effective, easy-to-maintain **Multimodal E-Commerce RAG Microservice** built with Python 3.14+, FastAPI, Pydantic v2, Qdrant Named Multi-Vectors, and Next-Gen Generative AI Vision models.

Designed strictly according to the architecture specification in [`RAG.md`](./RAG.md), pipeline flow documentation in [`response.md`](./response.md), and integration specifications in [`ECOM_AGENT_INTEGRATION.md`](./ECOM_AGENT_INTEGRATION.md) & [`json.md`](./json.md).

---

## ⚡ Key Technical Features

1. **Streamlined 2 Core APIs**:
   - **`GET /health`**: Health diagnostic & vector DB readiness probe.
   - **`POST /api/v1/rag/recommend`** (or `POST /`): Primary RAG recommendation API accepting query + 5 product parameters.
2. **Single Clean JSON Output**:
   - **`product_details`**: SKU, title, brand, category, price, match score, and reasoning.
   - **`image_generation_prompt`**: Prompt & style modifiers for Next-Gen AI image models.
   - **`generated_image_url`**: Direct AI image URL produced by Google Gemini 3.1 Flash Image (`models/gemini-3.1-flash-lite-image`) with 100% Free Flux / Stable Diffusion AI fallback.
3. **Named Multi-Vector Schema (Qdrant)**: Single point document storing dual vectors (`text_vector` 384d + `image_vector` 512d) and payload indexes (`brand`, `category_path`, `price`).
4. **Single-Stage Payload Filtering**: Hard filters on `brand` (keyword match), `category_path` (taxonomy tree match), and `price` (float range) executed directly inside Qdrant HNSW index traversal in $< 1.5\text{ms}$.
5. **Reciprocal Rank Fusion (RRF) & Price Elasticity Penalty**: Merges dense text rank and dense visual rank with an exponential budget dampener:
   $$\text{Final\_Score}(d) = RRF\_Score(d) \times \max\left(0.5, 1.0 - 0.3 \times \frac{|\text{Price}_d - T|}{T}\right)$$
6. **Sub-6ms SLA Performance**: Empirical P95 retrieval latency of **5.77ms** (target SLA $< 120\text{ms}$).
7. **One-Click Cloud Deployment**: Configured with `render.yaml` for instant Render deployment.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sdjml0/E-Com-RAG.git
cd E-Com-RAG

# Create & activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```env
ENVIRONMENT=development
QDRANT_URL=:memory:
COLLECTION_NAME=ecommerce_products_v1
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Microservice
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- RAG Recommendation: `http://localhost:8000/api/v1/rag/recommend`

### 4. Run SLA & Latency Evaluation Benchmark
```bash
python scripts/eval_pipeline.py
```
*Output: `Mean Latency: 3.55ms | P95 Latency: 5.77ms | SLA Requirement (< 120ms): PASSED 🟢`*

### 5. Run Automated Test Suite
```bash
pytest -v tests/
```
*Output: `6 passed in 7.58s 🟢`*

---

## 🔌 Core API Specifications

### 1. Health Probe (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health"
```
```json
{
  "status": "healthy",
  "vector_db_status": "green",
  "total_vectors_indexed": 12,
  "p95_latency_ms": 2.60,
  "active_subscribers": 0
}
```

### 2. RAG Recommendation (`POST /api/v1/rag/recommend` or `POST /`)
```bash
curl -X POST "http://localhost:8000/api/v1/rag/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Sleek black wireless noise-canceling headphones for travel under $400",
       "prod_title": "Sony WH-1000XM5",
       "prod_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
       "price": 398.00,
       "category": "Electronics > Audio > Headphones",
       "brand": "Sony"
     }'
```
```json
{
  "query": "Sleek black wireless noise-canceling headphones for travel under $400",
  "product_details": {
    "product_id": "SKU-38064",
    "title": "Sony WH-1000XM5",
    "brand": "sony",
    "category": "Electronics > Audio > Headphones",
    "price": 398.0,
    "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
    "match_score": 0.013115,
    "reasoning": "Top-ranked candidate for 'Sleek black wireless noise-canceling headphones for travel under $400'. Features premium Sony design in Electronics > Audio > Headphones priced at $398.00."
  },
  "image_generation_prompt": {
    "prompt": "Studio product photography of Sony WH-1000XM5 by Sony, crafted for 'Sleek black wireless noise-canceling headphones for travel under $400'. Rendered in clean commercial aesthetic, high-detail texture, soft studio lighting, neutral minimalist background, 8k resolution, photorealistic.",
    "action": "generate_or_edit",
    "base_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
    "style_modifiers": [
      "Sony signature style",
      "soft studio illumination",
      "clean minimalist backdrop",
      "8k ultra-detailed texture"
    ],
    "aspect_ratio": "1:1"
  },
  "generated_image_url": "https://image.pollinations.ai/prompt/Studio%20product%20photography%20of%20Sony%20WH-1000XM5%20by%20Sony%2C%20crafted%20for%20%27Sleek%20black%20wireless%20noise-canceling%20headphones%20for%20travel%20under%20%24400%27.%20Rendered%20in%20clean%20commercial%20aesthetic%2C%20high-detail%20texture%2C?width=512&height=512&nologo=true"
}
```

---

## 📂 Project Architecture

```
.
├── RAG.md                          # Production architecture specification
├── response.md                     # Pipeline flow & dual RAG documentation
├── ECOM_AGENT_INTEGRATION.md       # Integration guide for E-COM agents
├── json.md                         # Complete API JSON payload specs
├── render.yaml                     # One-click Render cloud deployment config
├── pyproject.toml                  # Packaging & dependencies
├── requirements.txt                # Pip requirements file
├── .env.example                    # Environment template
├── README.md                       # Service documentation
├── app/
│   ├── config.py                   # Pydantic Settings
│   ├── schemas.py                  # Pydantic V2 schemas
│   ├── db/
│   │   └── vector_db.py            # Qdrant Named Multi-Vector DB Manager
│   ├── embeddings/
│   │   ├── base.py                 # Abstract base class for embedders
│   │   ├── text_embedder.py        # Dense text embedder (384d)
│   │   └── vision_embedder.py      # Dense vision feature embedder (512d)
│   ├── search/
│   │   ├── rrf.py                  # Reciprocal Rank Fusion & price penalty engine
│   │   └── hybrid_searcher.py      # Multi-vector hybrid searcher
│   ├── llm/
│   │   ├── rag_generator.py        # Gemini Multimodal RAG generator
│   │   └── image_generator.py      # Gemini 3.1 Flash Image & Flux AI generator
│   └── main.py                     # FastAPI application entrypoint
├── scripts/
│   ├── seed_demo_data.py           # Product catalog seeding tool
│   └── eval_pipeline.py            # SLA latency evaluation script
└── tests/
    └── test_api.py                 # Integration & unit test suite
```

---

## 🌐 Deploying to Render

This repository includes a `render.yaml` blueprint:

1. Connect your repository at **[Render Dashboard](https://dashboard.render.com/)**.
2. Create a new **Web Service**.
3. Render automatically picks up `render.yaml` with build command `pip install -r requirements.txt` and start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Add your `GEMINI_API_KEY` under Environment Variables.
