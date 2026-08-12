# 🔌 E-Commerce AI Agent & Frontend Integration Specification

**Target Microservice:** Production Multimodal E-Commerce RAG Engine  
**Version:** `v1.0.0`  
**Protocol:** REST HTTP/2 + SSE (Server-Sent Events)  
**Data Format:** `application/json`, `text/event-stream`  
**Target SLA:** Search Latency < 120ms (p95), End-to-End Streaming RAG < 750ms (p95)

---

## 📑 Table of Contents
1. [Executive Summary & Connection Guide](#1-executive-summary--connection-guide)
2. [Health Diagnostics & Probes](#2-health-diagnostics--probes)
3. [RAG Strategy & Selector Parameters](#3-rag-strategy--selector-parameters)
4. [Catalog Data Ingestion API](#4-catalog-data-ingestion-api)
5. [Multi-Vector Hybrid Search API](#5-multi-vector-hybrid-search-api)
6. [Multimodal LLM RAG Streaming API](#6-multimodal-llm-rag-streaming-api)
7. [Real-Time Pipeline Telemetry & SSE Event Bus](#7-real-time-pipeline-telemetry--sse-event-bus)
8. [Code Examples for E-COM Agents](#8-code-examples-for-e-com-agents)

---

## 1. Executive Summary & Connection Guide

This microservice provides high-performance, multi-vector retrieval and multimodal RAG generation for e-commerce catalog search. It handles the 5 core product parameters: `prod_image`, `prod_title`, `price`, `category`, and `brand`.

### Microservice Base URL Configuration
* **Local Development:** `http://localhost:8000`
* **Production Cluster:** `https://rag-service.internal.ecom.domain/api/v1`

---

## 2. Health Diagnostics & Probes

Before routing traffic, the E-COM agent or load balancer should poll health endpoints.

### 2.1 Liveness Probe
`GET /healthz/liveness`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "timestamp": "2026-08-12T18:00:00Z"
}
```

### 2.2 Readiness Probe
`GET /healthz/readiness`

Checks Qdrant Vector DB connection and embedding model readiness.

**Response (`200 OK`):**
```json
{
  "status": "ready",
  "vector_db": "connected",
  "text_embedder": "loaded",
  "vision_embedder": "loaded",
  "collection": "ecommerce_products_v1"
}
```

### 2.3 System Telemetry & Metrics
`GET /api/v1/health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "total_vectors_indexed": 45210,
  "qdrant_status": "green",
  "p95_latency_ms": 42.5,
  "active_event_subscribers": 3
}
```

---

## 3. RAG Strategy & Selector Parameters

When sending search or RAG generation requests, the E-COM agent can control the retrieval pipeline using the **RAG Selector Matrix**:

| `rag_strategy` | Description | Typical Use Case |
| :--- | :--- | :--- |
| `hybrid` **(Recommended)** | Fuses Dense Text (`bge-m3`), Dense Vision (`OpenCLIP`), and hard payload filters with Reciprocal Rank Fusion (RRF). | Standard multimodal product search & recommendations. |
| `text_only` | Uses dense text vectors + payload pre-filtering. Bypasses image processing. | Text queries with hard brand/price constraints. |
| `vision_only` | Uses image visual embeddings + payload pre-filtering. | "Find similar visual style" or photo upload search. |
| `price_elastic` | Applies exponential budget cap penalties on products deviating from `target_price`. | "Budget-conscious" or "Deals under $X" intent. |

---

## 4. Catalog Data Ingestion API

To sync products into the RAG vector index in real time.

### 4.1 Ingest Single Product
`POST /api/v1/products/ingest`

**Request Payload:**
```json
{
  "product_id": "SKU-HEADPHONE-001",
  "prod_title": "Sony WH-1000XM5 Wireless Noise-Canceling Headphones",
  "prod_image_url": "https://images.ecom.com/products/wh1000xm5.jpg",
  "price": 398.00,
  "category": "Electronics > Audio > Headphones",
  "brand": "Sony"
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "product_id": "SKU-HEADPHONE-001",
  "message": "Product successfully embedded and indexed into Qdrant."
}
```

---

## 5. Multi-Vector Hybrid Search API

`POST /api/v1/search`

### Request Schema (`SearchQueryRequest`)
```json
{
  "query_text": "wireless noise cancelling headphones",
  "query_image_url": "https://images.ecom.com/queries/user_uploaded_headphone.jpg",
  "brand_filter": ["Sony", "Bose"],
  "category_filter": "Electronics > Audio",
  "min_price": 100.0,
  "max_price": 450.0,
  "target_price": 350.0,
  "rag_strategy": "hybrid",
  "top_k": 5,
  "weights": {
    "text": 0.45,
    "image": 0.35,
    "bm25": 0.20
  }
}
```

### Response Schema (`SearchQueryResponse`)
```json
{
  "total_hits": 5,
  "execution_time_ms": 28.4,
  "rag_strategy": "hybrid",
  "results": [
    {
      "product_id": "SKU-HEADPHONE-001",
      "prod_title": "Sony WH-1000XM5 Wireless Noise-Canceling Headphones",
      "prod_image_url": "https://images.ecom.com/products/wh1000xm5.jpg",
      "price": 398.00,
      "category": "Electronics > Audio > Headphones",
      "brand": "sony",
      "final_score": 0.0158,
      "score_breakdown": {
        "text_rank": 1,
        "visual_rank": 2,
        "rrf_score": 0.0162,
        "price_penalty": 0.975
      }
    }
  ]
}
```

---

## 6. Multimodal LLM RAG Streaming API

Use this endpoint when the E-COM agent wants natural language recommendation reasoning streamed directly to the end user.

`POST /api/v1/rag/stream`

### Request Payload:
```json
{
  "user_query": "I need quiet over-ear headphones for long flights under $400",
  "query_image_url": null,
  "brand_filter": ["Sony", "Bose", "Sennheiser"],
  "category_filter": "Audio",
  "max_price": 400.0,
  "target_price": 350.0,
  "rag_strategy": "hybrid",
  "top_k": 3
}
```

### Response Format: Server-Sent Events (`text/event-stream`)

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: context
data: {"retrieved_products": [{"product_id": "SKU-HEADPHONE-001", "prod_title": "Sony WH-1000XM5", "price": 398.0}]}

event: token
data: {"text": "Based on your search for quiet over-ear headphones under $400, I recommend the **Sony WH-1000XM5**."}

event: token
data: {"text": " It features industry-leading noise cancellation perfect for long flights and fits right at $398."}

event: done
data: {"status": "completed", "total_latency_ms": 410.2}
```

---

## 7. Real-Time Pipeline Telemetry & SSE Event Bus

For live microservice monitoring dashboards, tracking **moving data**, stage latencies, and **errors/events**:

`GET /api/v1/events/stream`

### Streamed Event Payload Types:

1. **`pipeline_stage`**: Step execution times (`parse`, `filter_build`, `vector_search`, `rrf_fusion`, `price_penalty`).
2. **`moving_data`**: Metrics on products ingested and candidate vector distance scores.
3. **`error_event`**: Circuit breaker events, database timeouts, fallback triggers.

```json
event: pipeline_stage
data: {
  "trace_id": "tr-98124-abc",
  "stage": "vector_search",
  "duration_ms": 12.4,
  "candidates_retrieved": 100
}

event: moving_data
data: {
  "trace_id": "tr-98124-abc",
  "action": "rrf_fusion_complete",
  "top_candidate": "SKU-HEADPHONE-001",
  "top_score": 0.0158
}

event: error_event
data: {
  "timestamp": "2026-08-12T18:05:12Z",
  "level": "WARNING",
  "component": "vision_embedder",
  "message": "Image download timeout for URL https://external.com/img.jpg. Falling back to text_only strategy."
}
```

---

## 8. Code Examples for E-COM Agents

### Python Integration Example (`httpx` + `asyncio`)

```python
import httpx
import asyncio
import json

RAG_SERVICE_URL = "http://localhost:8000/api/v1"

async def query_ecom_rag(user_prompt: str, max_budget: float):
    payload = {
        "user_query": user_prompt,
        "max_price": max_budget,
        "target_price": max_budget * 0.9,
        "rag_strategy": "hybrid",
        "top_k": 3
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Stream SSE recommendation tokens
        async with client.stream("POST", f"{RAG_SERVICE_URL}/rag/stream", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    data = json.loads(data_str)
                    if "text" in data:
                        print(data["text"], end="", flush=True)

if __name__ == "__main__":
    asyncio.run(query_ecom_rag("Best ANC headphones for travel", 400.0))
```

### JavaScript / Node.js Agent Integration Example

```javascript
async function searchCatalog(queryText, brandFilter, maxPrice) {
  const response = await fetch('http://localhost:8000/api/v1/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query_text: queryText,
      brand_filter: brandFilter ? [brandFilter] : null,
      max_price: maxPrice,
      rag_strategy: 'hybrid',
      top_k: 5
    })
  });

  const data = await response.json();
  console.log(`Found ${data.total_hits} products in ${data.execution_time_ms}ms`);
  return data.results;
}
```
