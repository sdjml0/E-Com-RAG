import time
import asyncio
import httpx
import numpy as np

BASE_URL = "http://localhost:8000/api/v1"

TEST_QUERIES = [
    {
        "name": "Text Search with Brand & Price Filter",
        "payload": {
            "query_text": "wireless noise cancelling headphones",
            "brand_filter": ["Sony"],
            "max_price": 400.0,
            "target_price": 350.0,
            "rag_strategy": "hybrid",
            "top_k": 5
        }
    },
    {
        "name": "Category Filter & Budget Soft Elasticity",
        "payload": {
            "query_text": "apple smartwatch GPS",
            "category_filter": "Electronics > Wearables",
            "target_price": 400.0,
            "rag_strategy": "price_elastic",
            "top_k": 3
        }
    },
    {
        "name": "Text Only Search Strategy",
        "payload": {
            "query_text": "running shoes sneakers",
            "rag_strategy": "text_only",
            "top_k": 5
        }
    }
]

async def evaluate_pipeline():
    print("=" * 60)
    print("🚀 Starting Multimodal RAG Microservice Latency & Evaluation Benchmark")
    print("=" * 60)

    latencies = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check health
        health_resp = await client.get(f"{BASE_URL}/health")
        if health_resp.status_code != 200:
            print("❌ Service health check failed. Is the server running?")
            return
        
        health_data = health_resp.json()
        print(f"✅ Service Health: {health_data['status']} | Indexed Points: {health_data['total_vectors_indexed']}\n")

        for test in TEST_QUERIES:
            name = test["name"]
            payload = test["payload"]
            
            t0 = time.perf_counter()
            resp = await client.post(f"{BASE_URL}/search", json=payload)
            t1 = time.perf_counter()
            
            duration_ms = (t1 - t0) * 1000
            latencies.append(duration_ms)

            if resp.status_code == 200:
                res = resp.json()
                print(f"Test: '{name}'")
                print(f"  - SLA Latency: {duration_ms:.2f}ms (Server Execution: {res['execution_time_ms']}ms)")
                print(f"  - Hits Returned: {res['total_hits']}")
                if res['results']:
                    top = res['results'][0]
                    print(f"  - Top Match: {top['prod_title']} | Score: {top['final_score']} | Penalty: {top['score_breakdown']['price_penalty']}")
                print("-" * 50)
            else:
                print(f"❌ Test '{name}' failed with status {resp.status_code}: {resp.text}")

        p95 = np.percentile(latencies, 95)
        print(f"\n📊 Evaluation Summary:")
        print(f"  - Total Queries Tested: {len(TEST_QUERIES)}")
        print(f"  - Mean Latency: {np.mean(latencies):.2f}ms")
        print(f"  - P95 Latency: {p95:.2f}ms")
        if p95 < 120.0:
            print("  - SLA Requirement (< 120ms): PASSED 🟢")
        else:
            print("  - SLA Requirement (< 120ms): WARNING 🟡")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())
