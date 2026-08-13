import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_product_ingest_and_vector_db_store():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "product_id": "SKU-TEST-VECTORDB-001",
            "prod_title": "CeraVe Hydrating Cleanser 8 fl oz",
            "prod_image_url": "https://images.unsplash.com/photo-1556228720-195a672e8a03",
            "price": 15.99,
            "category": "Beauty > Skincare > Cleansers",
            "brand": "CeraVe"
        }
        res = await ac.post("/api/v1/products/ingest", json=payload)
        assert res.status_code == 200
        json_resp = res.json()
        assert json_resp["status"] == "success"
        assert json_resp["ingested_count"] == 1

        # Verify Vector DB Health count updated
        health_res = await ac.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["total_vectors_indexed"] >= 1
