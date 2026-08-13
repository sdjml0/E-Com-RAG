import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data

@pytest.mark.asyncio
async def test_uptimerobot_ping_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # GET /ping
        res1 = await ac.get("/ping")
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "ok"
        assert data1["server"] == "active"
        assert "uptime_seconds" in data1

        # HEAD /ping
        res2 = await ac.head("/ping")
        assert res2.status_code == 200

        # GET /healthz
        res3 = await ac.get("/healthz")
        assert res3.status_code == 200
        assert res3.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_recommendation_api_strict_pattern():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "prod_title": "Sony WH-1000XM5",
            "prod_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
            "price": 398.00,
            "category": "Electronics > Audio > Headphones",
            "brand": "Sony"
        }
        resp = await ac.post("/api/v1/rag/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "product_description" in data
        assert "estimated_price" in data
        assert isinstance(data["key_features"], list)
        assert "detected_product_specifications_and_attributes" in data
        assert isinstance(data["mined_high_rank_seo_keywords"], list)
        assert "best_prompt_for_image_enhancement" in data

@pytest.mark.asyncio
async def test_image_generation_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
            "prompt": "Studio product photography on polished dark wood with ambient lighting"
        }
        resp = await ac.post("/api/v1/image/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "generated_image_url" in data
