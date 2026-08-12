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
    assert "total_vectors_indexed" in data

@pytest.mark.asyncio
async def test_simple_rag_recommendation_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "query": "Sleek black wireless noise-canceling headphones for travel under $400",
            "prod_title": "Sony WH-1000XM5",
            "prod_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
            "price": 398.00,
            "category": "Electronics > Audio > Headphones",
            "brand": "Sony"
        }
        resp = await ac.post("/api/v1/rag/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "product_details" in data
        assert "image_generation_prompt" in data
        assert "prompt" in data["image_generation_prompt"]
        assert data["image_generation_prompt"]["action"] == "generate_or_edit"

@pytest.mark.asyncio
async def test_image_generate_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "prompt": "Studio photography of matte black headphones on polished dark wood with ambient neon accent lighting",
            "base_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
            "product_title": "Sony WH-1000XM5",
            "brand": "Sony",
            "aspect_ratio": "1:1"
        }
        resp = await ac.post("/api/v1/image/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("success", "fallback", "success_free_ai")
        assert "generated_image_url" in data


