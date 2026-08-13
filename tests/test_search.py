import pytest
from app.search.rrf import RRFRanker
from app.schemas import ScoreBreakdown

class DummyHit:
    def __init__(self, product_id: str, price: float):
        self.payload = {
            "product_id": product_id,
            "prod_title": f"Product {product_id}",
            "price": price,
            "brand": "sony",
            "category": "Audio"
        }

def test_rrf_scoring_and_price_penalty():
    ranker = RRFRanker(k=60.0, alpha=0.3, min_factor=0.5)

    text_hits = [DummyHit("SKU-1", 100.0), DummyHit("SKU-2", 200.0)]
    image_hits = [DummyHit("SKU-2", 200.0), DummyHit("SKU-1", 100.0)]

    # Target price 100.0 -> SKU-1 has 0 penalty (1.0), SKU-2 has 0.3 * (100/100) = 0.3 -> penalty 0.7
    results = ranker.compute_rrf_and_penalties(
        text_hits=text_hits,
        image_hits=image_hits,
        target_price=100.0
    )

    assert len(results) == 2
    # Check that candidates are returned and ranked
    pid1, score1, breakdown1, _ = results[0]
    assert pid1 in ("SKU-1", "SKU-2")
    assert breakdown1.rrf_score > 0
    assert breakdown1.price_penalty <= 1.0

@pytest.mark.asyncio
async def test_vector_db_product_retrieval():
    from app.db.vector_db import vector_db_manager
    from app.search.hybrid_searcher import hybrid_searcher
    from app.schemas import ProductIngestRequest, SearchQueryRequest
    from app.embeddings.text_embedder import text_embedder
    from app.embeddings.vision_embedder import vision_embedder

    await vector_db_manager.init_collection()
    prod = ProductIngestRequest(
        product_id="TEST-HEADPHONE-999",
        prod_title="Sony WH-1000XM5 Premium Noise Canceling Headphones",
        prod_image_url="https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
        price=399.00,
        category="Electronics > Audio > Headphones",
        brand="Sony"
    )
    t_vec = await text_embedder.embed_text(f"{prod.brand} {prod.prod_title} {prod.category}")
    i_vec = await vision_embedder.embed_image_url(str(prod.prod_image_url))

    await vector_db_manager.upsert_product(prod, text_vector=t_vec, image_vector=i_vec)

    # Execute vector retrieval
    search_req = SearchQueryRequest(
        query_text="Noise Canceling Headphones",
        brand_filter=["Sony"],
        top_k=5
    )
    search_res = await hybrid_searcher.execute_search(search_req)

    assert search_res.total_hits > 0
    found_ids = [item.product_id for item in search_res.results]
    assert "TEST-HEADPHONE-999" in found_ids
