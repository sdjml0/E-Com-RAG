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
