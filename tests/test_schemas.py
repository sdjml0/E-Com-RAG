import pytest
from pydantic import ValidationError
from app.schemas import ProductIngestRequest, SearchQueryRequest, RAGQueryRequest

def test_product_ingest_schema():
    valid_payload = {
        "product_id": "SKU-001",
        "prod_title": "Test Headphone",
        "prod_image_url": "https://example.com/image.jpg",
        "price": 199.99,
        "category": "Electronics > Audio",
        "brand": "Sony"
    }
    req = ProductIngestRequest(**valid_payload)
    assert req.product_id == "SKU-001"
    assert req.price == 199.99

    # Invalid negative price
    invalid_payload = valid_payload.copy()
    invalid_payload["price"] = -10.0
    with pytest.raises(ValidationError):
        ProductIngestRequest(**invalid_payload)

def test_search_query_schema():
    req = SearchQueryRequest(
        query_text="headphones",
        brand_filter=["Sony"],
        min_price=50.0,
        max_price=300.0,
        rag_strategy="hybrid"
    )
    assert req.rag_strategy == "hybrid"
    assert req.top_k == 10
