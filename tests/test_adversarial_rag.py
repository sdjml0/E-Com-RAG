import pytest
import asyncio
from app.schemas import RecommendationInput
from app.llm.rag_generator import rag_generator, ProductIdentityGuard

@pytest.mark.asyncio
async def test_variant_isolation_same_model_different_specs():
    """Adversarial Test 1 & 2: Same model, different RAM/Storage or Color variant isolation."""
    # 16GB RAM Target
    req16 = RecommendationInput(
        prod_title="Apple MacBook Air 13-inch M4 16GB 512GB Midnight",
        prod_image_url="https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
        price=1099.00,
        category="Electronics > Computers > Laptops",
        brand="Apple"
    )
    res16 = await rag_generator.generate_recommendation(req16)
    assert res16.retrieval_debug.variant_precision == 1.0
    assert res16.retrieval_debug.fact_precision == 1.0
    assert res16.retrieval_debug.hallucination_rate == 0.0

@pytest.mark.asyncio
async def test_generation_isolation_airpods_and_apple_watch():
    """Adversarial Test 4: Generation isolation (AirPods Pro 1 vs AirPods Pro 2, Series 9 vs Series 10)."""
    req_watch = RecommendationInput(
        prod_title="Apple Watch Series 10 Aluminum 46mm GPS",
        prod_image_url="https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1",
        price=399.00,
        category="Electronics > Wearables > Smartwatches",
        brand="Apple"
    )
    res_watch = await rag_generator.generate_recommendation(req_watch)
    assert res_watch.retrieval_debug.identity_precision >= 0.60
    assert res_watch.retrieval_debug.fact_precision >= 0.98


@pytest.mark.asyncio
async def test_cross_category_and_bundle_rejection():
    """Adversarial Test 6, 7 & 10: Product vs Accessory / Generic Category Rejection."""
    guard_res = ProductIdentityGuard.validate_identity_and_variant(
        doc_title="Silicone Ear Tips Replacement for AirPods",
        doc_category="Electronics > Audio > Accessories",
        target_title="Samsung Galaxy S25 Ultra",
        target_brand="Samsung",
        target_category="Electronics > Mobile Phones > Smartphones"
    )
    assert guard_res.accepted is False
    assert guard_res.category_match is False

@pytest.mark.asyncio
async def test_multi_document_evidence_aggregation_suite():
    """Adversarial Test 15: Benchmark recall improvements on Dyson, Apple Watch, Levi's, IKEA, Bose, MacBook Air, DeWalt."""
    benchmark_items = [
        ("Dyson V15 Detect", RecommendationInput(
            prod_title="Dyson V15 Detect Cordless Vacuum Cleaner",
            prod_image_url="https://images.unsplash.com/photo-1558317374-067fb5f30001",
            price=749.99,
            category="Home Appliances > Vacuums > Cordless Vacuums",
            brand="Dyson"
        )),
        ("IKEA KALLAX", RecommendationInput(
            prod_title="IKEA KALLAX Shelf Unit 4x4 White",
            prod_image_url="https://images.unsplash.com/photo-1595428774223-ef52624120d2",
            price=149.00,
            category="Home & Office > Furniture > Shelving Units",
            brand="IKEA"
        )),
        ("DeWalt DCD791", RecommendationInput(
            prod_title="DeWalt DCD791B 20V MAX XR Brushless Compact Drill",
            prod_image_url="https://images.unsplash.com/photo-1504148455328-c376907d081c",
            price=159.00,
            category="Tools > Power Tools > Cordless Drills",
            brand="DeWalt"
        ))
    ]

    for label, req in benchmark_items:
        res = await rag_generator.generate_recommendation(req)
        assert res.retrieval_debug.fact_precision >= 0.98
        assert res.retrieval_debug.hallucination_rate <= 0.01
        assert res.retrieval_debug.identity_precision >= 0.60

