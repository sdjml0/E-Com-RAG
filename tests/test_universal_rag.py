import pytest
import asyncio
from app.schemas import RecommendationInput
from app.llm.rag_generator import rag_generator

@pytest.mark.asyncio
async def test_universal_rag_benchmark_suite():
    test_categories = [
        ("1. Apparel", RecommendationInput(
            prod_title="Levi's 501 Original Fit Jeans",
            prod_image_url="https://images.unsplash.com/photo-1542272604-780c36856d64",
            price=79.50,
            category="Apparel > Men's Clothing > Jeans",
            brand="Levi's"
        )),
        ("2. Furniture", RecommendationInput(
            prod_title="Herman Miller Aeron Ergonomic Chair",
            prod_image_url="https://images.unsplash.com/photo-1580481072645-022f9a6d1209",
            price=1295.00,
            category="Home & Office > Furniture > Office Chairs",
            brand="Herman Miller"
        )),
        ("3. Automotive Parts", RecommendationInput(
            prod_title="Bosch Icon 26A Wiper Blade",
            prod_image_url="https://images.unsplash.com/photo-1486006920555-c77dce18193b",
            price=27.99,
            category="Automotive > Replacement Parts > Wiper Blades",
            brand="Bosch"
        )),
        ("4. Beauty & Skincare", RecommendationInput(
            prod_title="CeraVe Hydrating Facial Cleanser",
            prod_image_url="https://images.unsplash.com/photo-1556228720-195a672e8a03",
            price=14.99,
            category="Beauty & Personal Care > Skincare > Facial Cleansers",
            brand="CeraVe"
        )),
        ("5. Groceries", RecommendationInput(
            prod_title="California Olive Ranch Extra Virgin Olive Oil",
            prod_image_url="https://images.unsplash.com/photo-1474979266404-7eaacbcd87c5",
            price=18.99,
            category="Grocery & Gourmet Food > Pantry Staples > Cooking Oils",
            brand="California Olive Ranch"
        )),
        ("6. Electronics", RecommendationInput(
            prod_title="Samsung Galaxy S25 Ultra",
            prod_image_url="https://images.unsplash.com/photo-1610945265064-0e34e5519bbf",
            price=1299.00,
            category="Electronics > Mobile Phones > Smartphones",
            brand="Samsung"
        ))
    ]

    for label, req in test_categories:
        res = await rag_generator.generate_recommendation(req)
        assert res.product_description is not None
        assert res.best_prompt_for_image_enhancement is not None
        assert res.retrieval_debug is not None

        dbg = res.retrieval_debug
        assert dbg.retrieval_recall <= 1.0
        assert dbg.fact_precision <= 1.0
        assert dbg.hallucination_rate <= 1.0
        assert dbg.evidence_coverage <= 1.0
        assert len(dbg.verified_fact_evidence or []) >= 1
