import pytest
from app.schemas import StrictRecommendationResponse
from app.adapters.marketplace import marketplace_adapter_engine

def test_marketplace_adapter_transformation():
    sample_universal_json = StrictRecommendationResponse(
        product_description="High-end noise-canceling headphones with studio grade drivers.",
        estimated_price=398.00,
        key_features=[
            "Industry leading noise cancelling",
            "30-hour battery life",
            "Ultra comfortable lightweight design",
            "Hands-free calling"
        ],
        detected_product_specifications_and_attributes={
            "brand": "Sony",
            "model_name": "Sony WH-1000XM5",
            "primary_color": "Black",
            "material_build": "Synthetic Leather & Alloy",
            "connectivity_tech": "Bluetooth 5.3",
            "intended_usage": "Travel & Office"
        },
        mined_high_rank_seo_keywords=["sony headphones", "wh-1000xm5", "noise cancelling"],
        best_prompt_for_image_enhancement="Studio product photo of Sony WH-1000XM5 on white background"
    )

    res = marketplace_adapter_engine.adapt(sample_universal_json, brand="Sony")

    # Amazon assertions
    assert "Sony" in res.amazon.title
    assert len(res.amazon.bullet_points) == 4
    assert res.amazon.attributes["model_name"] == "Sony WH-1000XM5"

    # Flipkart assertions
    assert "General" in res.flipkart.specification_groups
    assert res.flipkart.specification_groups["General"]["Brand"] == "Sony"

    # Shopify assertions
    assert res.shopify.vendor == "Sony"
    assert res.shopify.variants[0]["price"] == 398.00
