import pytest
from app.schemas import RecommendationInput
from app.llm.rag_generator import rag_generator

@pytest.mark.asyncio
async def test_cerave_evidence_provenance_grounding():
    req = RecommendationInput(
        prod_title="CeraVe Hydrating Facial Cleanser",
        prod_image_url="https://images.unsplash.com/photo-1556228720-195a672e8a03",
        price=15.99,
        category="Beauty > Skincare > Cleansers",
        brand="CeraVe",
        query="Hydrating non-foaming cleanser for dry skin with ceramides"
    )

    res = await rag_generator.generate_recommendation(req)

    # 1. Assert Strict Recommendation Output
    assert res.product_description is not None
    brand_val = res.detected_product_specifications_and_attributes.get("brand", "")
    assert "cerave" in brand_val.lower()

    # 2. Assert Fact Provenance Citing Sources & Evidence Spans
    assert res.fact_provenance is not None
    assert len(res.fact_provenance) > 0

    prov_list = {prov.attribute: prov for prov in res.fact_provenance}

    # Verify brand & model have exact document IDs & evidence spans
    assert "brand" in prov_list
    brand_prov = prov_list["brand"]
    assert brand_prov.source_document_id is not None
    assert len(brand_prov.evidence_span) > 0
    assert brand_prov.verified_status is True

    assert "model" in prov_list or "category" in prov_list

    # 3. Assert Retrieval Debug Telemetry
    assert res.retrieval_debug is not None
    assert res.retrieval_debug.hallucination_rate == 0.0
