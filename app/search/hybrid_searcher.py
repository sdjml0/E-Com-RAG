import time
import asyncio
import logging
from typing import List
from qdrant_client.http import models as rest_models
from app.db.vector_db import vector_db_manager
from app.embeddings.text_embedder import text_embedder
from app.embeddings.vision_embedder import vision_embedder
from app.search.rrf import rrf_ranker
from app.telemetry.event_bus import event_bus
from app.schemas import (
    SearchQueryRequest,
    SearchQueryResponse,
    ProductResponse,
    ScoreBreakdown
)

logger = logging.getLogger("hybrid_searcher")

class HybridSearcher:
    """Async 2-Stage Hybrid Search Service with Single-Stage Payload Pre-Filtering."""

    def __init__(self, db_manager=vector_db_manager):
        self.db = db_manager

    def _build_payload_filter(self, request: SearchQueryRequest) -> rest_models.Filter | None:
        must_conditions = []

        # 1. Brand Pre-filter (MatchAny)
        if request.brand_filter and len(request.brand_filter) > 0:
            clean_brands = [b.strip().lower() for b in request.brand_filter if b.strip()]
            if clean_brands:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="brand",
                        match=rest_models.MatchAny(any=clean_brands)
                    )
                )

        # 2. Category Path Pre-filter (Match leaf level token e.g. "wearables" from "Electronics > Wearables")
        if request.category_filter and request.category_filter.strip():
            cat_parts = [c.strip().lower() for c in request.category_filter.split(">") if c.strip()]
            if cat_parts:
                leaf_category = cat_parts[-1]
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="category_path",
                        match=rest_models.MatchValue(value=leaf_category)
                    )
                )


        # 3. Price Range Pre-filter (Range gte / lte)
        if request.min_price is not None or request.max_price is not None:
            range_args = {}
            if request.min_price is not None:
                range_args["gte"] = request.min_price
            if request.max_price is not None:
                range_args["lte"] = request.max_price
            must_conditions.append(
                rest_models.FieldCondition(
                    key="price",
                    range=rest_models.Range(**range_args)
                )
            )

        return rest_models.Filter(must=must_conditions) if must_conditions else None

    async def execute_search(self, request: SearchQueryRequest, trace_id: str | None = None) -> SearchQueryResponse:
        start_time = time.perf_counter()
        await self.db.ensure_collection()
        
        # Build Qdrant Single-Stage Payload Filter
        qdrant_filter = self._build_payload_filter(request)


        w_text = request.weights.text if request.weights else None
        w_image = request.weights.image if request.weights else None

        text_hits = []
        image_hits = []

        # Determine strategy execution
        strategy = request.rag_strategy

        # Embed text if provided & strategy allows
        if request.query_text and strategy in ("hybrid", "text_only", "price_elastic"):
            t0 = time.perf_counter()
            text_vec = await text_embedder.embed_text(request.query_text)
            t1 = time.perf_counter()
            
            text_res = await self.db.client.query_points(
                collection_name=self.db.collection_name,
                query=text_vec,
                using="text_vector",
                query_filter=qdrant_filter,
                limit=50
            )
            text_hits = text_res.points
            await event_bus.publish("pipeline_stage", {
                "stage": "text_vector_search",
                "duration_ms": round((t1 - t0) * 1000, 2),
                "hits_count": len(text_hits)
            }, trace_id=trace_id)

        # Embed image if provided & strategy allows
        if request.query_image_url and strategy in ("hybrid", "vision_only", "price_elastic"):
            t0 = time.perf_counter()
            img_vec = await vision_embedder.embed_image_url(str(request.query_image_url))
            t1 = time.perf_counter()
            
            img_res = await self.db.client.query_points(
                collection_name=self.db.collection_name,
                query=img_vec,
                using="image_vector",
                query_filter=qdrant_filter,
                limit=50
            )
            image_hits = img_res.points
            await event_bus.publish("pipeline_stage", {
                "stage": "vision_vector_search",
                "duration_ms": round((t1 - t0) * 1000, 2),
                "hits_count": len(image_hits)
            }, trace_id=trace_id)


        # Execute Reciprocal Rank Fusion & Price Penalty
        scored_candidates = rrf_ranker.compute_rrf_and_penalties(
            text_hits=text_hits,
            image_hits=image_hits,
            target_price=request.target_price,
            weight_text=w_text,
            weight_image=w_image
        )

        top_candidates = scored_candidates[:request.top_k]

        results = [
            ProductResponse(
                product_id=pid,
                prod_title=payload["prod_title"],
                prod_image_url=payload["prod_image_url"],
                price=payload["price"],
                category=payload["category"],
                brand=payload["brand"],
                final_score=round(score, 6),
                score_breakdown=breakdown
            )
            for pid, score, breakdown, payload in top_candidates
        ]

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Broadcast moving data telemetry event
        await event_bus.publish("moving_data", {
            "action": "hybrid_search_complete",
            "hits_found": len(results),
            "execution_time_ms": total_time_ms,
            "top_product": results[0].product_id if results else None
        }, trace_id=trace_id)

        return SearchQueryResponse(
            total_hits=len(results),
            execution_time_ms=total_time_ms,
            rag_strategy=strategy,
            results=results
        )

    # Alias for search execution
    async def search(self, request: SearchQueryRequest, trace_id: str | None = None) -> SearchQueryResponse:
        return await self.execute_search(request, trace_id)

hybrid_searcher = HybridSearcher()

