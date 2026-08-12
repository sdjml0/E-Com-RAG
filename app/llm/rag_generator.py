import time
import json
import logging
import asyncio
from typing import List, AsyncGenerator, Dict, Any
from app.config import settings
from app.schemas import RAGQueryRequest, RAGResponse, ProductResponse
from app.search.hybrid_searcher import hybrid_searcher
from app.schemas import SearchQueryRequest

logger = logging.getLogger("rag_generator")

class RAGGenerator:
    """Multimodal LLM Synthesis Engine (Gemini 2.0 Flash / Structured Context Streaming)."""

    def __init__(self, api_key: str | None = settings.GEMINI_API_KEY):
        self.api_key = api_key
        self.client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Successfully initialized Gemini API client.")
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}. Using intelligent fallback synthesizer.")
                self.client = None
        else:
            logger.info("No GEMINI_API_KEY set. Using intelligent context synthesizer.")

    def _build_context_prompt(self, user_query: str, products: List[ProductResponse]) -> str:
        products_text = "\n".join([
            f"- SKU: {p.product_id} | Title: '{p.prod_title}' | Price: ${p.price:.2f} | Brand: {p.brand} | Category: {p.category} | RRF Score: {p.final_score}"
            for p in products
        ])
        return (
            f"User Shopping Query: '{user_query}'\n\n"
            f"Retrieved Context Catalog Items:\n{products_text}\n\n"
            f"Instructions: You are an expert E-Commerce Personal Shopper AI. "
            f"Provide a concise recommendation rationale for why these products match the user's intent. "
            f"Highlight key features, budget fit, and brand options."
        )

    def _generate_fallback_recommendation(self, user_query: str, products: List[ProductResponse]) -> str:
        if not products:
            return f"I searched the catalog for '{user_query}', but no matching products satisfied your hard brand or price constraints."
        
        top = products[0]
        rec = f"Based on your query '{user_query}', I highly recommend the **{top.prod_title}** by {top.brand.capitalize()} priced at **${top.price:.2f}**.\n\n"
        rec += f"It matches your category requirements ({top.category}) and scored highest in our multi-vector hybrid retrieval pipeline (Score: {top.final_score})."
        
        if len(products) > 1:
            rec += "\n\n**Alternative Top Matches:**\n"
            for p in products[1:]:
                rec += f"- **{p.prod_title}** (${p.price:.2f}, Brand: {p.brand.capitalize()})\n"
        return rec

    async def generate_rag_response(self, request: RAGQueryRequest) -> RAGResponse:
        start_time = time.perf_counter()
        
        # 1. Execute Multi-Vector Search to retrieve grounded context
        search_req = SearchQueryRequest(
            query_text=request.user_query,
            query_image_url=request.query_image_url,
            brand_filter=request.brand_filter,
            category_filter=request.category_filter,
            min_price=request.min_price,
            max_price=request.max_price,
            target_price=request.target_price,
            rag_strategy=request.rag_strategy,
            top_k=request.top_k
        )
        search_res = await hybrid_searcher.execute_search(search_req)
        products = search_res.results

        # 2. LLM Synthesis
        if self.client and products:
            try:
                prompt = self._build_context_prompt(request.user_query, products)
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt
                )
                recommendation = response.text
            except Exception as e:
                logger.error(f"Gemini API generation error: {e}. Falling back to template synthesis.")
                recommendation = self._generate_fallback_recommendation(request.user_query, products)
        else:
            recommendation = self._generate_fallback_recommendation(request.user_query, products)

        exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return RAGResponse(
            query=request.user_query,
            recommendation=recommendation,
            retrieved_products=products,
            execution_time_ms=exec_time_ms
        )

    async def stream_rag_tokens(self, request: RAGQueryRequest) -> AsyncGenerator[str, None]:
        """Streams Server-Sent Events (SSE) tokens for real-time response rendering."""
        start_time = time.perf_counter()
        
        # 1. Retrieve Context
        search_req = SearchQueryRequest(
            query_text=request.user_query,
            query_image_url=request.query_image_url,
            brand_filter=request.brand_filter,
            category_filter=request.category_filter,
            min_price=request.min_price,
            max_price=request.max_price,
            target_price=request.target_price,
            rag_strategy=request.rag_strategy,
            top_k=request.top_k
        )
        search_res = await hybrid_searcher.execute_search(search_req)
        products = search_res.results

        # Event 1: Send Context Payload
        context_data = [p.model_dump() for p in products]
        yield f"event: context\ndata: {json.dumps({'retrieved_products': context_data})}\n\n"
        await asyncio.sleep(0.01)

        # 2. Stream Recommendation text chunks
        rec_text = self._generate_fallback_recommendation(request.user_query, products)
        
        # Stream chunks (word by word for fast UI rendering)
        words = rec_text.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.03)

        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield f"event: done\ndata: {json.dumps({'status': 'completed', 'total_latency_ms': total_ms})}\n\n"

    async def generate_simple_rag_recommendation(self, request: Any) -> Any:
        """
        Simplified single response containing product details and tailored Next-Gen AI Image Generation prompt.
        If product parameters are provided, automatically indexes the product into Qdrant vector store.
        """
        from app.schemas import SimpleRAGResponse, SimpleProductDetail, ImageGenPrompt, ProductIngestRequest
        from app.embeddings.text_embedder import text_embedder
        from app.embeddings.vision_embedder import vision_embedder
        from app.db.vector_db import vector_db_manager

        # 1. Auto-ingest product if title and price are supplied
        if request.prod_title and request.price and request.prod_image_url:
            try:
                prod_id = f"SKU-{abs(hash(request.prod_title)) % 100000:05d}"
                ingest_req = ProductIngestRequest(
                    product_id=prod_id,
                    prod_title=request.prod_title,
                    prod_image_url=request.prod_image_url,
                    price=request.price,
                    category=request.category or "General",
                    brand=request.brand or "Generic"
                )
                composite_text = f"Brand: {ingest_req.brand} | Title: {ingest_req.prod_title} | Category: {ingest_req.category}"
                t_vec = await text_embedder.embed_text(composite_text)
                i_vec = await vision_embedder.embed_image_url(str(ingest_req.prod_image_url))
                await vector_db_manager.upsert_product(ingest_req, t_vec, i_vec)
            except Exception as e:
                logger.warning(f"Auto-ingestion skipped or failed ({e}). Proceeding to search.")

        brand_list = [request.brand] if request.brand and request.brand.strip() else None
        
        search_req = SearchQueryRequest(
            query_text=request.query,
            query_image_url=request.prod_image_url,
            brand_filter=brand_list,
            category_filter=request.category,
            target_price=request.price,
            max_price=request.price * 1.3 if request.price else None,
            rag_strategy="hybrid",
            top_k=1
        )
        search_res = await hybrid_searcher.execute_search(search_req)

        
        if search_res.results:
            top = search_res.results[0]
            prod_id = top.product_id
            title = top.prod_title
            brand = top.brand
            category = top.category
            price = top.price
            img_url = top.prod_image_url
            score = top.final_score
            reasoning = f"Top-ranked candidate for '{request.query}'. Features premium {brand.capitalize()} design in {category} priced at ${price:.2f}."
        else:
            prod_id = "SKU-CUSTOM-001"
            title = request.prod_title or f"Custom Product for '{request.query}'"
            brand = request.brand or "ECOM Brand"
            category = request.category or "Electronics"
            price = request.price or 299.99
            img_url = str(request.prod_image_url) if request.prod_image_url else "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"
            score = 0.9500
            reasoning = f"Custom tailored response generated for query: '{request.query}'."

        # Construct image generation & edit prompt for Next-Gen AI Models (Imagen 3 / Flux / Midjourney)
        image_prompt_text = (
            f"Studio product photography of {title} by {brand.capitalize()}, "
            f"crafted for '{request.query}'. Rendered in clean commercial aesthetic, "
            f"high-detail texture, soft studio lighting, neutral minimalist background, 8k resolution, photorealistic."
        )

        style_modifiers = [
            f"{brand.capitalize()} signature style",
            "soft studio illumination",
            "clean minimalist backdrop",
            "8k ultra-detailed texture"
        ]

        product_details = SimpleProductDetail(
            product_id=prod_id,
            title=title,
            brand=brand,
            category=category,
            price=price,
            image_url=img_url,
            match_score=score,
            reasoning=reasoning
        )

        image_prompt = ImageGenPrompt(
            prompt=image_prompt_text,
            action="generate_or_edit",
            base_image_url=img_url,
            style_modifiers=style_modifiers,
            aspect_ratio="1:1"
        )

        from app.llm.image_generator import image_generator
        gen_res = await image_generator.generate_image(prompt=image_prompt_text, base_image_url=img_url)
        gen_image_url = gen_res.get("image_url", img_url)

        return SimpleRAGResponse(
            query=request.query,
            product_details=product_details,
            image_generation_prompt=image_prompt,
            generated_image_url=gen_image_url
        )

rag_generator = RAGGenerator()


