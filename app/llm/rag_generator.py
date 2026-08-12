import time
import json
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional
from app.config import settings
from app.schemas import (
    RecommendationInput,
    StrictRecommendationResponse,
    ProductIngestRequest
)

logger = logging.getLogger("rag_generator")

class RAGGenerator:
    """Next-Gen E-Commerce Ad Copy Synthesizer (Gemini 3.1 Flash Interactions + Rich Fallback)."""

    def __init__(self, api_key: Optional[str] = settings.GEMINI_API_KEY):
        self.api_key = api_key
        self.client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Successfully initialized Gemini API client for interactions.")
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}.")
                self.client = None

    def _clean_json_str(self, text: str) -> str:
        """Strips markdown code fences (```json ... ```) from LLM output string."""
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _build_rich_ad_fallback(self, title: str, brand: str, category: str, price: float) -> StrictRecommendationResponse:
        """Generates a rich, point-to-point, highly attractive seller advertisement copy fallback."""
        brand_cap = brand.capitalize() if brand else "Premium"
        cat_clean = " ".join(category.replace(">", " ").split()) if category else "General"
        est_price = round(price if price > 0 else 299.99, 2)


        desc = (
            f"Experience absolute craftsmanship with the {title} by {brand_cap}. "
            f"Engineered for those who refuse to compromise, this flagship model combines sleek, "
            f"modern aesthetics with studio-grade performance to transform your daily routine into an extraordinary experience."
        )

        features = [
            f"🔥 Flagship {brand_cap} Craftsmanship: Engineered for peak performance and uncompromised durability",
            f"🎧 Immersive Audio & Smart Tech: Tailored specifically for modern {cat_clean} enthusiasts",
            f"⚡ Long-Lasting Battery & Fast Charge: Designed for all-day performance without interruptions",
            f"☁️ Ultra-Lightweight Ergonomic Comfort: Premium materials crafted for long-duration wear"
        ]

        specs = {
            "brand": brand_cap,
            "model_name": title,
            "category": category,
            "build_quality": "Reinforced Ergonomic Alloy",
            "connectivity": "Wireless High-Bandwidth Bluetooth & Fast Charge",
            "warranty_rating": "1-Year Official Manufacturer Warranty"
        }

        seo = [
            f"{brand.lower()} {title.lower()}",
            f"best {cat_clean.lower()} 2026",
            f"buy {title.lower()} online",
            f"{brand.lower()} flagship price and features",
            f"top rated {cat_clean.lower()}"
        ]

        img_prompt = (
            f"Commercial studio product photography of {title} by {brand_cap}, "
            f"rendered in clean commercial e-commerce aesthetic, soft studio lighting, "
            f"minimalist background, high-detail texture, 8k resolution, photorealistic."
        )

        return StrictRecommendationResponse(
            product_description=desc,
            estimated_price=est_price,
            key_features=features,
            detected_product_specifications_and_attributes=specs,
            mined_high_rank_seo_keywords=seo,
            best_prompt_for_image_enhancement=img_prompt
        )

    async def generate_recommendation(self, request: RecommendationInput) -> StrictRecommendationResponse:
        """
        Generates high-converting, persuasive, point-to-point seller advertisement ad copy.
        Uses Gemini 3.1 Flash Lite Interactions API with rich fallback protection.
        """
        title = request.prod_title.strip()
        brand = request.brand.strip()
        category = request.category.strip()
        price = request.price if request.price > 0 else 299.99

        # Auto-ingest into vector DB for grounding
        try:
            prod_id = f"SKU-{abs(hash(request.prod_title)) % 100000:05d}"
            ingest_req = ProductIngestRequest(
                product_id=prod_id,
                prod_title=title,
                prod_image_url=request.prod_image_url,
                price=price,
                category=category,
                brand=brand
            )
            from app.embeddings.text_embedder import text_embedder
            from app.embeddings.vision_embedder import vision_embedder
            from app.db.vector_db import vector_db_manager

            composite_text = f"Brand: {brand} | Title: {title} | Category: {category}"
            t_vec = await text_embedder.embed_text(composite_text)
            i_vec = await vision_embedder.embed_image_url(str(request.prod_image_url))
            await vector_db_manager.upsert_product(ingest_req, t_vec, i_vec)
        except Exception as e:
            logger.warning(f"Auto-ingestion check: {e}")

        # Attempt Gemini 3.1 Flash Lite Interactions API for Rich Ad Copy Generation
        if self.client:
            try:
                prompt = (
                    f"You are an elite E-Commerce Copywriter, Advertising Strategist, and Product Specialist.\n"
                    f"Generate a high-converting, irresistible, point-to-point seller advertisement ad response for this product:\n"
                    f"Title: {title}\n"
                    f"Brand: {brand}\n"
                    f"Category: {category}\n"
                    f"Price: ${price:.2f}\n\n"
                    f"Return ONLY a valid raw JSON object matching this exact schema:\n"
                    f"{{\n"
                    f'  "product_description": "High-impact, highly attractive sales ad description (2-3 punchy, persuasive sentences that make customers want to buy immediately)",\n'
                    f'  "estimated_price": {price:.2f},\n'
                    f'  "key_features": [\n'
                    f'    "🎧 Feature 1: Point-to-point attractive benefit + spec",\n'
                    f'    "⚡ Feature 2: Point-to-point attractive benefit + spec",\n'
                    f'    "💎 Feature 3: Point-to-point attractive benefit + spec",\n'
                    f'    "☁️ Feature 4: Point-to-point attractive benefit + spec"\n'
                    f'  ],\n'
                    f'  "detected_product_specifications_and_attributes": {{\n'
                    f'    "brand": "{brand}",\n'
                    f'    "model": "{title}",\n'
                    f'    "category": "{category}",\n'
                    f'    "key_spec_1": "value",\n'
                    f'    "key_spec_2": "value"\n'
                    f'  }},\n'
                    f'  "mined_high_rank_seo_keywords": [\n'
                    f'    "keyword 1", "keyword 2", "keyword 3", "keyword 4"\n'
                    f'  ],\n'
                    f'  "best_prompt_for_image_enhancement": "Hyper-realistic commercial studio product photography prompt..."\n'
                    f"}}\n"
                )

                interaction = self.client.interactions.create(
                    model='models/gemini-3.1-flash-lite',
                    input=prompt
                )
                output_text = getattr(interaction, 'output_text', '')
                clean_text = self._clean_json_str(output_text)

                if clean_text:
                    parsed = json.loads(clean_text)
                    return StrictRecommendationResponse(
                        product_description=parsed.get("product_description", "").strip(),
                        estimated_price=float(parsed.get("estimated_price", price)),
                        key_features=parsed.get("key_features", []),
                        detected_product_specifications_and_attributes=parsed.get("detected_product_specifications_and_attributes", {}),
                        mined_high_rank_seo_keywords=parsed.get("mined_high_rank_seo_keywords", []),
                        best_prompt_for_image_enhancement=parsed.get("best_prompt_for_image_enhancement", "").strip()
                    )
            except Exception as e:
                logger.warning(f"Gemini 3.1 Flash interaction error ({e}). Using Rich Ad Synthesizer fallback.")

        # Fallback to Rich Seller Advertisement Copy Synthesizer
        return self._build_rich_ad_fallback(title, brand, category, price)

rag_generator = RAGGenerator()
