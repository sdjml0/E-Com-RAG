import os
import base64
import logging
import urllib.parse
from typing import Optional
from app.config import settings

logger = logging.getLogger("image_generator")

class ImageGenerationEngine:
    """
    E-Commerce Marketplace Product-Level Image Generation Pipeline.
    Engineered specifically for clean Amazon/Shopify/Flipkart white-background catalog standards.
    """

    def __init__(self, api_key: Optional[str] = settings.GEMINI_API_KEY):
        self.api_key = api_key
        self.client = None
        self._init_genai_client()

    def _init_genai_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google Gemini client for image generation.")
            except Exception as e:
                logger.warning(f"Could not initialize genai client: {e}")
                self.client = None

    def _generate_ecom_catalog_image_url(self, prompt: str) -> str:
        """
        Generates HD 1024x1024 E-Commerce Catalog Product Images using FLUX Realism AI.
        Optimized for pure white studio background and bright softbox commercial marketplace lighting.
        """
        ecom_prompt = (
            f"High-end commercial e-commerce product catalog photo of {prompt[:160]}, "
            f"isolated on seamless pure white studio background, bright softbox commercial studio lighting, "
            f"centered hero composition, sharp focus, ultra-crisp 8k detail, professional online marketplace listing picture"
        )
        encoded_prompt = urllib.parse.quote(ecom_prompt)
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"

    async def generate_product_image(self, image_url: str, prompt: str) -> dict:
        """
        Takes image_url and prompt, returning a clean, high-resolution product-level e-commerce catalog image.
        """
        full_prompt = (
            f"E-commerce product photo for listing ({image_url}): {prompt}. "
            f"Pure white studio background, bright commercial studio lighting, sharp product focus."
        )

        if self.client:
            try:
                generation_config = {
                    'temperature': 1,
                    'max_output_tokens': 65536,
                    'top_p': 0.95,
                    'thinking_level': 'minimal',
                }

                interaction = self.client.interactions.create(
                    model='models/gemini-3.1-flash-lite-image',
                    input=full_prompt,
                    generation_config=generation_config,
                    response_modalities=['image', 'text'],
                )

                extracted_b64 = None

                for step in getattr(interaction, "steps", []):
                    if getattr(step, "type", "") == 'model_output' and getattr(step, "content", None):
                        for part in step.content:
                            p_type = getattr(part, "type", "")
                            if p_type == 'image':
                                raw_data = getattr(part, "data", None)
                                if raw_data:
                                    if isinstance(raw_data, bytes):
                                        extracted_b64 = base64.b64encode(raw_data).decode("utf-8")
                                    else:
                                        extracted_b64 = str(raw_data)

                if extracted_b64:
                    return {
                        "status": "success",
                        "generated_image_url": f"data:image/png;base64,{extracted_b64}",
                        "model_used": "models/gemini-3.1-flash-lite-image"
                    }

            except Exception as e:
                logger.info(f"Gemini image API limit/error ({e}). Using FLUX E-Commerce Catalog AI Generator.")

        # High-Resolution FLUX E-Commerce Marketplace Catalog AI Generator
        free_ai_url = self._generate_ecom_catalog_image_url(prompt)
        return {
            "status": "success",
            "generated_image_url": free_ai_url,
            "model_used": "flux-ecom-catalog-ai"
        }

image_generator = ImageGenerationEngine()
