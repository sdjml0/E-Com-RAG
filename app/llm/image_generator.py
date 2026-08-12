import os
import base64
import logging
import urllib.parse
from typing import Optional
from app.config import settings

logger = logging.getLogger("image_generator")

class ImageGenerationEngine:
    """
    Simple Amazon & Flipkart Style Product Listing Image Engine.
    Keeps the product clean, simple, centered, and isolated on a solid marketplace background.
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

    def _generate_simple_ecom_catalog_url(self, prompt: str) -> str:
        """
        Generates a clean, simple Amazon/Flipkart marketplace product photo.
        Clean centered product shot on a plain solid white background with subtle shadow.
        """
        simple_prompt = (
            f"Simple Amazon product listing photo of {prompt[:150]}, "
            f"plain solid white background, centered product view, soft subtle shadow, "
            f"clean official e-commerce store item photo"
        )
        encoded_prompt = urllib.parse.quote(simple_prompt)
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"

    async def generate_product_image(self, image_url: str, prompt: str) -> dict:
        """
        Takes image_url and prompt, returning a clean Amazon/Flipkart style product listing image.
        """
        simple_prompt = (
            f"Official marketplace product listing photo for ({image_url}): {prompt}. "
            f"Plain clean white background, centered product, soft natural shadow."
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
                    input=simple_prompt,
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
                logger.info(f"Gemini image API limit/error ({e}). Using Simple E-Commerce Catalog AI Generator.")

        # Simple Amazon / Flipkart Catalog AI Generator
        free_ai_url = self._generate_simple_ecom_catalog_url(prompt)
        return {
            "status": "success",
            "generated_image_url": free_ai_url,
            "model_used": "flux-simple-ecom"
        }

image_generator = ImageGenerationEngine()
