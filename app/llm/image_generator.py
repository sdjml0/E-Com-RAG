import os
import base64
import logging
import urllib.parse
from typing import Optional
from app.config import settings

logger = logging.getLogger("image_generator")

class ImageGenerationEngine:
    """
    Multimodal Image Generation Pipeline.
    Supports Google Gemini/Imagen 3 with automatic fallback to 100% Free AI Image Generation (Flux / Stable Diffusion).
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
                logger.info("Initialized Google Gemini client for image interactions.")
            except Exception as e:
                logger.warning(f"Could not initialize genai client: {e}")
                self.client = None

    def _generate_free_ai_image_url(self, prompt: str) -> str:
        """Generates a 100% Free AI Image URL using Pollinations AI (Flux / Stable Diffusion)."""
        clean_prompt = prompt[:200] if len(prompt) > 200 else prompt
        encoded_prompt = urllib.parse.quote(clean_prompt)
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"

    async def generate_image(self, prompt: str, base_image_url: Optional[str] = None) -> dict:
        """
        Generates or edits a product image using Gemini 3.1 Flash Image.
        Falls back seamlessly to 100% Free Flux/Stable Diffusion AI Image generation if Gemini free tier limit is reached.
        """
        if self.client:
            try:
                generation_config = {
                    'temperature': 1,
                    'max_output_tokens': 65536,
                    'top_p': 0.95,
                    'thinking_level': 'minimal',
                }

                user_input = prompt
                if base_image_url:
                    user_input = f"Edit product image ({base_image_url}): {prompt}"

                interaction = self.client.interactions.create(
                    model='models/gemini-3.1-flash-lite-image',
                    input=user_input,
                    generation_config=generation_config,
                    response_modalities=['image', 'text'],
                )

                extracted_b64 = None
                extracted_text = ""

                for step in getattr(interaction, "steps", []):
                    if getattr(step, "type", "") == 'model_output' and getattr(step, "content", None):
                        for part in step.content:
                            p_type = getattr(part, "type", "")
                            if p_type == 'text':
                                extracted_text += getattr(part, "text", "")
                            elif p_type == 'image':
                                raw_data = getattr(part, "data", None)
                                if raw_data:
                                    if isinstance(raw_data, bytes):
                                        extracted_b64 = base64.b64encode(raw_data).decode("utf-8")
                                    else:
                                        extracted_b64 = str(raw_data)

                if extracted_b64:
                    return {
                        "status": "success",
                        "image_url": f"data:image/png;base64,{extracted_b64}",
                        "model_used": "models/gemini-3.1-flash-lite-image",
                        "text_output": extracted_text.strip()
                    }

            except Exception as e:
                logger.info(f"Gemini image API limit/error ({e}). Using Free AI Image Generator (Flux).")

        # 100% Free AI Image Generation Fallback (Flux / Stable Diffusion)
        free_ai_url = self._generate_free_ai_image_url(prompt)
        return {
            "status": "success_free_ai",
            "image_url": free_ai_url,
            "model_used": "flux-schnell-free-ai"
        }

image_generator = ImageGenerationEngine()
