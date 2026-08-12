import numpy as np
import hashlib
import logging
from typing import List
from app.embeddings.base import BaseTextEmbedder
from app.config import settings

logger = logging.getLogger("text_embedder")

class TextEmbedder(BaseTextEmbedder):
    """Dense Text Embedding Engine (bge-m3 / all-MiniLM-L6-v2 / Fast fallback)."""

    def __init__(self, dimension: int = settings.TEXT_VECTOR_SIZE):
        self.dimension = dimension
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load lightweight model for fast local inference
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer model with dimension {self.dimension}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using lightweight fast deterministic text embedder.")
            self._model = None

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic pseudo-vector generator based on text sha256 seed."""
        hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(hash_digest[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            text = "empty query"
        if self._model:
            try:
                embedding = self._model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error during model text encoding: {e}. Falling back.")
        return self._fallback_embed(text)

    async def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        if self._model:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.error(f"Batch text encoding failed: {e}. Falling back.")
        return [self._fallback_embed(t) for t in texts]

# Singleton instance
text_embedder = TextEmbedder()
