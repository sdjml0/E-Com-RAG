from abc import ABC, abstractmethod
from typing import List

class BaseTextEmbedder(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate a dense floating-point vector for input text."""
        pass

    @abstractmethod
    async def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vectors for a batch of text strings."""
        pass

class BaseVisionEmbedder(ABC):
    @abstractmethod
    async def embed_image_url(self, image_url: str) -> List[float]:
        """Fetch image from URL and generate dense visual vector."""
        pass

    @abstractmethod
    async def embed_image_bytes(self, image_bytes: bytes) -> List[float]:
        """Generate dense visual vector from raw image bytes."""
        pass
