import uuid
import logging
from typing import List, Dict, Any, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models
from app.config import settings
from app.schemas import ProductIngestRequest

logger = logging.getLogger("vector_db")

class VectorDBManager:
    """Async Manager for Qdrant Named Multi-Vector Index."""

    def __init__(
        self,
        qdrant_url: str = settings.QDRANT_URL,
        api_key: str | None = settings.QDRANT_API_KEY,
        collection_name: str = settings.COLLECTION_NAME
    ):
        self.collection_name = collection_name
        if qdrant_url == ":memory:":
            self.client = AsyncQdrantClient(":memory:")
        else:
            self.client = AsyncQdrantClient(url=qdrant_url, api_key=api_key)

    async def init_collection(self, text_size: int = settings.TEXT_VECTOR_SIZE, image_size: int = settings.IMAGE_VECTOR_SIZE):
        """Creates collection with named multi-vectors and payload field indices."""
        collections_response = await self.client.get_collections()
        existing_names = [c.name for c in collections_response.collections]

        if self.collection_name not in existing_names:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with Named Multi-Vectors...")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vector": rest_models.VectorParams(
                        size=text_size,
                        distance=rest_models.Distance.COSINE,
                        hnsw_config=rest_models.HnswConfigDiff(m=16, ef_construct=128)
                    ),
                    "image_vector": rest_models.VectorParams(
                        size=image_size,
                        distance=rest_models.Distance.COSINE,
                        hnsw_config=rest_models.HnswConfigDiff(m=16, ef_construct=128)
                    )
                }
            )
            
            # Setup Payload Indices for single-stage pre-filtering
            logger.info("Setting up payload index schemas for brand, category_path, price...")
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="brand",
                field_schema=rest_models.PayloadSchemaType.KEYWORD
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category_path",
                field_schema=rest_models.PayloadSchemaType.KEYWORD
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="price",
                field_schema=rest_models.PayloadSchemaType.FLOAT
            )
            logger.info(f"Collection '{self.collection_name}' successfully initialized.")

    async def ensure_collection(self):
        """Ensures collection exists, initializing it if necessary."""
        try:
            collections_response = await self.client.get_collections()
            existing_names = [c.name for c in collections_response.collections]
            if self.collection_name not in existing_names:
                await self.init_collection()
        except Exception as e:
            logger.warning(f"Failed checking collection status ({e}). Trying init...")
            await self.init_collection()

    def _string_to_uuid(self, string_val: str) -> str:
        """Helper to derive deterministic UUID string for Qdrant point IDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_val))

    async def upsert_product(
        self,
        product: ProductIngestRequest,
        text_vector: List[float],
        image_vector: List[float]
    ):
        """Idempotent product upsert into Qdrant multi-vector store."""
        await self.ensure_collection()
        brand_clean = product.brand.strip().lower()
        category_paths = [c.strip().lower() for c in product.category.split(">")]

        
        point_id = self._string_to_uuid(product.product_id)

        payload = {
            "product_id": product.product_id,
            "prod_title": product.prod_title,
            "prod_image_url": str(product.prod_image_url),
            "price": float(product.price),
            "category": product.category,
            "category_path": category_paths,
            "brand": brand_clean
        }

        point = rest_models.PointStruct(
            id=point_id,
            vector={
                "text_vector": text_vector,
                "image_vector": image_vector
            },
            payload=payload
        )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        logger.debug(f"Upserted product {product.product_id} into Qdrant.")

    async def count_points(self) -> int:
        try:
            info = await self.client.get_collection(collection_name=self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

# Singleton DB instance
vector_db_manager = VectorDBManager()
