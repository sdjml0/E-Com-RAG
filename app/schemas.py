from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl

# RAG Strategy Types
RAGStrategy = Literal["hybrid", "text_only", "vision_only", "price_elastic"]

# Product Ingestion Schemas
class ProductIngestRequest(BaseModel):
    product_id: str = Field(..., description="Unique product SKU or ID", json_schema_extra={"example": "SKU-HEADPHONE-001"})
    prod_title: str = Field(..., min_length=2, max_length=500, description="Product display title", json_schema_extra={"example": "Sony WH-1000XM5 Wireless Headphones"})
    prod_image_url: HttpUrl = Field(..., description="Absolute URL to product image")
    price: float = Field(..., gt=0.0, description="Product price in USD")
    category: str = Field(..., min_length=1, description="Hierarchical category (e.g., Electronics > Audio > Headphones)")
    brand: str = Field(..., min_length=1, description="Brand name", json_schema_extra={"example": "Sony"})


class BatchIngestRequest(BaseModel):
    products: List[ProductIngestRequest]

class IngestResponse(BaseModel):
    status: str = "success"
    ingested_count: int
    message: str

# Weights schema for custom rank fusion
class SearchWeights(BaseModel):
    text: float = Field(0.45, ge=0.0, le=1.0)
    image: float = Field(0.35, ge=0.0, le=1.0)
    bm25: float = Field(0.20, ge=0.0, le=1.0)

# Multi-Vector Search Query Request
class SearchQueryRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Natural language search term")
    query_image_url: Optional[HttpUrl] = Field(None, description="Image URL for visual similarity search")
    brand_filter: Optional[List[str]] = Field(None, description="List of brands for hard pre-filtering")
    category_filter: Optional[str] = Field(None, description="Category name or path for pre-filtering")
    min_price: Optional[float] = Field(None, ge=0.0, description="Minimum price boundary")
    max_price: Optional[float] = Field(None, ge=0.0, description="Maximum price boundary")
    target_price: Optional[float] = Field(None, ge=0.0, description="Ideal budget for soft elasticity dampening")
    rag_strategy: RAGStrategy = Field("hybrid", description="Retrieval strategy mode")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of candidate results to return")
    weights: Optional[SearchWeights] = None

# Individual Product Search Score & Breakdown
class ScoreBreakdown(BaseModel):
    text_rank: Optional[int] = None
    visual_rank: Optional[int] = None
    rrf_score: float
    price_penalty: float = 1.0

class ProductResponse(BaseModel):
    product_id: str
    prod_title: str
    prod_image_url: str
    price: float
    category: str
    brand: str
    final_score: float
    score_breakdown: ScoreBreakdown

class SearchQueryResponse(BaseModel):
    total_hits: int
    execution_time_ms: float
    rag_strategy: RAGStrategy
    results: List[ProductResponse]

# RAG Multimodal LLM Generation Request & Response
class RAGQueryRequest(BaseModel):
    user_query: str = Field(..., min_length=2, description="User search or query prompt")
    query_image_url: Optional[HttpUrl] = None
    brand_filter: Optional[List[str]] = None
    category_filter: Optional[str] = None
    min_price: Optional[float] = Field(None, ge=0.0)
    max_price: Optional[float] = Field(None, ge=0.0)
    target_price: Optional[float] = Field(None, ge=0.0)
    rag_strategy: RAGStrategy = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)

class RAGResponse(BaseModel):
    query: str
    recommendation: str
    retrieved_products: List[ProductResponse]
    execution_time_ms: float

# Telemetry & Event Stream Schemas
class PipelineTelemetryEvent(BaseModel):
    timestamp: str
    event_type: Literal["health_update", "pipeline_stage", "moving_data", "error_event"]
    trace_id: Optional[str] = None
    details: Dict[str, Any]

# Simplified Single-Response RAG Request & Response Schemas
class SimpleRAGRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language shopping intent or image edit request", json_schema_extra={"example": "Sleek black wireless noise-canceling headphones for travel under $400"})
    prod_title: Optional[str] = Field(None, description="Product title reference", json_schema_extra={"example": "Sony WH-1000XM5"})
    prod_image_url: Optional[HttpUrl] = Field(None, description="Current product image URL")
    price: Optional[float] = Field(None, ge=0.0, description="Product target price")
    category: Optional[str] = Field(None, description="Category path", json_schema_extra={"example": "Electronics > Audio > Headphones"})
    brand: Optional[str] = Field(None, description="Brand name", json_schema_extra={"example": "Sony"})

class SimpleProductDetail(BaseModel):
    product_id: str
    title: str
    brand: str
    category: str
    price: float
    image_url: str
    match_score: float
    reasoning: str

class ImageGenPrompt(BaseModel):
    prompt: str = Field(..., description="Prompt generated for Next-Gen Image AI Model (Imagen 3, Flux, Midjourney)")
    action: str = Field("generate_or_edit", description="Action hint: generate new, edit background, or enhance visual style")
    base_image_url: Optional[str] = None
    style_modifiers: List[str] = []
    aspect_ratio: str = "1:1"

class SimpleRAGResponse(BaseModel):
    query: str
    product_details: SimpleProductDetail
    image_generation_prompt: ImageGenPrompt
    generated_image_url: Optional[str] = Field(None, description="Direct URL / Data-URI of image produced by Imagen 3 pipeline")

# Dedicated Standalone Image Generation Schemas
class ImageGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Image generation or editing prompt", json_schema_extra={"example": "Studio photography of matte black headphones on polished dark wood with ambient neon accent lighting"})
    base_image_url: Optional[HttpUrl] = Field(None, description="Existing product image URL to edit, enhance, or use as reference")
    product_title: Optional[str] = Field(None, description="Product title reference", json_schema_extra={"example": "Sony WH-1000XM5"})
    brand: Optional[str] = Field(None, description="Brand visual aesthetic reference", json_schema_extra={"example": "Sony"})
    style_modifiers: Optional[List[str]] = Field(default_factory=lambda: ["soft studio lighting", "clean minimalist backdrop", "8k ultra-detailed"])
    aspect_ratio: str = Field("1:1", description="Target aspect ratio: 1:1, 16:9, 4:3, 9:16")

class ImageGenerateResponse(BaseModel):
    status: str
    prompt_used: str
    generated_image_url: str
    model_used: str

# System Health Probe Response



# System Health Probe Response
class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    vector_db_status: str
    total_vectors_indexed: int
    p95_latency_ms: float
    active_subscribers: int


