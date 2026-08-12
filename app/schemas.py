from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl

RAGStrategy = Literal["hybrid", "text_only", "vision_only", "price_elastic"]

# Product Ingestion Schemas
class ProductIngestRequest(BaseModel):
    product_id: str = Field(..., description="Unique product SKU or ID", json_schema_extra={"example": "SKU-HEADPHONE-001"})
    prod_title: str = Field(..., min_length=2, max_length=500, description="Product display title", json_schema_extra={"example": "Sony WH-1000XM5 Wireless Headphones"})
    prod_image_url: HttpUrl = Field(..., description="Absolute URL to product image")
    price: float = Field(..., ge=0.0, description="Product price in USD")
    category: str = Field(..., min_length=1, description="Hierarchical category (e.g., Electronics > Audio > Headphones)")
    brand: str = Field(..., min_length=1, description="Brand name", json_schema_extra={"example": "Sony"})

class BatchIngestRequest(BaseModel):
    products: List[ProductIngestRequest]

class IngestResponse(BaseModel):
    status: str = "success"
    ingested_count: int
    message: str

# 1. Recommendation API Input Schema (Takes 5 core params)
class RecommendationInput(BaseModel):
    prod_title: str = Field(..., description="Product Title", json_schema_extra={"example": "Sony WH-1000XM5"})
    prod_image_url: HttpUrl = Field(..., description="Absolute URL to product image", json_schema_extra={"example": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"})
    price: float = Field(..., ge=0.0, description="Product price or estimated price", json_schema_extra={"example": 398.00})
    category: str = Field(..., description="Category path", json_schema_extra={"example": "Electronics > Audio > Headphones"})
    brand: str = Field(..., description="Brand name", json_schema_extra={"example": "Sony"})
    query: Optional[str] = Field(None, description="Optional user search intent query")

class ProductIdentityValidationInfo(BaseModel):
    brand_match: bool = True
    model_match: bool = True
    category_match: bool = True
    generation_match: bool = True
    accepted: bool = True
    reason: str = "Verified exact product identity match"

class FactEvidenceValidation(BaseModel):
    attribute: str
    requirement_tier: str = "Required"
    normalized_value: str
    source_document_id: str
    evidence_span: str
    confidence: float = 0.98
    product_identity_validation: bool = True
    category_validation: bool = True
    generation_validation: bool = True
    verified_status: bool = True


class RetrievalDebugInfo(BaseModel):
    queries_generated: int = 0
    documents_retrieved: int = 0
    documents_after_deduplication: int = 0
    documents_after_reranking: int = 0
    identity_valid_documents: int = 0
    identity_rejected_documents: int = 0
    category_valid_documents: int = 0
    category_rejected_documents: int = 0
    generation_valid_documents: int = 0
    generation_rejected_documents: int = 0
    identity_precision: float = 1.0
    retrievable_verified_facts: int = 0
    retrieved_verified_facts: int = 0
    extracted_verified_facts: int = 0
    final_verified_facts: int = 0
    retrieval_recall: float = 1.0
    extraction_recall: float = 1.0
    final_recall: float = 1.0
    fact_precision: float = 1.0
    hallucination_rate: float = 0.0
    evidence_coverage: float = 1.0
    schema_attribute_coverage: float = 1.0
    missing_facts: List[str] = Field(default_factory=list)
    product_identity_validation: Optional[ProductIdentityValidationInfo] = None
    verified_fact_evidence: Optional[List[FactEvidenceValidation]] = None


# 1. Recommendation API Output Schema (Strictly follows pattern)
class StrictRecommendationResponse(BaseModel):
    product_description: str = Field(..., description="Generated e-commerce product description")
    estimated_price: float = Field(..., description="Estimated market price of product")
    key_features: List[str] = Field(..., description="Generated key features in bullet points")
    detected_product_specifications_and_attributes: Dict[str, Any] = Field(..., description="Detected product specifications & attributes")
    mined_high_rank_seo_keywords: List[str] = Field(..., description="Mined high-rank SEO keywords")
    best_prompt_for_image_enhancement: str = Field(..., description="Best prompt for image enhancement")
    retrieval_debug: Optional[RetrievalDebugInfo] = Field(None, description="Detailed 18-point RAG pipeline debug metrics")

# 2. Image Generation API Input Schema
class ImageGenerationInput(BaseModel):
    image_url: HttpUrl = Field(..., description="User provided base product image URL", json_schema_extra={"example": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"})
    prompt: str = Field(..., description="Prompt for image enhancement/editing", json_schema_extra={"example": "Studio product photography on polished dark wood with ambient studio lighting"})

# 2. Image Generation API Output Schema
class ImageGenerationOutput(BaseModel):
    status: str = "success"
    generated_image_url: str = Field(..., description="New product level image for e-commerce platform")
    model_used: str = "flux-schnell-free-ai"

# Multi-Vector Search & Ranking Models (for internal search engines)
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

class SearchWeights(BaseModel):
    text: float = 0.45
    image: float = 0.35
    bm25: float = 0.20

class SearchQueryRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_url: Optional[HttpUrl] = None
    brand_filter: Optional[List[str]] = None
    category_filter: Optional[str] = None
    generation_filter: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    target_price: Optional[float] = None
    rag_strategy: RAGStrategy = "hybrid"
    top_k: int = 10
    weights: Optional[SearchWeights] = None


class SearchQueryResponse(BaseModel):
    total_hits: int
    execution_time_ms: float
    rag_strategy: RAGStrategy
    results: List[ProductResponse]

class RAGQueryRequest(BaseModel):
    user_query: str
    query_image_url: Optional[HttpUrl] = None
    brand_filter: Optional[List[str]] = None
    category_filter: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    target_price: Optional[float] = None
    rag_strategy: RAGStrategy = "hybrid"
    top_k: int = 5

class RAGResponse(BaseModel):
    query: str
    recommendation: str
    retrieved_products: List[ProductResponse]
    execution_time_ms: float

class PipelineTelemetryEvent(BaseModel):
    timestamp: str
    event_type: Literal["health_update", "pipeline_stage", "moving_data", "error_event"]
    trace_id: Optional[str] = None
    details: Dict[str, Any]

# Legacy compatibility models
class SimpleRAGRequest(RecommendationInput):
    pass

class SimpleRAGResponse(StrictRecommendationResponse):
    pass

# System Health Probe Response
class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    vector_db_status: str
    total_vectors_indexed: int
    p95_latency_ms: float
    active_subscribers: int
