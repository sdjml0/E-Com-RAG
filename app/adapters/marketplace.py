from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.schemas import StrictRecommendationResponse

class AmazonMarketplaceListing(BaseModel):
    title: str = Field(..., description="Amazon optimized product title (Brand + Model + Key Spec)")
    bullet_points: List[str] = Field(..., max_length=5, description="Amazon 5 key feature bullet points")
    description: str = Field(..., description="Amazon product description block")
    backend_search_terms: str = Field(..., description="Amazon 250-byte backend search keywords")
    a_plus_hero_prompt: str = Field(..., description="Image prompt for Amazon A+ content creation")
    attributes: Dict[str, Any] = Field(..., description="Amazon technical specification table")

class FlipkartMarketplaceListing(BaseModel):
    title: str = Field(..., description="Flipkart listing title")
    highlights: List[str] = Field(..., description="Flipkart key highlights")
    specification_groups: Dict[str, Dict[str, Any]] = Field(..., description="Flipkart specs grouped by category")
    search_keywords: List[str] = Field(..., description="Flipkart search tags")
    enhanced_image_prompt: str = Field(..., description="Flipkart banner image prompt")

class ShopifyMarketplaceListing(BaseModel):
    title: str = Field(..., description="Shopify product title")
    body_html: str = Field(..., description="HTML formatted product description for Shopify theme")
    vendor: str = Field(..., description="Vendor / Brand name")
    tags: List[str] = Field(..., description="Shopify tags for collection filtering")
    variants: List[Dict[str, Any]] = Field(..., description="Shopify pricing & variant matrix")

class MarketplaceAdaptationResponse(BaseModel):
    amazon: AmazonMarketplaceListing
    flipkart: FlipkartMarketplaceListing
    shopify: ShopifyMarketplaceListing

class MarketplaceAdapterEngine:
    """Transforms Universal Product JSON into platform-specific marketplace schemas."""

    @staticmethod
    def adapt(universal_json: StrictRecommendationResponse, brand: str = "Brand") -> MarketplaceAdaptationResponse:
        specs = universal_json.detected_product_specifications_and_attributes or {}
        model_name = specs.get("model_name", "Product")
        
        # 1. Amazon Adapter Format
        amazon_bullets = [bp[:150] for bp in universal_json.key_features[:5]]
        backend_keywords = " ".join(universal_json.mined_high_rank_seo_keywords)[:250]
        amazon_title = f"{brand} {model_name} - {universal_json.product_description[:60]}".strip()
        
        amazon = AmazonMarketplaceListing(
            title=amazon_title,
            bullet_points=amazon_bullets,
            description=universal_json.product_description,
            backend_search_terms=backend_keywords,
            a_plus_hero_prompt=universal_json.best_prompt_for_image_enhancement,
            attributes=specs
        )

        # 2. Flipkart Adapter Format
        flipkart_groups = {
            "General": {
                "Brand": specs.get("brand", brand),
                "Model Name": specs.get("model_name", model_name),
                "Primary Color": specs.get("primary_color", "Standard"),
                "Category": specs.get("category_hierarchy", "General")
            },
            "Specifications": {
                "Material": specs.get("material_build", "Standard Build"),
                "Connectivity": specs.get("connectivity_tech", "Wireless / Standard"),
                "Usage": specs.get("intended_usage", "Daily Use")
            }
        }

        flipkart = FlipkartMarketplaceListing(
            title=f"{brand} {model_name} ({specs.get('primary_color', 'Default')})",
            highlights=universal_json.key_features[:6],
            specification_groups=flipkart_groups,
            search_keywords=universal_json.mined_high_rank_seo_keywords,
            enhanced_image_prompt=universal_json.best_prompt_for_image_enhancement
        )

        # 3. Shopify Adapter Format
        body_html = f"<h3>Product Overview</h3><p>{universal_json.product_description}</p><h4>Key Features</h4><ul>"
        for kf in universal_json.key_features:
            body_html += f"<li>{kf}</li>"
        body_html += "</ul>"

        shopify = ShopifyMarketplaceListing(
            title=f"{brand} {model_name}",
            body_html=body_html,
            vendor=brand,
            tags=[brand] + universal_json.mined_high_rank_seo_keywords[:5],
            variants=[
                {
                    "price": universal_json.estimated_price,
                    "sku": f"SKU-{brand.upper()[:3]}-001",
                    "requires_shipping": True
                }
            ]
        )

        return MarketplaceAdaptationResponse(
            amazon=amazon,
            flipkart=flipkart,
            shopify=shopify
        )

marketplace_adapter_engine = MarketplaceAdapterEngine()
