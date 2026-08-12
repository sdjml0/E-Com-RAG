# 📋 Canonical JSON Specifications (Strict API Pattern)

## 1. Recommendation API (`POST /api/v1/rag/recommend` and `POST /`)

### Request JSON (Takes 5 Core Parameters)
```json
{
  "prod_title": "Sony WH-1000XM5",
  "prod_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
  "price": 398.00,
  "category": "Electronics > Audio > Headphones",
  "brand": "Sony"
}
```

### Response JSON (Strict Output Pattern)
```json
{
  "product_description": "Experience superior craftsmanship with the Sony WH-1000XM5 by Sony. Specially engineered for optimal performance in Electronics > Audio > Headphones, it combines sleek design with cutting-edge technology to deliver an unmatched user experience.",
  "estimated_price": 398.00,
  "key_features": [
    "Premium ergonomic design engineered by Sony",
    "Optimized performance tailored for Electronics > Audio > Headphones",
    "High-durability build quality with precision finish",
    "Seamless compatibility and long-lasting energy efficiency"
  ],
  "detected_product_specifications_and_attributes": {
    "brand": "Sony",
    "model_name": "Sony WH-1000XM5",
    "category_hierarchy": "Electronics > Audio > Headphones",
    "primary_color": "Matte Black / Platinum",
    "material_build": "Reinforced Composite Alloy",
    "connectivity_tech": "Wireless Bluetooth 5.3 & USB-C Fast Charge",
    "intended_usage": "Travel, Daily Use, and Professional Audio/Electronics"
  },
  "mined_high_rank_seo_keywords": [
    "sony sony wh-1000xm5",
    "best electronics audio headphones 2026",
    "buy sony online",
    "sony wh-1000xm5 price and features",
    "top rated electronics audio headphones"
  ],
  "best_prompt_for_image_enhancement": "Studio product photography of Sony WH-1000XM5 by Sony, rendered in clean commercial e-commerce aesthetic, soft studio lighting, minimalist background, high-detail texture, 8k resolution, photorealistic."
}
```

---

## 2. Generate Image API (`POST /api/v1/image/generate`)

### Request JSON
```json
{
  "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
  "prompt": "Studio product photography on dark polished wood with ambient neon studio lighting"
}
```

### Response JSON
```json
{
  "status": "success",
  "generated_image_url": "https://image.pollinations.ai/prompt/E-Commerce%20product%20photography%3A%20Studio%20product%20photography%20on%20dark%20polished%20wood?width=512&height=512&nologo=true",
  "model_used": "flux-schnell-free-ai"
}
```
