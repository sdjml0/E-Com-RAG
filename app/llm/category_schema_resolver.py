import logging
from typing import List, Dict, Set, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger("category_schema_resolver")

class CategorySchemaResolution(BaseModel):
    category_path: str
    primary_domain: str
    required_attributes: List[str]
    recommended_attributes: List[str]
    optional_attributes: List[str]
    non_applicable_attributes: List[str]

class CategorySchemaResolver:
    """
    Universal E-Commerce Category & Attribute Taxonomy Resolver Engine.
    Dynamically determines required, recommended, optional, and non-applicable attributes
    for any arbitrary product category (Apparel, Electronics, Furniture, Automotive, Beauty, etc.)
    without hardcoding category attribute lists in the core retrieval pipeline.
    """

    DOMAINS: Dict[str, Dict[str, List[str]]] = {
        "apparel": {
            "required": ["brand", "model", "category", "size", "gender", "color", "materials"],
            "recommended": ["fit", "care_instructions", "pattern", "closure_type", "origin"],
            "optional": ["neckline", "sleeve_length", "season", "included_accessories"],
            "non_applicable": ["processor", "ram", "storage", "battery_life", "charging", "display", "operating_system", "fuel_type", "part_number"]
        },
        "electronics": {
            "required": ["brand", "model", "category", "color", "dimensions", "weight"],
            "recommended": ["display", "processor", "ram", "storage", "camera", "battery_life", "charging", "connectivity", "operating_system"],
            "optional": ["materials", "noise_cancellation", "audio_features", "microphones", "compatibility", "included_accessories"],
            "non_applicable": ["fit", "care_instructions", "size", "gender", "flammable", "expiration_date", "part_number", "fitment"]
        },
        "appliances": {
            "required": ["brand", "model", "category", "dimensions", "weight", "color", "capacity"],
            "recommended": ["power_consumption", "energy_rating", "voltage", "finish_type", "control_type"],
            "optional": ["noise_level", "installation_type", "warranty", "included_accessories"],
            "non_applicable": ["processor", "ram", "storage", "gender", "fit", "care_instructions", "expiration_date"]
        },
        "furniture": {
            "required": ["brand", "model", "category", "dimensions", "weight", "color", "materials"],
            "recommended": ["weight_capacity", "assembly_required", "finish_type", "style", "room_type"],
            "optional": ["care_instructions", "upholstery_material", "frame_material", "warranty"],
            "non_applicable": ["processor", "ram", "storage", "battery_life", "charging", "operating_system", "camera", "connectivity"]
        },
        "automotive": {
            "required": ["brand", "model", "category", "part_number", "fitment", "materials"],
            "recommended": ["oem_compatibility", "vehicle_position", "dimensions", "weight", "warranty"],
            "optional": ["finish_type", "operating_temperature", "included_accessories"],
            "non_applicable": ["processor", "ram", "storage", "battery_life", "gender", "care_instructions", "expiration_date"]
        },
        "beauty": {
            "required": ["brand", "model", "category", "volume", "skin_type", "formulation"],
            "recommended": ["key_ingredients", "benefits", "fragrance_free", "cruelty_free", "sun_protection"],
            "optional": ["shade_color", "application_area", "expiration_date"],
            "non_applicable": ["processor", "ram", "storage", "battery_life", "dimensions", "weight_capacity", "fitment"]
        },
        "groceries": {
            "required": ["brand", "model", "category", "weight", "package_quantity", "dietary_type"],
            "recommended": ["ingredients", "allergens", "organic", "expiration_date", "storage_instructions"],
            "optional": ["serving_size", "calories", "origin"],
            "non_applicable": ["processor", "ram", "storage", "battery_life", "display", "fitment", "care_instructions"]
        },
        "sports": {
            "required": ["brand", "model", "category", "sport_type", "materials", "weight"],
            "recommended": ["size", "color", "skill_level", "durability_rating", "dimensions"],
            "optional": ["included_accessories", "warranty"],
            "non_applicable": ["processor", "ram", "storage", "operating_system", "expiration_date"]
        },
        "tools": {
            "required": ["brand", "model", "category", "voltage", "power_source", "weight"],
            "recommended": ["dimensions", "materials", "included_accessories", "warranty", "speed_rpm"],
            "optional": ["finish_type", "operating_type"],
            "non_applicable": ["skin_type", "expiration_date", "dietary_type"]
        },
        "general": {
            "required": ["brand", "model", "category", "color", "materials", "dimensions", "weight"],
            "recommended": ["features", "warranty", "included_accessories"],
            "optional": ["origin", "finish_type"],
            "non_applicable": []
        }
    }

    ATTRIBUTE_SYNONYMS: Dict[str, List[str]] = {
        "weight": ["weight", "item weight", "product weight", "net weight", "unit weight", "mass"],
        "dimensions": ["dimensions", "measurements", "product dimensions", "item dimensions", "size", "height width depth"],
        "materials": ["material", "materials", "construction", "composition", "fabric", "build"],
        "included_accessories": ["included", "package contents", "box contents", "what's included", "accessories", "in the box"],
        "part_number": ["part number", "MPN", "OEM part number", "SKU", "model number", "catalog number"],
        "fitment": ["fitment", "vehicle compatibility", "fits", "compatibility", "application"],
        "display": ["display", "screen", "panel", "resolution", "display size", "screen size"],
        "battery_life": ["battery life", "battery", "runtime", "endurance", "battery capacity", "playtime"],
        "care_instructions": ["care instructions", "washing instructions", "care", "cleaning instructions", "maintenance"],
        "key_ingredients": ["ingredients", "key ingredients", "active ingredients", "formula", "formulation"]
    }

    @classmethod
    def get_attribute_synonyms(cls, attribute: str) -> List[str]:
        attr_lower = attribute.lower()
        return cls.ATTRIBUTE_SYNONYMS.get(attr_lower, [attribute])

    @classmethod
    def resolve_schema(cls, category_path: str, product_title: str = "") -> CategorySchemaResolution:
        """
        Dynamically resolves taxonomy and attribute requirements based on category path and product title.
        """
        cat_lower = f"{category_path.lower()} {product_title.lower()}"

        primary_domain = "general"
        if any(k in cat_lower for k in ["apparel", "clothing", "dress", "shirt", "pant", "jeans", "shoe", "footwear", "jacket"]):
            primary_domain = "apparel"
        elif any(k in cat_lower for k in ["electronic", "phone", "mobile", "laptop", "audio", "headphone", "earbud", "tv", "camera", "tablet", "computer", "console", "switch", "vacuum"]):
            primary_domain = "electronics"
        elif any(k in cat_lower for k in ["appliance", "refrigerator", "washer", "dryer", "microwave", "oven"]):
            primary_domain = "appliances"
        elif any(k in cat_lower for k in ["furniture", "chair", "table", "desk", "sofa", "bed", "cabinet", "shelf", "kallax"]):
            primary_domain = "furniture"
        elif any(k in cat_lower for k in ["automotive", "auto", "car", "wiper", "brake", "engine", "oil filter", "tire"]):
            primary_domain = "automotive"
        elif any(k in cat_lower for k in ["beauty", "skincare", "cleanser", "serum", "moisturizer", "makeup", "lotion", "cosmetic"]):
            primary_domain = "beauty"
        elif any(k in cat_lower for k in ["grocery", "groceries", "food", "olive oil", "beverage", "snack", "coffee", "tea"]):
            primary_domain = "groceries"
        elif any(k in cat_lower for k in ["sports", "fitness", "exercise", "racket", "ball", "gym", "outdoor"]):
            primary_domain = "sports"
        elif any(k in cat_lower for k in ["tool", "drill", "dewalt", "saw", "wrench"]):
            primary_domain = "tools"

        domain_info = cls.DOMAINS[primary_domain]

        return CategorySchemaResolution(
            category_path=category_path,
            primary_domain=primary_domain,
            required_attributes=domain_info["required"],
            recommended_attributes=domain_info["recommended"],
            optional_attributes=domain_info["optional"],
            non_applicable_attributes=domain_info["non_applicable"]
        )

category_schema_resolver = CategorySchemaResolver()
