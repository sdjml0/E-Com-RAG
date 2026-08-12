#!/usr/bin/env python3
"""
Smart PostgreSQL Schema Inspector & Automated Ingestion Tool.

Features:
  1. AUTOMATIC COLUMN DETECTION: You don't need to know column names! Automatically detects
     product_id, title, image_url, price, category, and brand columns using fuzzy matching.
  2. TABLE INSPECTION: Run `python scripts/ingest_from_postgres.py --inspect` to inspect all
     tables and column names in your PostgreSQL database.
  3. SECURE: Reads credentials from .env without exposing passwords.

Usage:
    # Inspect database tables and column names
    python scripts/ingest_from_postgres.py --inspect

    # Ingest Amazon and Flipkart tables automatically
    python scripts/ingest_from_postgres.py --preset ecom_all
"""

import os
import sys
import re
import argparse
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.schemas import ProductIngestRequest
from app.embeddings.text_embedder import text_embedder
from app.embeddings.vision_embedder import vision_embedder
from app.db.vector_db import vector_db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_postgres_ingestor")

def get_database_url() -> str:
    """Reads database URL securely from environment variables without command line exposure."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "postgres")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"

def parse_price_value(val: Any) -> float:
    """Cleans currency strings like '$398.00', '₹2,999', or None into a clean float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    clean = re.sub(r"[^\d.]", "", s)
    try:
        return float(clean) if clean else 0.0
    except ValueError:
        return 0.0

def inspect_database():
    """Inspects all tables and column names in the PostgreSQL database."""
    db_url = get_database_url()
    logger.info("🔍 Inspecting PostgreSQL Database Schema...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            rows = cur.fetchall()
            
            tables = {}
            for r in rows:
                t = r["table_name"]
                if t not in tables:
                    tables[t] = []
                tables[t].append(f"{r['column_name']} ({r['data_type']})")

            print("\n" + "="*70)
            print("📊 POSTGRESQL TABLES AND COLUMN NAMES FOUND:")
            print("="*70)
            for t_name, cols in tables.items():
                print(f"\n📌 TABLE: {t_name}")
                print("   COLUMNS:")
                for c in cols:
                    print(f"     - {c}")
            print("="*70 + "\n")
        conn.close()
    except Exception as e:
        logger.error(f"Error inspecting database: {e}")

def detect_smart_columns(sample_row: Dict[str, Any]) -> Dict[str, str]:
    """Smart fuzzy detection of column names from any unknown database table."""
    keys = list(sample_row.keys())
    mapping = {}

    def find_best_key(patterns: List[str]) -> str | None:
        for p in patterns:
            for k in keys:
                if p.lower() in k.lower():
                    return k
        return None

    mapping["id"] = find_best_key(["uniq_id", "product_id", "id", "sku", "code", "index", "key"]) or keys[0]
    mapping["title"] = find_best_key(["product_name", "prod_title", "title", "name", "description", "item"]) or keys[1]
    mapping["image"] = find_best_key(["img_link", "image_url", "prod_image_url", "image", "img", "photo", "picture", "url", "link", "src"])
    mapping["price"] = find_best_key(["discounted_price", "actual_price", "retail_price", "price", "cost", "amount", "rate", "value", "mrp"])
    mapping["category"] = find_best_key(["product_category_tree", "category_path", "category", "tree", "class", "dept", "type"])
    mapping["brand"] = find_best_key(["brand", "manufacturer", "make", "vendor", "seller", "company"])

    return mapping

def fetch_and_auto_map_table(db_url: str, table_name: str, limit: int = 100000) -> List[Dict[str, Any]]:
    """Fetches rows and automatically maps columns to standard RAG fields."""
    products = []
    logger.info(f"Auto-detecting schema and streaming records from '{table_name}'...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            # Handle tables with hyphens or special chars by wrapping in double quotes
            safe_table = f'"{table_name}"' if "-" in table_name or " " in table_name else table_name
            cur.execute(f"SELECT * FROM {safe_table} LIMIT %s", (limit,))
            rows = cur.fetchall()

            if not rows:
                logger.warning(f"Table '{table_name}' is empty.")
                return []

            # Auto-detect column mapping on the first row
            cmap = detect_smart_columns(rows[0])
            logger.info(f"Smart Column Map for '{table_name}': {cmap}")

            for i, r in enumerate(rows):
                p_id = str(r.get(cmap["id"]) or f"SKU-{i:06d}")
                title = str(r.get(cmap["title"]) or "Product")
                
                # Image URL detection & cleaning
                img_val = str(r.get(cmap["image"]) or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e") if cmap["image"] else "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"
                if img_val.startswith("[") and "http" in img_val:
                    urls = re.findall(r"https?://[^\s\"'\]]+", img_val)
                    if urls:
                        img_val = urls[0]
                if not img_val.startswith("http"):
                    img_val = "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"

                # Price parsing
                price_val = parse_price_value(r.get(cmap["price"])) if cmap["price"] else 0.0

                # Category cleaning
                cat_val = str(r.get(cmap["category"]) or "General") if cmap["category"] else "General"
                cat_val = cat_val.replace("[", "").replace("]", "").replace('"', '').replace(">>", ">").strip()

                # Brand detection
                brand_val = str(r.get(cmap["brand"]) or title.split()[0] if title else "Generic") if cmap["brand"] else (title.split()[0] if title else "Generic")

                products.append({
                    "product_id": p_id,
                    "prod_title": title[:300],
                    "prod_image_url": img_val,
                    "price": price_val,
                    "category": cat_val[:200],
                    "brand": brand_val[:100]
                })
        conn.close()
    except Exception as e:
        logger.warning(f"Error reading table '{table_name}': {e}")

    logger.info(f"Successfully processed {len(products):,} clean product items from '{table_name}'.")
    return products

async def process_batch(batch: List[Dict[str, Any]]):
    ingest_requests = []
    text_vectors = []
    image_vectors = []

    for item in batch:
        try:
            req = ProductIngestRequest(
                product_id=item["product_id"],
                prod_title=item["prod_title"],
                prod_image_url=item["prod_image_url"],
                price=item["price"],
                category=item["category"],
                brand=item["brand"]
            )
            ingest_requests.append(req)

            composite_text = f"Brand: {req.brand} | Title: {req.prod_title} | Category: {req.category}"
            t_vec = await text_embedder.embed_text(composite_text)
            i_vec = await vision_embedder.embed_image_url(str(req.prod_image_url))

            text_vectors.append(t_vec)
            image_vectors.append(i_vec)
        except Exception as e:
            logger.warning(f"Error processing item {item.get('product_id')}: {e}")

    if ingest_requests:
        await vector_db_manager.upsert_batch(ingest_requests, text_vectors, image_vectors)

async def main(preset: str, inspect_mode: bool, batch_size: int = 500):
    if inspect_mode:
        inspect_database()
        return

    await vector_db_manager.init_collection()
    db_url = get_database_url()
    all_products = []

    if preset in ("amazon", "ecom_all"):
        all_products.extend(fetch_and_auto_map_table(db_url, "amazon_sales_data_uncleaned"))

    if preset in ("flipkart", "ecom_all"):
        all_products.extend(fetch_and_auto_map_table(db_url, "flipkart_com-ecommerce_sample"))

    if not all_products:
        logger.warning("No records found. Run `python scripts/ingest_from_postgres.py --inspect` to list tables.")
        return

    logger.info(f"Total unified products ready to index into Qdrant: {len(all_products):,}")

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(all_products), batch_size):
        chunk = all_products[i:i + batch_size]
        await process_batch(chunk)
        total_processed += len(chunk)
        logger.info(f"Indexed {total_processed:,} / {len(all_products):,} products into Qdrant...")

    logger.info(f"✅ Auto-Detection Ingestion Complete! Synced {total_processed:,} products in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart PostgreSQL Schema Inspector & Automated Ingestor")
    parser.add_argument("--inspect", action="store_true", help="Inspect and print all database tables and column names")
    parser.add_argument("--preset", type=str, default="ecom_all", choices=["ecom_all", "amazon", "flipkart"], help="Preset tables to ingest (default: ecom_all)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vector indexing (default: 500)")
    args = parser.parse_args()

    asyncio.run(main(args.preset, args.inspect, args.batch_size))
