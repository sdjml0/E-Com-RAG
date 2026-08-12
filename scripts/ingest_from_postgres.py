#!/usr/bin/env python3
"""
Secure PostgreSQL Ingestion Tool for Multimodal E-Commerce RAG Engine.

Pre-configured for Amazon & Flipkart Datasets:
  - Table 1: amazon_sales_data_uncleaned
  - Table 2: "flipkart_com-ecommerce_sample"

Usage:
    # Run automatic ingestion for Amazon & Flipkart tables (credentials read from .env)
    python scripts/ingest_from_postgres.py --preset ecom_all
"""

import os
import sys
import re
import argparse
import asyncio
import logging
import time
from typing import List, Dict, Any

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.schemas import ProductIngestRequest
from app.embeddings.text_embedder import text_embedder
from app.embeddings.vision_embedder import vision_embedder
from app.db.vector_db import vector_db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("postgres_ingestor")

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

def fetch_table_records(db_url: str, table_name: str, query_sql: str) -> List[Dict[str, Any]]:
    """Executes SQL query for a specific dataset table."""
    products = []
    logger.info(f"Querying PostgreSQL table '{table_name}'...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute(query_sql)
            rows = cur.fetchall()
            for r in rows:
                p_id = str(r.get("product_id") or r.get("uniq_id") or r.get("id") or f"SKU-{len(products):06d}")
                title = str(r.get("prod_title") or r.get("product_name") or r.get("title") or "Product")
                img = str(r.get("prod_image_url") or r.get("img_link") or r.get("image") or r.get("image_url") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e")
                
                # Parse image array strings if stored as ["http://..."]
                if img.startswith("[") and "http" in img:
                    urls = re.findall(r"https?://[^\s\"'\]]+", img)
                    if urls:
                        img = urls[0]

                raw_price = r.get("price") or r.get("discounted_price") or r.get("actual_price") or r.get("retail_price")
                price = parse_price_value(raw_price)

                cat = str(r.get("category") or r.get("product_category_tree") or "General")
                if cat.startswith("["):
                    cat = cat.replace("[", "").replace("]", "").replace('"', '').replace(">>", ">")

                brand = str(r.get("brand") or title.split()[0] if title else "Generic")

                products.append({
                    "product_id": p_id,
                    "prod_title": title[:300],
                    "prod_image_url": img if img.startswith("http") else "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
                    "price": price,
                    "category": cat[:200],
                    "brand": brand[:100]
                })
        conn.close()
    except Exception as e:
        logger.warning(f"Error fetching from table '{table_name}': {e}")

    logger.info(f"Retrieved {len(products):,} clean product records from table '{table_name}'.")
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

async def main(preset: str, custom_query: str = None, batch_size: int = 500):
    await vector_db_manager.init_collection()
    db_url = get_database_url()
    all_products = []

    if preset in ("amazon", "ecom_all"):
        q_amazon = "SELECT * FROM amazon_sales_data_uncleaned"
        all_products.extend(fetch_table_records(db_url, "amazon_sales_data_uncleaned", q_amazon))

    if preset in ("flipkart", "ecom_all"):
        q_flipkart = 'SELECT * FROM "flipkart_com-ecommerce_sample"'
        all_products.extend(fetch_table_records(db_url, "flipkart_com-ecommerce_sample", q_flipkart))

    if custom_query:
        all_products.extend(fetch_table_records(db_url, "custom_query", custom_query))

    if not all_products:
        logger.warning("No records found to ingest. Check PostgreSQL connection or table names.")
        return

    logger.info(f"Total unified products to index across tables: {len(all_products):,}")

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(all_products), batch_size):
        chunk = all_products[i:i + batch_size]
        await process_batch(chunk)
        total_processed += len(chunk)
        logger.info(f"Indexed {total_processed:,} / {len(all_products):,} products into Qdrant...")

    logger.info(f"✅ Ingestion Complete! Synced {total_processed:,} products across tables in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Amazon & Flipkart PostgreSQL Tables into RAG Engine")
    parser.add_argument("--preset", type=str, default="ecom_all", choices=["ecom_all", "amazon", "flipkart", "custom"], help="Table preset to ingest (default: ecom_all)")
    parser.add_argument("--query", type=str, default=None, help="Custom SQL query if using preset=custom")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vector indexing (default: 500)")
    args = parser.parse_args()

    asyncio.run(main(args.preset, args.query, args.batch_size))
