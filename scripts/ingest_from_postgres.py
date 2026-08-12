#!/usr/bin/env python3
"""
Secure PostgreSQL Ingestion Tool for Multimodal E-Commerce RAG Engine.

Features:
  1. SECURE: Reads database connection details from .env (DATABASE_URL or POSTGRES_*) without exposing passwords.
  2. MULTI-TABLE: Supports custom SQL JOIN queries across multiple database tables (e.g. products JOIN categories JOIN brands).

Usage:
    # Uses DATABASE_URL from .env automatically
    python scripts/ingest_from_postgres.py

    # Or specify custom SQL JOIN query for normalized multi-table database schemas
    python scripts/ingest_from_postgres.py --query "
        SELECT 
            p.id as product_id, 
            p.title as prod_title, 
            p.image_url as prod_image_url, 
            p.price, 
            c.name as category, 
            b.name as brand 
        FROM products p 
        LEFT JOIN categories c ON p.category_id = c.id 
        LEFT JOIN brands b ON p.brand_id = b.id
    "
"""

import os
import sys
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

    # Fallback to individual POSTGRES_* environment variables
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "postgres")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"

def fetch_postgres_data(query_sql: str, limit: int = 100000) -> List[Dict[str, Any]]:
    """Executes secure SQL query across single or multiple joined tables in PostgreSQL."""
    db_url = get_database_url()
    products = []
    logger.info("Executing secure PostgreSQL data fetch...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            full_query = f"{query_sql} LIMIT %s" if "LIMIT" not in query_sql.upper() else query_sql
            if "LIMIT" not in query_sql.upper():
                cur.execute(full_query, (limit,))
            else:
                cur.execute(full_query)
            rows = cur.fetchall()
            for r in rows:
                products.append({
                    "product_id": str(r.get("product_id") or r.get("id") or r.get("sku") or "SKU-PG"),
                    "prod_title": str(r.get("prod_title") or r.get("title") or r.get("name") or "Product"),
                    "prod_image_url": str(r.get("prod_image_url") or r.get("image_url") or r.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(r.get("price") or 0.0),
                    "category": str(r.get("category") or r.get("category_name") or "General"),
                    "brand": str(r.get("brand") or r.get("brand_name") or "Generic")
                })
        conn.close()
    except ImportError:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text(query_sql))
            for r in result.mappings():
                products.append({
                    "product_id": str(r.get("product_id") or r.get("id") or r.get("sku") or "SKU-PG"),
                    "prod_title": str(r.get("prod_title") or r.get("title") or r.get("name") or "Product"),
                    "prod_image_url": str(r.get("prod_image_url") or r.get("image_url") or r.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(r.get("price") or 0.0),
                    "category": str(r.get("category") or r.get("category_name") or "General"),
                    "brand": str(r.get("brand") or r.get("brand_name") or "Generic")
                })

    logger.info(f"Successfully retrieved {len(products):,} records from PostgreSQL.")
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

async def main(query_sql: str, batch_size: int = 500):
    await vector_db_manager.init_collection()
    products = fetch_postgres_data(query_sql)

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(products), batch_size):
        chunk = products[i:i + batch_size]
        await process_batch(chunk)
        total_processed += len(chunk)
        logger.info(f"Progress: {total_processed:,} / {len(products):,} products indexed into Qdrant")

    logger.info(f"✅ Secure Multi-Table Ingestion Complete! Synced {total_processed:,} products in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    default_query = "SELECT * FROM products"
    parser = argparse.ArgumentParser(description="Secure Multi-Table PostgreSQL Ingestion Tool")
    parser.add_argument("--query", type=str, default=default_query, help="SQL query or JOIN statement for multi-table schemas")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vector indexing (default: 500)")
    args = parser.parse_args()

    asyncio.run(main(args.query, args.batch_size))
