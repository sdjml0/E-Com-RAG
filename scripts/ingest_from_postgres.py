#!/usr/bin/env python3
"""
PostgreSQL Table Ingestion Tool for Multimodal E-Commerce RAG Engine.

Connects directly to your PostgreSQL database, streams rows from your products table,
generates dual multi-vectors, and syncs them into Qdrant.

Usage:
    python scripts/ingest_from_postgres.py --db-url "postgresql://user:password@localhost:5432/dbname" --table products
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

from app.schemas import ProductIngestRequest
from app.embeddings.text_embedder import text_embedder
from app.embeddings.vision_embedder import vision_embedder
from app.db.vector_db import vector_db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("postgres_ingestor")

def fetch_postgres_products(db_url: str, table_name: str, limit: int = 10000) -> List[Dict[str, Any]]:
    """Fetches product records directly from a PostgreSQL table using psycopg2 / asyncpg / sqlalchemy."""
    products = []
    logger.info(f"Connecting to PostgreSQL database to read table '{table_name}'...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            query = f"SELECT * FROM {table_name} LIMIT %s"
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            for r in rows:
                products.append({
                    "product_id": str(r.get("product_id") or r.get("id") or r.get("sku") or "SKU-PG"),
                    "prod_title": str(r.get("prod_title") or r.get("title") or r.get("name") or "Product"),
                    "prod_image_url": str(r.get("prod_image_url") or r.get("image_url") or r.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(r.get("price") or 29.99),
                    "category": str(r.get("category") or "General"),
                    "brand": str(r.get("brand") or "Generic")
                })
        conn.close()
    except ImportError:
        logger.info("psycopg2 not installed. Trying sqlalchemy...")
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
            for r in result.mappings():
                products.append({
                    "product_id": str(r.get("product_id") or r.get("id") or r.get("sku") or "SKU-PG"),
                    "prod_title": str(r.get("prod_title") or r.get("title") or r.get("name") or "Product"),
                    "prod_image_url": str(r.get("prod_image_url") or r.get("image_url") or r.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(r.get("price") or 29.99),
                    "category": str(r.get("category") or "General"),
                    "brand": str(r.get("brand") or "Generic")
                })

    logger.info(f"Successfully fetched {len(products):,} products from PostgreSQL table '{table_name}'.")
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

async def main(db_url: str, table: str, batch_size: int = 500):
    await vector_db_manager.init_collection()
    products = fetch_postgres_products(db_url, table)

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(products), batch_size):
        chunk = products[i:i + batch_size]
        await process_batch(chunk)
        total_processed += len(chunk)
        logger.info(f"Progress: {total_processed:,} / {len(products):,} products indexed from PostgreSQL")

    logger.info(f"✅ PostgreSQL Ingestion Complete! Synced {total_processed:,} products in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PostgreSQL Products Table into Multimodal RAG Engine")
    parser.add_argument("--db-url", type=str, required=True, help="PostgreSQL connection string (e.g. postgresql://user:pass@localhost:5432/dbname)")
    parser.add_argument("--table", type=str, default="products", help="Name of products table (default: products)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vector indexing (default: 500)")
    args = parser.parse_args()

    asyncio.run(main(args.db_url, args.table, args.batch_size))
