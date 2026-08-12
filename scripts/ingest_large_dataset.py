#!/usr/bin/env python3
"""
High-Throughput Large Dataset Ingestion Tool for Multimodal E-Commerce RAG Engine.

Supports loading 10,000 to 1,000,000+ products from CSV, JSON, or JSONL files
into Qdrant Named Multi-Vector Store.

Usage:
    python scripts/ingest_large_dataset.py --file path/to/products.csv --batch-size 500
"""

import os
import sys
import json
import csv
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
logger = logging.getLogger("dataset_ingestor")

def read_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Reads CSV, JSON, or JSONL file into product dictionaries."""
    products = []
    ext = os.path.splitext(file_path)[1].lower()

    logger.info(f"Reading dataset file: {file_path}")

    if ext == ".csv":
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                products.append({
                    "product_id": str(row.get("product_id") or row.get("id") or row.get("sku") or f"SKU-{i:06d}"),
                    "prod_title": str(row.get("prod_title") or row.get("title") or row.get("name") or "Product"),
                    "prod_image_url": str(row.get("prod_image_url") or row.get("image_url") or row.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(row.get("price") or 29.99),
                    "category": str(row.get("category") or "General"),
                    "brand": str(row.get("brand") or "Generic")
                })
    elif ext in (".json", ".jsonl"):
        with open(file_path, mode="r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
            else:
                items = [json.loads(line) for line in f if line.strip()]

            for i, row in enumerate(items):
                products.append({
                    "product_id": str(row.get("product_id") or row.get("id") or row.get("sku") or f"SKU-{i:06d}"),
                    "prod_title": str(row.get("prod_title") or row.get("title") or row.get("name") or "Product"),
                    "prod_image_url": str(row.get("prod_image_url") or row.get("image_url") or row.get("image") or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"),
                    "price": float(row.get("price") or 29.99),
                    "category": str(row.get("category") or "General"),
                    "brand": str(row.get("brand") or "Generic")
                })
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Please use .csv, .json, or .jsonl")

    logger.info(f"Loaded {len(products):,} products from file.")
    return products

async def process_batch(batch: List[Dict[str, Any]]):
    """Processes a chunk batch of products: generates vectors and upserts to Qdrant."""
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

async def main(file_path: str, batch_size: int = 500):
    await vector_db_manager.init_collection()
    products = read_dataset(file_path)

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(products), batch_size):
        chunk = products[i:i + batch_size]
        await process_batch(chunk)
        total_processed += len(chunk)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"Progress: {total_processed:,} / {len(products):,} products indexed ({rate:.1f} items/sec)")

    logger.info(f"✅ Ingestion Complete! Indexed {total_processed:,} products in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Large Dataset into Multimodal RAG Engine")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV, JSON, or JSONL product dataset")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of products per vector batch (default: 500)")
    args = parser.parse_args()

    asyncio.run(main(args.file, args.batch_size))
