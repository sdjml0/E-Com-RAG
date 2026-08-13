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

import re

def _clean_price(val: Any) -> float:
    if not val:
        return 29.99
    val_str = str(val).replace("$", "").replace(",", "").strip()
    try:
        f = float(val_str)
        return f if f >= 0 else 29.99
    except ValueError:
        match = re.search(r"(\d+(?:\.\d+)?)", val_str)
        if match:
            return float(match.group(1))
        return 29.99

def _clean_image_url(val: Any) -> str:
    if not val:
        return "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"
    val_str = str(val).strip()
    if val_str.startswith("["):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            pass
        match = re.search(r"https?://[^\s\"'\]]+", val_str)
        if match:
            return match.group(0)
    if val_str.startswith("http://") or val_str.startswith("https://"):
        return val_str
    return "https://images.unsplash.com/photo-1505740420928-5e560c06d30e"

def _clean_category(val: Any) -> str:
    if not val:
        return "General"
    val_str = str(val).strip()
    if val_str.startswith("["):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            pass
        val_str = re.sub(r"[\[\]\"]", "", val_str).strip()
    return val_str or "General"

def _normalize_product_dict(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    prod_id = (
        row.get("product_id") or row.get("uniq_id") or row.get("pid") or
        row.get("id") or row.get("sku") or f"SKU-{i:06d}"
    )
    title = (
        row.get("prod_title") or row.get("product_name") or row.get("Title") or
        row.get("title") or row.get("name") or "Product"
    )
    raw_img = (
        row.get("prod_image_url") or row.get("image_url") or row.get("image")
    )
    raw_price = (
        row.get("price") or row.get("Current/discounted_price") or
        row.get("discounted_price") or row.get("retail_price") or row.get("Price_on_variant")
    )
    cat = row.get("category") or row.get("product_category_tree")
    brand = row.get("brand") or row.get("Brand") or "Generic"

    return {
        "product_id": str(prod_id).strip(),
        "prod_title": str(title).strip(),
        "prod_image_url": _clean_image_url(raw_img),
        "price": _clean_price(raw_price),
        "category": _clean_category(cat),
        "brand": str(brand).strip()
    }

def read_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Reads CSV, JSON, or JSONL file into product dictionaries."""
    products = []
    ext = os.path.splitext(file_path)[1].lower()

    logger.info(f"Reading dataset file: {file_path}")

    if ext == ".csv":
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                products.append(_normalize_product_dict(row, i))
    elif ext in (".json", ".jsonl"):
        with open(file_path, mode="r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
            else:
                items = [json.loads(line) for line in f if line.strip()]

            for i, row in enumerate(items):
                products.append(_normalize_product_dict(row, i))
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Please use .csv, .json, or .jsonl")

    logger.info(f"Loaded {len(products):,} products from file.")
    return products

async def process_batch(batch: List[Dict[str, Any]], fast_embedding: bool = False):
    """Processes a chunk batch of products: generates vectors in parallel and upserts to Qdrant."""
    ingest_requests = []
    text_tasks = []
    vision_tasks = []

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
            text_tasks.append(text_embedder.embed_text(composite_text))
            if fast_embedding:
                vision_tasks.append(asyncio.sleep(0, result=vision_embedder._fallback_embed(str(req.prod_image_url))))
            else:
                vision_tasks.append(vision_embedder.embed_image_url(str(req.prod_image_url)))
        except Exception as e:
            logger.warning(f"Error preparing item {item.get('product_id')}: {e}")

    if ingest_requests:
        text_vectors = await asyncio.gather(*text_tasks)
        vision_vectors = await asyncio.gather(*vision_tasks)
        await vector_db_manager.upsert_batch(ingest_requests, text_vectors, vision_vectors)

async def main(file_path: str, batch_size: int = 500, fast_embedding: bool = False):
    await vector_db_manager.init_collection()
    products = read_dataset(file_path)

    start_time = time.time()
    total_processed = 0

    for i in range(0, len(products), batch_size):
        chunk = products[i:i + batch_size]
        await process_batch(chunk, fast_embedding=fast_embedding)
        total_processed += len(chunk)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"Progress: {total_processed:,} / {len(products):,} products indexed ({rate:.1f} items/sec)")

    logger.info(f"✅ Ingestion Complete! Indexed {total_processed:,} products in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Large Dataset into Multimodal RAG Engine")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV, JSON, or JSONL product dataset")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of products per vector batch (default: 500)")
    parser.add_argument("--fast", action="store_true", help="Enable high-speed feature hashing for vision embeddings")
    args = parser.parse_args()

    asyncio.run(main(args.file, args.batch_size, fast_embedding=args.fast))
