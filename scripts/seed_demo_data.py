import asyncio
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_script")

BASE_URL = "http://localhost:8000/api/v1"

SAMPLE_PRODUCTS = [
    {
        "product_id": "SKU-AUDIO-001",
        "prod_title": "Sony WH-1000XM5 Wireless Noise-Canceling Over-Ear Headphones",
        "prod_image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
        "price": 398.00,
        "category": "Electronics > Audio > Headphones",
        "brand": "Sony"
    },
    {
        "product_id": "SKU-AUDIO-002",
        "prod_title": "Bose QuietComfort Ultra Wireless Noise Cancelling Headphones",
        "prod_image_url": "https://images.unsplash.com/photo-1546435770-a3e426bf472b",
        "price": 429.00,
        "category": "Electronics > Audio > Headphones",
        "brand": "Bose"
    },
    {
        "product_id": "SKU-AUDIO-003",
        "prod_title": "Sennheiser Momentum 4 Wireless Headphones",
        "prod_image_url": "https://images.unsplash.com/photo-1583394838336-acd977736f90",
        "price": 289.00,
        "category": "Electronics > Audio > Headphones",
        "brand": "Sennheiser"
    },
    {
        "product_id": "SKU-AUDIO-004",
        "prod_title": "Apple AirPods Max Wireless Over-Ear Headphones",
        "prod_image_url": "https://images.unsplash.com/photo-1628202926206-c63a34b1618f",
        "price": 549.00,
        "category": "Electronics > Audio > Headphones",
        "brand": "Apple"
    },
    {
        "product_id": "SKU-AUDIO-005",
        "prod_title": "Sony WF-1000XM5 Noise-Canceling Truly Wireless Earbuds",
        "prod_image_url": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df",
        "price": 278.00,
        "category": "Electronics > Audio > Earbuds",
        "brand": "Sony"
    },
    {
        "product_id": "SKU-WEARABLE-001",
        "prod_title": "Apple Watch Series 9 GPS 45mm Smartwatch",
        "prod_image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30",
        "price": 429.00,
        "category": "Electronics > Wearables > Smartwatches",
        "brand": "Apple"
    },
    {
        "product_id": "SKU-WEARABLE-002",
        "prod_title": "Samsung Galaxy Watch 6 Classic 47mm Bluetooth",
        "prod_image_url": "https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1",
        "price": 349.00,
        "category": "Electronics > Wearables > Smartwatches",
        "brand": "Samsung"
    },
    {
        "product_id": "SKU-LAPTOP-001",
        "prod_title": "Apple MacBook Pro 14-inch M3 Chip 18GB Memory 512GB SSD Space Black",
        "prod_image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
        "price": 1999.00,
        "category": "Electronics > Computers > Laptops",
        "brand": "Apple"
    },
    {
        "product_id": "SKU-LAPTOP-002",
        "prod_title": "Dell XPS 15 9530 15.6 inch FHD Laptop Intel i7 16GB 512GB SSD",
        "prod_image_url": "https://images.unsplash.com/photo-1593642632823-8f785ba67e45",
        "price": 1499.00,
        "category": "Electronics > Computers > Laptops",
        "brand": "Dell"
    },
    {
        "product_id": "SKU-CAMERA-001",
        "prod_title": "Sony Alpha 7 IV Full-Frame Mirrorless Interchangeable Lens Camera",
        "prod_image_url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32",
        "price": 2498.00,
        "category": "Electronics > Cameras > Mirrorless",
        "brand": "Sony"
    },
    {
        "product_id": "SKU-FASHION-001",
        "prod_title": "Nike Air Force 1 '07 Classic White Sneakers",
        "prod_image_url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a",
        "price": 115.00,
        "category": "Apparel > Footwear > Sneakers",
        "brand": "Nike"
    },
    {
        "product_id": "SKU-FASHION-002",
        "prod_title": "Adidas Ultraboost Light Running Shoes Black",
        "prod_image_url": "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2",
        "price": 190.00,
        "category": "Apparel > Footwear > Sneakers",
        "brand": "Adidas"
    }
]

async def seed_data():
    logger.info("Connecting to microservice ingestion API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BASE_URL}/products/batch-ingest", json={"products": SAMPLE_PRODUCTS})
            if resp.status_code == 200:
                logger.info(f"Successfully seeded demo catalog! Response: {resp.json()}")
            else:
                logger.error(f"Seeding failed with HTTP status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Could not connect to {BASE_URL}. Ensure FastAPI microservice is running. Error: {e}")

if __name__ == "__main__":
    asyncio.run(seed_data())
