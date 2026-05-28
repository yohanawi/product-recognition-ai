"""Backfill MongoDB image embeddings and build a FAISS index for jewelry products."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pymongo import MongoClient

from app import BASE_DIR, extract_features, read_image_from_bytes, read_image_from_url
from vector_search import build_faiss_index, save_faiss_artifacts

logger = logging.getLogger("catalog_indexer")
logging.basicConfig(level=logging.INFO)

DEFAULT_BACKEND_ROOT = BASE_DIR.parent / "backend"
DEFAULT_DOTENV_PATH = DEFAULT_BACKEND_ROOT / ".env"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "faiss"
DEFAULT_INDEX_PATH = DEFAULT_OUTPUT_DIR / "jewelry_products.index"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "jewelry_products.metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Index MongoDB jewelry products for visual search")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", ""), help="MongoDB connection string")
    parser.add_argument("--database", default=os.getenv("AI_SEARCH_DB_NAME", ""), help="MongoDB database name override")
    parser.add_argument("--collection", default=os.getenv("AI_SEARCH_COLLECTION", "products"), help="MongoDB collection name")
    parser.add_argument("--backend-root", default=str(DEFAULT_BACKEND_ROOT), help="Path to the backend project root")
    parser.add_argument("--env-file", default=str(DEFAULT_DOTENV_PATH), help="Optional dotenv file to load when mongo-uri is not set")
    parser.add_argument("--limit", type=int, default=0, help="Optional product limit for partial indexing")
    parser.add_argument("--skip-mongo-update", action="store_true", help="Do not write extracted vectors back to MongoDB")
    parser.add_argument("--skip-faiss", action="store_true", help="Do not write FAISS index artifacts")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Target FAISS index path")
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH), help="Target metadata JSON path")
    return parser.parse_args()


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def resolve_mongo_uri(args) -> str:
    dotenv_path = Path(args.env_file)
    if not args.mongo_uri:
        load_dotenv_file(dotenv_path)
    mongo_uri = args.mongo_uri or os.getenv("MONGO_URI", "")
    if not mongo_uri:
        raise ValueError("MongoDB URI is required. Pass --mongo-uri or configure backend/.env")
    return mongo_uri


def resolve_database_name(mongo_uri: str, explicit_name: str) -> str:
    if explicit_name:
        return explicit_name

    parsed = urlparse(mongo_uri)
    database_name = parsed.path.lstrip("/")
    if database_name:
        return database_name

    raise ValueError("Database name is required. Include it in the MongoDB URI path or pass --database")


def normalize_image_source(value) -> str:
    return str(value or "").strip()


def get_product_image_source(product: dict) -> str:
    thumbnail = normalize_image_source(product.get("thumbnailImage"))
    if thumbnail:
        return thumbnail

    images = product.get("images") or []
    if images:
        return normalize_image_source(images[0])
    return ""


def try_resolve_upload_path(image_source: str, backend_root: Path) -> Optional[Path]:
    normalized = normalize_image_source(image_source).replace("\\", "/")
    if not normalized:
        return None

    uploads_index = normalized.lower().find("/uploads/")
    if uploads_index >= 0:
        relative = normalized[uploads_index + 1 :]
        return backend_root / Path(relative)

    if normalized.lower().startswith("uploads/"):
        return backend_root / Path(normalized)

    return None


def resolve_local_image_path(image_source: str, backend_root: Path) -> Path:
    upload_path = try_resolve_upload_path(image_source, backend_root)
    if upload_path:
        return upload_path

    candidate = Path(image_source)
    if candidate.is_absolute():
        return candidate
    return backend_root / candidate


def load_image(image_source: str, backend_root: Path):
    if image_source.lower().startswith(("http://", "https://")):
        return read_image_from_url(image_source)

    local_path = resolve_local_image_path(image_source, backend_root)
    image_bytes = local_path.read_bytes()
    return read_image_from_bytes(image_bytes)


def build_metadata(product: dict, image_source: str, feature_size: int) -> dict:
    return {
        "product_id": str(product.get("_id")),
        "name": product.get("name", ""),
        "slug": product.get("slug", ""),
        "sku": product.get("sku", ""),
        "category": str(product.get("category") or ""),
        "image_source": image_source,
        "feature_size": feature_size,
    }


def index_products(collection, backend_root: Path, limit: int = 0, skip_mongo_update: bool = False):
    query = {
        "status": "active",
        "isArchived": {"$ne": True},
    }
    projection = {
        "name": 1,
        "slug": 1,
        "sku": 1,
        "category": 1,
        "thumbnailImage": 1,
        "images": 1,
    }

    cursor = collection.find(query, projection)
    if limit > 0:
        cursor = cursor.limit(limit)

    vectors = []
    metadata = []
    summary = {
        "total": 0,
        "indexed": 0,
        "skipped_no_image": 0,
        "failed": 0,
    }

    for product in cursor:
        summary["total"] += 1
        image_source = get_product_image_source(product)
        if not image_source:
            summary["skipped_no_image"] += 1
            continue

        try:
            image = load_image(image_source, backend_root)
            vector = extract_features(image)
            vectors.append(vector)
            metadata.append(build_metadata(product, image_source, len(vector)))

            if not skip_mongo_update:
                collection.update_one(
                    {"_id": product["_id"]},
                    {
                        "$set": {
                            "features": vector,
                            "featuresIndexed": True,
                            "featuresImageSignature": image_source,
                        }
                    },
                )

            summary["indexed"] += 1
            logger.info("Indexed %s (%s)", product.get("sku") or product.get("_id"), image_source)
        except Exception as error:  # pragma: no cover - depends on local catalog assets
            summary["failed"] += 1
            logger.warning("Failed indexing %s: %s", product.get("_id"), error)

    return vectors, metadata, summary


def main():
    args = parse_args()
    mongo_uri = resolve_mongo_uri(args)
    database_name = resolve_database_name(mongo_uri, args.database)
    backend_root = Path(args.backend_root).resolve()

    client = MongoClient(mongo_uri)
    try:
        collection = client[database_name][args.collection]
        vectors, metadata, summary = index_products(
            collection=collection,
            backend_root=backend_root,
            limit=args.limit,
            skip_mongo_update=args.skip_mongo_update,
        )

        if vectors and not args.skip_faiss:
            index, _normalized_vectors = build_faiss_index(vectors)
            save_faiss_artifacts(
                index=index,
                metadata=metadata,
                index_path=args.index_path,
                metadata_path=args.metadata_path,
            )
            summary["faiss_index_path"] = str(Path(args.index_path).resolve())
            summary["faiss_metadata_path"] = str(Path(args.metadata_path).resolve())
        elif not vectors:
            logger.warning("No vectors were generated, so FAISS artifacts were not written.")

        print(json.dumps(summary, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
