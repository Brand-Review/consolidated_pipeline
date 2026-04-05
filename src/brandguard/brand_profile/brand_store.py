"""
Brand Store
MongoDB CRUD for BrandProfile documents.
Collection: brand_profiles
"""

from __future__ import annotations
import logging
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_COLLECTION = "brand_profiles"


class BrandStore:
    """
    Thin wrapper around a MongoDB collection for storing brand profiles.
    Connection is lazy — no DB call at construction time.
    """

    def __init__(self, mongo_uri: str = None, db_name: str = None):
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        self.db_name = db_name or os.environ.get("MONGODB_DB")
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy DB connection
    # ------------------------------------------------------------------

    def _get_collection(self):
        if self._collection is None:
            from pymongo import MongoClient
            self._client = MongoClient(self.mongo_uri)
            self._collection = self._client[self.db_name][_COLLECTION]
            # Ensure index on brand_id for fast lookups
            self._collection.create_index("brand_id", unique=True)
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, brand_name: str, brand_id: str = None) -> str:
        """
        Create a new brand profile stub and return the brand_id.
        If brand_id is provided (e.g. a MongoDB folder _id), it is used directly;
        otherwise a new UUID is generated.
        Uses upsert so re-onboarding the same folder is safe.
        The caller is responsible for calling update() to fill in rules/RAG metadata.
        """
        brand_id = brand_id or str(uuid.uuid4())
        now = datetime.utcnow()
        doc = {
            "brand_id": brand_id,
            "brand_name": brand_name,
            "created_at": now,
            "updated_at": now,
            "rules": {
                "color_rules": {"palette": [], "gradient": None, "forbidden": [], "raw_description": ""},
                "typography_rules": {
                    "bangla_font": None, "english_font": None,
                    "approved_fonts": [], "forbidden_fonts": [], "raw_description": ""
                },
                "logo_rules": {
                    "position": None, "min_height_px": None,
                    "dark_bg_use_white": False, "colorful_on_white_only": False,
                    "allowed_zones": [], "raw_description": ""
                },
                "brand_voice_rules": {"tone": None, "language": None, "raw_description": ""},
            },
            "qdrant_guideline_collection": None,
            "qdrant_asset_collection": None,
            "chunk_count": 0,
            "asset_count": 0,
        }
        self._get_collection().update_one(
            {"brand_id": brand_id},
            {"$setOnInsert": {"created_at": now}, "$set": {k: v for k, v in doc.items() if k != "created_at"}},
            upsert=True,
        )
        logger.info(f"Upserted brand profile: {brand_id} ({brand_name})")
        return brand_id

    def get(self, brand_id: str) -> Optional[Dict[str, Any]]:
        """Return the brand profile document or None if not found."""
        doc = self._get_collection().find_one({"brand_id": brand_id}, {"_id": 0})
        return doc

    def update(self, brand_id: str, updates: Dict[str, Any]):
        """Merge updates into an existing brand profile."""
        updates["updated_at"] = datetime.utcnow()
        self._get_collection().update_one(
            {"brand_id": brand_id},
            {"$set": updates},
        )

    def delete(self, brand_id: str):
        """Delete a brand profile from MongoDB (Qdrant collections must be deleted separately)."""
        self._get_collection().delete_one({"brand_id": brand_id})
        logger.info(f"Deleted brand profile: {brand_id}")

    def list_brands(self) -> List[Dict[str, Any]]:
        """Return all brand profiles (without _id field)."""
        return list(self._get_collection().find({}, {"_id": 0}))
