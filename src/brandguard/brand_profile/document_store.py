"""
DocumentStore — persistence for per-brand uploaded documents and per-brand
RAG config overrides. Appends to `brand_profiles.documents[]`.
"""

from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_COLLECTION = "brand_profiles"


class DocumentStore:
    def __init__(self, mongo_uri: str = None, db_name: str = None):
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        self.db_name = db_name or os.environ.get("MONGODB_DB")
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            from pymongo import MongoClient
            self._client = MongoClient(self.mongo_uri)
            self._collection = self._client[self.db_name][_COLLECTION]
        return self._collection

    def add_document(self, brand_id: str, doc_meta: Dict[str, Any]):
        doc_meta = dict(doc_meta)
        doc_meta.setdefault("uploaded_at", datetime.utcnow())
        self._get_collection().update_one(
            {"brand_id": brand_id},
            {"$push": {"documents": doc_meta}, "$set": {"updated_at": datetime.utcnow()}},
            upsert=True,
        )
        logger.info(f"Recorded document {doc_meta.get('doc_id')} for brand {brand_id}")

    def list_documents(self, brand_id: str) -> List[Dict[str, Any]]:
        doc = self._get_collection().find_one({"brand_id": brand_id}, {"_id": 0, "documents": 1})
        return (doc or {}).get("documents", []) or []

    def delete_document(self, brand_id: str, doc_id: str):
        self._get_collection().update_one(
            {"brand_id": brand_id},
            {"$pull": {"documents": {"doc_id": doc_id}}, "$set": {"updated_at": datetime.utcnow()}},
        )

    def get_rag_overrides(self, brand_id: str) -> Optional[Dict[str, Any]]:
        doc = self._get_collection().find_one(
            {"brand_id": brand_id}, {"_id": 0, "rag_config_overrides": 1}
        )
        if not doc:
            return None
        return doc.get("rag_config_overrides")

    def set_rag_overrides(self, brand_id: str, overrides: Dict[str, Any]):
        self._get_collection().update_one(
            {"brand_id": brand_id},
            {"$set": {"rag_config_overrides": overrides, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
