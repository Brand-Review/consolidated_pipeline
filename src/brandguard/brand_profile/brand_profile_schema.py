"""
Brand Profile Schema
Pydantic models for structured brand compliance rules extracted from PDF guidelines.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ColorRules(BaseModel):
    palette: List[str] = Field(default_factory=list, description="Hex color codes in brand palette")
    gradient: Optional[str] = None
    forbidden: List[str] = Field(default_factory=list)
    raw_description: str = ""


class TypographyRules(BaseModel):
    bangla_font: Optional[str] = None
    english_font: Optional[str] = None
    approved_fonts: List[str] = Field(default_factory=list)
    forbidden_fonts: List[str] = Field(default_factory=list)
    raw_description: str = ""


class LogoRules(BaseModel):
    position: Optional[str] = None          # e.g. "top-right"
    min_height_px: Optional[int] = None     # e.g. 85
    dark_bg_use_white: bool = False
    colorful_on_white_only: bool = False
    allowed_zones: List[str] = Field(default_factory=list)
    raw_description: str = ""


class BrandVoiceRules(BaseModel):
    tone: Optional[str] = None
    language: Optional[str] = None
    raw_description: str = ""


class BrandProfileRules(BaseModel):
    color_rules: ColorRules = Field(default_factory=ColorRules)
    typography_rules: TypographyRules = Field(default_factory=TypographyRules)
    logo_rules: LogoRules = Field(default_factory=LogoRules)
    brand_voice_rules: BrandVoiceRules = Field(default_factory=BrandVoiceRules)


class BrandProfile(BaseModel):
    brand_id: str
    brand_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Structured rules extracted from PDF by LLM
    rules: BrandProfileRules = Field(default_factory=BrandProfileRules)

    # Qdrant collection references (set after indexing)
    qdrant_guideline_collection: Optional[str] = None
    qdrant_asset_collection: Optional[str] = None

    # Counts for quick status checks
    chunk_count: int = 0
    asset_count: int = 0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
