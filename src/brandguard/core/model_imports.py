"""
Author: Omer Sayem
Date: 2025-09-14
Version: 1.0.0
Description:
Model Imports and Initialization
Handles importing and initializing all BrandGuard models
Self-contained - uses only local implementations within consolidated_pipeline
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global flag to track if models are loaded
MODELS_LOADED = False
imported_models = {}

# Model references for pipeline orchestrator
FontIdentifier = None
TextExtractor = None
ColorPaletteExtractor = None
ContrastChecker = None
TypographyValidator = None
ToneAnalyzer = None
CopywritingTextExtractor = None
BrandVoiceValidator = None
LogoDetector = None
LogoValidator = None


def import_all_models():
    """Import all available models from local consolidated_pipeline modules"""
    global MODELS_LOADED, imported_models
    global FontIdentifier, TextExtractor, ColorPaletteExtractor, ContrastChecker
    global TypographyValidator, ToneAnalyzer, CopywritingTextExtractor
    global BrandVoiceValidator, LogoDetector, LogoValidator
    
    logger.info("🚀 Starting import_all_models (self-contained mode)...")
    try:
        # Try to import FontIdentifier (HuggingFace-based font detection + PaddleOCR)
        try:
            from .font_identifier import MultilingualFontIdentifier, create_font_identifier
            FontIdentifier = create_font_identifier
            logger.info("✅ MultilingualFontIdentifier loaded (PaddleOCR + HuggingFace font-identifier)")
        except ImportError as e:
            logger.warning(f"⚠️ FontIdentifier not available: {e}")
            FontIdentifier = None
        
        # Import LogoDetector (HuggingFace YOLOv8-based logo detection)
        try:
            from .logo_detector import LogoDetector, LogoPlacementValidator, create_logo_detector
            LogoDetector = create_logo_detector
            LogoValidator = LogoPlacementValidator
            logger.info("✅ LogoDetector loaded (YOLOv8 logo detection + brand identification)")
        except ImportError as e:
            logger.warning(f"⚠️ LogoDetector not available: {e}")
            LogoDetector = None
            LogoValidator = None
        
        # Self-contained mode: Use local implementations
        # All analyzers (ColorAnalyzer, LogoAnalyzer, TypographyAnalyzer, CopywritingAnalyzer)
        # are implemented within consolidated_pipeline and handle their own model initialization
        
        logger.info("✅ Using self-contained model implementations")
        logger.info("   All models are handled by local analyzers within consolidated_pipeline")
        
        # Return dict with all models for pipeline orchestrator
        MODELS_LOADED = True
        imported_models = {
            'FontIdentifier': FontIdentifier,
            'LogoDetector': LogoDetector,
            'LogoValidator': LogoValidator,
        }
        
    except Exception as e:
        MODELS_LOADED = False
        logger.warning(f"Error during model import: {e}. Using fallback implementations.")
        imported_models = {
            'FontIdentifier': None,
            'LogoDetector': None,
            'LogoValidator': None,
        }

    logger.info("💡 All models use local implementations within consolidated_pipeline")
    logger.info("   No external directory dependencies required")

    return MODELS_LOADED, imported_models

def get_imported_models():
    """Get the dictionary of imported models"""
    global MODELS_LOADED
    if not MODELS_LOADED:
        import_all_models()
    return imported_models

def is_models_loaded():
    """Check if models are loaded"""
    return MODELS_LOADED
