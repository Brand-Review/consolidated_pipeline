"""
Model Imports and Initialization
Handles importing and initializing all BrandGuard models
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global flag to track if models are loaded
MODELS_LOADED = False
imported_models = {}

def import_all_models():
    """Import all available models from individual modules"""
    global MODELS_LOADED, imported_models
    
    try:
        # Get the parent directory (one level up from consolidated_pipeline)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: core -> brandguard -> src -> consolidated_pipeline -> brandReviewModels
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        
        # Add paths to individual model directories
        color_path = os.path.join(parent_dir, 'ColorPaletteChecker', 'src')
        typography_path = os.path.join(parent_dir, 'FontTypographyChecker', 'src')
        copywriting_path = os.path.join(parent_dir, 'CopywritingToneChecker', 'src')
        logo_path = os.path.join(parent_dir, 'LogoDetector', 'src')
        
        # Add paths to sys.path
        for path in [color_path, typography_path, copywriting_path, logo_path]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Import Color Analysis components using importlib
        try:
            import importlib.util
            color_palette_path = os.path.join(color_path, 'brandguard', 'core', 'color_palette.py')
            if os.path.exists(color_palette_path):
                spec = importlib.util.spec_from_file_location("color_palette", color_palette_path)
                color_palette_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(color_palette_module)
                ColorPaletteExtractor = color_palette_module.ColorPaletteExtractor
                ColorPaletteValidator = color_palette_module.ColorPaletteValidator
                imported_models['ColorPaletteExtractor'] = ColorPaletteExtractor
                imported_models['ColorPaletteValidator'] = ColorPaletteValidator
                print("✅ ColorPaletteExtractor and ColorPaletteValidator imported successfully using importlib")
            else:
                print(f"❌ ColorPaletteExtractor file not found at: {color_palette_path}")
                ColorPaletteExtractor = None
                ColorPaletteValidator = None
        except Exception as e:
            print(f"❌ ColorPaletteExtractor import failed: {e}")
            ColorPaletteExtractor = None
            ColorPaletteValidator = None
        
        # Import Typography Analysis components using importlib
        try:
            import importlib.util
            font_identifier_path = os.path.join(typography_path, 'brandguard', 'core', 'font_identifier.py')
            if os.path.exists(font_identifier_path):
                spec = importlib.util.spec_from_file_location("font_identifier", font_identifier_path)
                font_identifier_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(font_identifier_module)
                FontIdentifier = font_identifier_module.FontIdentifier
                imported_models['FontIdentifier'] = FontIdentifier
                print("✅ FontIdentifier imported successfully using importlib")
            else:
                print(f"❌ FontIdentifier file not found at: {font_identifier_path}")
                FontIdentifier = None
        except Exception as e:
            print(f"❌ FontIdentifier import failed: {e}")
            FontIdentifier = None
        
        try:
            import importlib.util
            typography_validator_path = os.path.join(typography_path, 'brandguard', 'core', 'typography_validator.py')
            if os.path.exists(typography_validator_path):
                spec = importlib.util.spec_from_file_location("typography_validator", typography_validator_path)
                typography_validator_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(typography_validator_module)
                TypographyValidator = typography_validator_module.TypographyValidator
                imported_models['TypographyValidator'] = TypographyValidator
                print("✅ TypographyValidator imported successfully using importlib")
            else:
                print(f"❌ TypographyValidator file not found at: {typography_validator_path}")
                TypographyValidator = None
        except Exception as e:
            print(f"❌ TypographyValidator import failed: {e}")
            TypographyValidator = None
        
        # Import FontComplianceChecker (main orchestrator)
        try:
            import importlib.util
            
            font_compliance_path = os.path.join(typography_path, 'brandguard', 'core', 'font_compliance.py')
            if os.path.exists(font_compliance_path):
                # Import the module with proper path handling
                spec = importlib.util.spec_from_file_location("font_compliance", font_compliance_path)
                font_compliance_module = importlib.util.module_from_spec(spec)
                
                # Set up the module's __package__ attribute to avoid relative import issues
                font_compliance_module.__package__ = 'brandguard.core'
                
                spec.loader.exec_module(font_compliance_module)
                FontComplianceChecker = font_compliance_module.FontComplianceChecker
                imported_models['FontComplianceChecker'] = FontComplianceChecker
                print("✅ FontComplianceChecker imported successfully using importlib")
            else:
                print(f"❌ FontComplianceChecker file not found at: {font_compliance_path}")
                FontComplianceChecker = None
        except Exception as e:
            print(f"❌ FontComplianceChecker import failed: {e}")
            FontComplianceChecker = None
        
        # Import TextExtractor (PaddleOCR integration)
        try:
            import importlib.util
            text_extractor_path = os.path.join(typography_path, 'brandguard', 'core', 'text_extractor.py')
            if os.path.exists(text_extractor_path):
                spec = importlib.util.spec_from_file_location("text_extractor", text_extractor_path)
                text_extractor_module = importlib.util.module_from_spec(spec)
                
                # Set up the module's __package__ attribute to avoid relative import issues
                text_extractor_module.__package__ = 'brandguard.core'
                
                spec.loader.exec_module(text_extractor_module)
                TextExtractor = text_extractor_module.TextExtractor
                imported_models['TextExtractor'] = TextExtractor
                print("✅ TextExtractor imported successfully using importlib")
            else:
                print(f"❌ TextExtractor file not found at: {text_extractor_path}")
                TextExtractor = None
        except Exception as e:
            print(f"❌ TextExtractor import failed: {e}")
            TextExtractor = None
        
        # Import Copywriting Analysis components using importlib
        try:
            import importlib.util
            tone_analyzer_path = os.path.join(copywriting_path, 'brandguard', 'core', 'tone_analyzer.py')
            if os.path.exists(tone_analyzer_path):
                spec = importlib.util.spec_from_file_location("tone_analyzer", tone_analyzer_path)
                tone_analyzer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tone_analyzer_module)
                ToneAnalyzer = tone_analyzer_module.ToneAnalyzer
                imported_models['ToneAnalyzer'] = ToneAnalyzer
                print("✅ ToneAnalyzer imported successfully using importlib")
            else:
                print(f"❌ ToneAnalyzer file not found at: {tone_analyzer_path}")
                ToneAnalyzer = None
        except Exception as e:
            print(f"❌ ToneAnalyzer import failed: {e}")
            ToneAnalyzer = None
        
        try:
            import importlib.util
            brand_voice_validator_path = os.path.join(copywriting_path, 'brandguard', 'core', 'brand_voice_validator.py')
            if os.path.exists(brand_voice_validator_path):
                spec = importlib.util.spec_from_file_location("brand_voice_validator", brand_voice_validator_path)
                brand_voice_validator_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(brand_voice_validator_module)
                BrandVoiceValidator = brand_voice_validator_module.BrandVoiceValidator
                imported_models['BrandVoiceValidator'] = BrandVoiceValidator
                print("✅ BrandVoiceValidator imported successfully using importlib")
            else:
                print(f"❌ BrandVoiceValidator file not found at: {brand_voice_validator_path}")
                BrandVoiceValidator = None
        except Exception as e:
            print(f"❌ BrandVoiceValidator import failed: {e}")
            BrandVoiceValidator = None
        
        # Import Logo Detection components using importlib
        try:
            import importlib.util
            logo_detector_path = os.path.join(logo_path, 'brandguard', 'core', 'detector.py')
            if os.path.exists(logo_detector_path):
                spec = importlib.util.spec_from_file_location("logo_detector", logo_detector_path)
                logo_detector_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(logo_detector_module)
                LogoDetector = logo_detector_module.LogoDetector
                imported_models['LogoDetector'] = LogoDetector
                print("✅ LogoDetector imported successfully using importlib")
            else:
                print(f"❌ LogoDetector file not found at: {logo_detector_path}")
                LogoDetector = None
        except Exception as e:
            print(f"❌ LogoDetector import failed: {e}")
            LogoDetector = None
        
        try:
            import importlib.util
            logo_validator_path = os.path.join(logo_path, 'brandguard', 'core', 'validator.py')
            if os.path.exists(logo_validator_path):
                spec = importlib.util.spec_from_file_location("logo_validator", logo_validator_path)
                logo_validator_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(logo_validator_module)
                LogoPlacementValidator = logo_validator_module.LogoPlacementValidator
                imported_models['LogoPlacementValidator'] = LogoPlacementValidator
                print("✅ LogoPlacementValidator imported successfully using importlib")
            else:
                print(f"❌ LogoPlacementValidator file not found at: {logo_validator_path}")
                LogoPlacementValidator = None
        except Exception as e:
            print(f"❌ LogoPlacementValidator import failed: {e}")
            LogoPlacementValidator = None
        
        try:
            import importlib.util
            pdf_processor_path = os.path.join(logo_path, 'brandguard', 'core', 'pdf_processor.py')
            if os.path.exists(pdf_processor_path):
                spec = importlib.util.spec_from_file_location("pdf_processor", pdf_processor_path)
                pdf_processor_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pdf_processor_module)
                PDFImageExtractor = pdf_processor_module.PDFImageExtractor
                PDFLogoDetector = pdf_processor_module.PDFLogoDetector
                imported_models['PDFImageExtractor'] = PDFImageExtractor
                imported_models['PDFLogoDetector'] = PDFLogoDetector
                print("✅ PDFImageExtractor and PDFLogoDetector imported successfully using importlib")
            else:
                print(f"❌ PDFImageExtractor file not found at: {pdf_processor_path}")
                PDFImageExtractor = None
                PDFLogoDetector = None
        except Exception as e:
            print(f"❌ PDFImageExtractor import failed: {e}")
            PDFImageExtractor = None
            PDFLogoDetector = None
        
        # LogoValidator is now imported above as LogoPlacementValidator
        LogoValidator = LogoPlacementValidator
        
        # Check if we have at least some models loaded
        if any(imported_models.values()):
            MODELS_LOADED = True
            logger.info("✅ Some models loaded successfully")
        else:
            MODELS_LOADED = False
            logger.warning("No real models could be imported. Using fallback implementations.")
        
    except Exception as e:
        MODELS_LOADED = False
        logger.warning(f"Error during model import: {e}. Using fallback implementations.")

    logger.info("💡 Note: Some models require additional ML libraries (torch, transformers)")
    logger.info("   Install them separately for full typography and copywriting analysis")

    return MODELS_LOADED, imported_models

def get_imported_models():
    """Get the dictionary of imported models"""
    return imported_models

def is_models_loaded():
    """Check if models are loaded"""
    return MODELS_LOADED
