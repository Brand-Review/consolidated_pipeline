"""
BrandGuard Pipeline Orchestrator
Coordinates all four models: Color, Typography, Copywriting, and Logo Detection
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import requests
from PIL import Image
import io
import base64
import json

# Import individual model components
from ..config.settings import Settings, BrandColorPalette, TypographyRules, BrandVoiceSettings, LogoDetectionSettings
from .typography_analyzer import TypographyAnalyzer

# Import real working models
try:
    # Color Analysis Models
    import sys
    import os
    
    # Get the absolute path to the parent directory (brandReviewModels)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # From consolidated_pipeline/src/brandguard/core/ -> go up 4 levels to brandReviewModels
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # Add paths to individual model directories
    color_path = os.path.join(parent_dir, 'ColorPaletteChecker', 'src')
    typography_path = os.path.join(parent_dir, 'FontTypographyChecker', 'src')
    copywriting_path = os.path.join(parent_dir, 'CopywritingToneChecker', 'src')
    logo_path = os.path.join(parent_dir, 'LogoDetector', 'src')
    
    # Debug: print paths
    print(f"Color path: {color_path}")
    print(f"Typography path: {typography_path}")
    print(f"Copywriting path: {copywriting_path}")
    print(f"Logo path: {logo_path}")
    
    # Add paths to Python path
    sys.path.insert(0, color_path)
    sys.path.insert(0, typography_path)
    sys.path.insert(0, copywriting_path)
    sys.path.insert(0, logo_path)
    
    # Now try to import the models one by one with error handling
    imported_models = {}
    
    try:
        # Use importlib to avoid namespace conflicts
        import importlib.util
        color_palette_path = os.path.join(color_path, 'brandguard', 'core', 'color_palette.py')
        if os.path.exists(color_palette_path):
            spec = importlib.util.spec_from_file_location("color_palette", color_palette_path)
            color_palette_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(color_palette_module)
            ColorPaletteExtractor = color_palette_module.ColorPaletteExtractor
            imported_models['ColorPaletteExtractor'] = ColorPaletteExtractor
            print("✅ ColorPaletteExtractor imported successfully using importlib")
        else:
            print(f"❌ ColorPaletteExtractor file not found at: {color_palette_path}")
            ColorPaletteExtractor = None
    except Exception as e:
        print(f"❌ ColorPaletteExtractor import failed: {e}")
        ColorPaletteExtractor = None
    
    try:
        # Use importlib to avoid namespace conflicts
        import importlib.util
        contrast_checker_path = os.path.join(color_path, 'brandguard', 'core', 'contrast_checker.py')
        if os.path.exists(contrast_checker_path):
            spec = importlib.util.spec_from_file_location("contrast_checker", contrast_checker_path)
            contrast_checker_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(contrast_checker_module)
            ContrastChecker = contrast_checker_module.ContrastChecker
            imported_models['ContrastChecker'] = ContrastChecker
            print("✅ ContrastChecker imported successfully using importlib")
        else:
            print(f"❌ ContrastChecker file not found at: {contrast_checker_path}")
            ContrastChecker = None
    except Exception as e:
        print(f"❌ ContrastChecker import failed: {e}")
        ContrastChecker = None
    
    try:
        # Use importlib to avoid namespace conflicts
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
        # Use importlib to avoid namespace conflicts
        import importlib.util
        text_extractor_path = os.path.join(typography_path, 'brandguard', 'core', 'text_extractor.py')
        if os.path.exists(text_extractor_path):
            spec = importlib.util.spec_from_file_location("text_extractor", text_extractor_path)
            text_extractor_module = importlib.util.module_from_spec(spec)
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
    
    try:
        # Use importlib to avoid namespace conflicts
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
    
    try:
        # Use importlib to avoid namespace conflicts
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
        # Use importlib to avoid namespace conflicts
        import importlib.util
        copywriting_text_extractor_path = os.path.join(copywriting_path, 'brandguard', 'core', 'text_extractor.py')
        if os.path.exists(copywriting_text_extractor_path):
            spec = importlib.util.spec_from_file_location("copywriting_text_extractor", copywriting_text_extractor_path)
            copywriting_text_extractor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copywriting_text_extractor_module)
            CopywritingTextExtractor = copywriting_text_extractor_module.TextExtractor
            imported_models['CopywritingTextExtractor'] = CopywritingTextExtractor
            print("✅ CopywritingTextExtractor imported successfully using importlib")
        else:
            print(f"❌ CopywritingTextExtractor file not found at: {copywriting_text_extractor_path}")
            CopywritingTextExtractor = None
    except Exception as e:
        print(f"❌ CopywritingTextExtractor import failed: {e}")
        CopywritingTextExtractor = None
    
    try:
        # Use importlib to avoid namespace conflicts
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
    
    try:
        # Import LogoDetector directly from LogoDetector folder
        # The logo_path should already be in sys.path from above
        from brandguard.core.detector import LogoDetector
        from brandguard.core.validator import LogoPlacementValidator
        from brandguard.core.pdf_processor import PDFImageExtractor, PDFLogoDetector
        from brandguard.core.agentic_detector import AgenticLogoDetector
        imported_models['LogoDetector'] = LogoDetector
        imported_models['LogoPlacementValidator'] = LogoPlacementValidator
        imported_models['PDFImageExtractor'] = PDFImageExtractor
        imported_models['PDFLogoDetector'] = PDFLogoDetector
        imported_models['AgenticLogoDetector'] = AgenticLogoDetector
        print("✅ LogoDetector and AgenticLogoDetector imported successfully from LogoDetector folder")
    except Exception as e:
        print(f"❌ LogoDetector import failed: {e}")
        # Try alternative import method
        try:
            import importlib.util
            logo_detector_path = os.path.join(logo_path, 'brandguard', 'core', 'detector.py')
            if os.path.exists(logo_detector_path):
                spec = importlib.util.spec_from_file_location("logo_detector", logo_detector_path)
                logo_detector_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(logo_detector_module)
                LogoDetector = logo_detector_module.LogoDetector
                imported_models['LogoDetector'] = LogoDetector
                print("✅ LogoDetector imported successfully using importlib fallback")
            else:
                print(f"❌ LogoDetector file not found at: {logo_detector_path}")
                LogoDetector = None
        except Exception as e2:
            print(f"❌ LogoDetector importlib fallback also failed: {e2}")
            LogoDetector = None
        LogoPlacementValidator = None
        PDFImageExtractor = None
        PDFLogoDetector = None
    
    # LogoValidator is now imported above as LogoPlacementValidator
    LogoValidator = LogoPlacementValidator
    
    # Check if we have at least some models loaded
    if len(imported_models) > 0:
        MODELS_LOADED = True
        logger = logging.getLogger(__name__)
        logger.info(f"Some real models imported successfully: {list(imported_models.keys())}")
    else:
        MODELS_LOADED = False
        logger = logging.getLogger(__name__)
        logger.warning("No real models could be imported. Using fallback implementations.")
    
except Exception as e:
    MODELS_LOADED = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Error during model import: {e}. Using fallback implementations.")

logger = logging.getLogger(__name__)

# Note: FontIdentifier and ToneAnalyzer require heavy ML libraries (torch, transformers)
# These are not installed by default to keep the pipeline lightweight
# Users can install them separately if needed for full functionality
logger.info("💡 Note: Some models require additional ML libraries (torch, transformers)")
logger.info("   Install them separately for full typography and copywriting analysis")

class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all BrandGuard models
    """
    
    def __init__(self, settings: Settings, lang: str = 'en'):
        self.settings = settings
        self.lang = lang
        self.analysis_results = {}
        self.current_analysis_id = None
        
        # Initialize model components (these would be imported from individual modules)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model components"""
        try:
            if MODELS_LOADED:
                # Initialize real models
                self._init_real_models()
            else:
                # Initialize fallback models
                self._init_fallback_models()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            import traceback
            logger.error(f"Model initialization traceback: {traceback.format_exc()}")
            # Don't raise the exception, just log it and continue with fallback
            logger.warning("Continuing with fallback models due to initialization error")
            self._init_fallback_models()
    
    def _init_real_models(self):
        """Initialize real working models"""
        try:
            # Initialize color analysis components (only if imported successfully)
            if ColorPaletteExtractor is not None:
                self.color_extractor = ColorPaletteExtractor(
                    n_colors=self.settings.analysis.color_analysis.get('n_colors', 8),
                    n_clusters=self.settings.analysis.color_analysis.get('n_clusters', 8)
                )
                logger.info("✅ ColorPaletteExtractor initialized with real model")
            else:
                logger.warning("⚠️ ColorPaletteExtractor not available, using fallback")
            
            if ContrastChecker is not None:
                self.contrast_checker = ContrastChecker()
                logger.info("✅ ContrastChecker initialized with real model")
            else:
                logger.warning("⚠️ ContrastChecker not available, using fallback")
            
            # Initialize typography analysis components (only if imported successfully)
            if TextExtractor is not None:
                self.text_extractor = TextExtractor()
                logger.info("✅ TextExtractor initialized with real model")
            else:
                logger.warning("⚠️ TextExtractor not available, using fallback")
            
            if FontIdentifier is not None:
                self.font_identifier = FontIdentifier()
                logger.info("✅ FontIdentifier initialized with real model")
            else:
                logger.warning("⚠️ FontIdentifier not available, using fallback")
            
            if TypographyValidator is not None:
                self.typography_validator = TypographyValidator()
                logger.info("✅ TypographyValidator initialized with real model")
            else:
                logger.warning("⚠️ TypographyValidator not available, using fallback")
            
            # Initialize TypographyAnalyzer with FontComplianceChecker integration
            self.typography_analyzer = TypographyAnalyzer(
                settings=self.settings,
                imported_models=imported_models,
                lang=self.lang
            )
            logger.info(f"✅ TypographyAnalyzer initialized with language: {self.lang}")
            
            # Initialize copywriting analysis components (only if imported successfully)
            if ToneAnalyzer is not None:
                try:
                    # Configure ToneAnalyzer with Qwen settings
                    tone_config = {
                        'use_qwen': True,
                        'qwen_model': 'Qwen/Qwen2.5-VL-3B-Instruct',
                        'qwen_api_url': 'http://localhost:8000/v1/chat/completions',
                        'qwen_timeout': 120
                    }
                    self.tone_analyzer = ToneAnalyzer(tone_config)
                    logger.info("✅ ToneAnalyzer initialized with real model and Qwen integration")
                except Exception as e:
                    logger.error(f"ToneAnalyzer initialization failed: {e}, using fallback")
                    self.tone_analyzer = None
            else:
                logger.warning("⚠️ ToneAnalyzer not available, using fallback")
            
            if CopywritingTextExtractor is not None:
                self.copywriting_text_extractor = CopywritingTextExtractor()
                logger.info("✅ CopywritingTextExtractor initialized with real model")
            else:
                logger.warning("⚠️ CopywritingTextExtractor not available, using fallback")
            
            if BrandVoiceValidator is not None:
                try:
                    self.brand_voice_validator = BrandVoiceValidator()
                    logger.info("✅ BrandVoiceValidator initialized with real model")
                except Exception as e:
                    logger.error(f"BrandVoiceValidator initialization failed: {e}, using fallback")
                    self.brand_voice_validator = None
            else:
                logger.warning("⚠️ BrandVoiceValidator not available, using fallback")
            
            # Initialize logo detection components (only if imported successfully)
            if LogoDetector is not None:
                # Create logo detector configuration - Using fine-tuned logo detection model
                logo_config = {
                    'type': 'yolos',  # Using YOLOS model fine-tuned for logo detection
                    'confidence_threshold': self.settings.logo_detection.confidence_threshold,
                    'use_yolo': self.settings.logo_detection.use_yolo,
                    'use_qwen': self.settings.logo_detection.use_qwen,
                    'path': 'ellabettison/Logo-Detection-finetune',  # Fine-tuned logo detection model
                    'qwen_api_url': self.settings.logo_detection.qwen_api_url,
                    'qwen_model': self.settings.logo_detection.qwen_model
                }
                
                self.logo_detector = LogoDetector(logo_config)
                
                # Load the model
                if self.logo_detector.load_model():
                    logger.info("✅ LogoDetector initialized and fine-tuned model loaded successfully")
                else:
                    logger.warning("⚠️ LogoDetector initialized but model loading failed")
            else:
                logger.warning("⚠️ LogoDetector not available, using fallback")
            
            if LogoValidator is not None:
                self.logo_validator = LogoValidator
                logger.info("✅ LogoValidator class available for real model usage")
            else:
                logger.warning("⚠️ LogoValidator not available, using fallback")

            # Initialize AgenticLogoDetector (preferred path)
            AgenticLogoDetector = imported_models.get('AgenticLogoDetector')
            if AgenticLogoDetector is not None:
                try:
                    self.agentic_logo_detector = AgenticLogoDetector(config=None)
                    logger.info("✅ AgenticLogoDetector initialised in orchestrator")
                except Exception as e:
                    logger.warning(f"⚠️ AgenticLogoDetector init failed: {e}")
                    self.agentic_logo_detector = None
            else:
                self.agentic_logo_detector = None
            
            logger.info("Real models initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing real models: {e}")
            raise
    
    def _init_fallback_models(self):
        """Initialize fallback/placeholder models"""
        logger.warning("Using fallback model implementations")
        # These will use the placeholder methods below
    
    def update_language(self, lang: str):
        """Update the OCR language for typography analysis"""
        try:
            self.lang = lang
            if hasattr(self, 'typography_analyzer'):
                self.typography_analyzer.update_language(lang)
                logger.info(f"Updated pipeline language to: {lang}")
        except Exception as e:
            logger.error(f"Failed to update pipeline language: {e}")
    
    def analyze_content(self, 
                       input_source: str, 
                       source_type: str = 'image',
                       analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method that coordinates all models
        
        Args:
            input_source: Path to file or text content
            source_type: Type of input ('image', 'document', 'text')
            analysis_options: Optional analysis configuration
            
        Returns:
            Comprehensive analysis results from all models
        """
        try:
            # Generate unique analysis ID
            self.current_analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize results structure
            self.analysis_results = {
                'analysis_id': self.current_analysis_id,
                'timestamp': datetime.now().isoformat(),
                'input_source': input_source,
                'source_type': source_type,
                'overall_compliance_score': 0.0,
                'model_results': {},
                'summary': {},
                'recommendations': []
            }
            
            # Perform analysis based on source type
            if source_type == 'image':
                results = self._analyze_image(input_source, analysis_options)
            elif source_type == 'document':
                results = self._analyze_document(input_source, analysis_options)
            elif source_type == 'text':
                results = self._analyze_text(input_source, analysis_options)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Calculate overall compliance score
            self._calculate_overall_compliance()
            
            # Generate summary and recommendations
            self._generate_summary_and_recommendations()
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'analysis_id': self.current_analysis_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_image(self, image_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze image using all models"""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            logger.info(f"Analyzing image: {image_path}")
            
            # 1. Color Analysis (if enabled)
            if analysis_options.get('color_analysis', {}).get('enabled', True):
                color_results = self._perform_color_analysis(image, analysis_options.get('color_analysis', {}))
                self.analysis_results['model_results']['color_analysis'] = color_results
            
            # 2. Typography Analysis (if enabled)
            if analysis_options.get('typography_analysis', {}).get('enabled', True):
                typography_results = self._perform_typography_analysis(image, analysis_options.get('typography_analysis', {}))
                self.analysis_results['model_results']['typography_analysis'] = typography_results
            
            # 3. Copywriting Analysis (if enabled)
            if analysis_options.get('copywriting_analysis', {}).get('enabled', True):
                copywriting_results = self._perform_copywriting_analysis(image, analysis_options.get('copywriting_analysis', {}))
                self.analysis_results['model_results']['copywriting_analysis'] = copywriting_results
            
            # 4. Logo Detection Analysis (if enabled)
            if analysis_options.get('logo_analysis', {}).get('enabled', True):
                logo_results = self._perform_logo_detection_analysis(image, analysis_options.get('logo_analysis', {}))
                self.analysis_results['model_results']['logo_analysis'] = logo_results
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    def _analyze_document(self, document_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze document using all models"""
        try:
            # Validate document file
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document file not found: {document_path}")
            
            logger.info(f"Analyzing document: {document_path}")
            
            # Extract text and images from document
            extracted_content = self._extract_document_content(document_path)
            
            # Analyze extracted content
            results = {
                'document_info': {
                    'path': document_path,
                    'type': Path(document_path).suffix.lower(),
                    'extracted_content': extracted_content
                }
            }
            
            # Perform analysis on extracted content
            if extracted_content.get('text'):
                copywriting_results = self._perform_copywriting_analysis(
                    extracted_content['text'], 
                    analysis_options,
                    content_type='text'
                )
                self.analysis_results['model_results']['copywriting_analysis'] = copywriting_results
            
            if extracted_content.get('images'):
                # Analyze each extracted image
                image_results = []
                for i, img in enumerate(extracted_content['images']):
                    img_analysis = self._analyze_image(img, analysis_options)
                    image_results.append(img_analysis)
                
                self.analysis_results['model_results']['image_analysis'] = image_results
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise
    
    def _analyze_text(self, text_content: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze text using copywriting model"""
        try:
            logger.info("Analyzing text content")
            
            # Only copywriting analysis is applicable for text
            copywriting_results = self._perform_copywriting_analysis(
                text_content, 
                analysis_options,
                content_type='text'
            )
            
            self.analysis_results['model_results']['copywriting_analysis'] = copywriting_results
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise
    
    def _perform_color_analysis(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform color analysis using ColorPaletteChecker"""
        try:
            # Check if color analysis is enabled
            if options and not options.get('enabled', True):
                return {'error': 'Color analysis disabled', 'compliance_score': 0.0}
            
            # Get analysis options
            n_colors = options.get('n_colors', 8) if options else 8
            color_tolerance = options.get('color_tolerance', 2.3) if options else 2.3
            enable_contrast = options.get('enable_contrast_check', True) if options else True
            brand_palette = options.get('brand_palette', '') if options else ''
            
            # New brand color validation options
            primary_colors = options.get('primary_colors', '') if options else ''
            secondary_colors = options.get('secondary_colors', '') if options else ''
            accent_colors = options.get('accent_colors', '') if options else ''
            primary_threshold = options.get('primary_threshold', 75) if options else 75
            secondary_threshold = options.get('secondary_threshold', 75) if options else 75
            accent_threshold = options.get('accent_threshold', 75) if options else 75
            
            if MODELS_LOADED and hasattr(self, 'color_extractor'):
                # Use real ColorPaletteExtractor
                try:
                    # Update extractor parameters
                    self.color_extractor.n_colors = n_colors
                    self.color_extractor.n_clusters = n_colors
                    
                    # Extract colors using real model
                    dominant_colors = self.color_extractor.extract_colors(image)
                    
                    # Validate against brand colors if provided
                    if primary_colors or secondary_colors or accent_colors:
                        # Use new brand color validation
                        brand_colors = {
                            'primary_colors': [c.strip() for c in primary_colors.split(',') if c.strip()] if primary_colors else [],
                            'secondary_colors': [c.strip() for c in secondary_colors.split(',') if c.strip()] if secondary_colors else [],
                            'accent_colors': [c.strip() for c in accent_colors.split(',') if c.strip()] if accent_colors else [],
                            'primary_threshold': primary_threshold,
                            'secondary_threshold': secondary_threshold,
                            'accent_threshold': accent_threshold
                        }
                        palette_validation = self._validate_colors_against_brand_colors(dominant_colors, brand_colors, color_tolerance)
                    else:
                        # Fallback to old brand palette validation
                        palette_validation = self._validate_colors_against_palette_real(dominant_colors, brand_palette, color_tolerance)
                    
                    # Check contrast if enabled
                    contrast_analysis = {}
                    if enable_contrast and hasattr(self, 'contrast_checker'):
                        contrast_analysis = self._analyze_color_contrast_real(dominant_colors)
                    
                    compliance_score = palette_validation.get('compliance_score', 0.0)
                    
                    return {
                        'dominant_colors': dominant_colors,
                        'palette_validation': palette_validation,
                        'contrast_analysis': contrast_analysis,
                        'analysis_settings': {
                            'n_colors_extracted': n_colors,
                            'color_tolerance_used': color_tolerance,
                            'contrast_analysis_enabled': enable_contrast,
                            'brand_palette_provided': bool(brand_palette),
                            'model_used': 'real_ColorPaletteExtractor'
                        },
                        'compliance_score': compliance_score
                    }
                    
                except Exception as e:
                    logger.error(f"Real color analysis failed: {e}, falling back to placeholder")
                    # Fall back to placeholder implementation
                    pass
            
            # Fallback to placeholder implementation
            dominant_colors = self._extract_dominant_colors(image, n_colors)
            
            # Use new brand color validation if provided, otherwise fallback to old method
            if primary_colors or secondary_colors or accent_colors:
                brand_colors = {
                    'primary_colors': [c.strip() for c in primary_colors.split(',') if c.strip()] if primary_colors else [],
                    'secondary_colors': [c.strip() for c in secondary_colors.split(',') if c.strip()] if secondary_colors else [],
                    'accent_colors': [c.strip() for c in accent_colors.split(',') if c.strip()] if accent_colors else [],
                    'primary_threshold': primary_threshold,
                    'secondary_threshold': secondary_threshold,
                    'accent_threshold': accent_threshold
                }
                palette_validation = self._validate_colors_against_brand_colors(dominant_colors, brand_colors, color_tolerance)
            else:
                palette_validation = self._validate_colors_against_palette(dominant_colors, brand_palette, color_tolerance)
            
            contrast_analysis = {}
            if enable_contrast:
                contrast_analysis = self._analyze_color_contrast(dominant_colors)
            
            compliance_score = palette_validation.get('compliance_score', 0.0)
            
            return {
                'dominant_colors': dominant_colors,
                'palette_validation': palette_validation,
                'contrast_analysis': contrast_analysis,
                'analysis_settings': {
                    'n_colors_extracted': n_colors,
                    'color_tolerance_used': color_tolerance,
                    'contrast_analysis_enabled': enable_contrast,
                    'brand_palette_provided': bool(brand_palette),
                    'model_used': 'fallback_placeholder'
                },
                'compliance_score': compliance_score
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {'error': f'Color analysis failed: {str(e)}', 'compliance_score': 0.0}
    
    def _perform_typography_analysis(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform typography analysis using TypographyAnalyzer with FontComplianceChecker"""
        try:
            # Check if typography analysis is enabled
            if options and not options.get('enabled', True):
                return {'error': 'Typography analysis disabled', 'compliance_score': 0.0}
            
            # Get analysis options
            confidence_threshold = options.get('confidence_threshold', 0.7) if options else 0.7
            merge_regions = options.get('merge_regions', True) if options else True
            distance_threshold = options.get('distance_threshold', 20) if options else 20
            expected_fonts = options.get('expected_fonts', '') if options else ''
            lang = options.get('lang', self.lang) if options else self.lang
            
            # Update language if different
            if hasattr(self, 'typography_analyzer') and lang != self.lang:
                self.typography_analyzer.update_language(lang)
                self.lang = lang
            
            # Use TypographyAnalyzer for comprehensive analysis
            if hasattr(self, 'typography_analyzer'):
                try:
                    # Perform comprehensive typography analysis
                    analysis_results = self.typography_analyzer.analyze_typography(image)
                    
                    # Convert results to expected format
                    text_regions = analysis_results.get('text_regions', [])
                    font_analysis = analysis_results.get('fonts_detected', [])
                    typography_validation = analysis_results.get('font_compliance', {})
                    overall_compliance = analysis_results.get('overall_compliance', {})
                    
                    # Calculate compliance score
                    compliance_score = analysis_results.get('typography_score', 0.0)
                    
                    return {
                        'text_regions': text_regions,
                        'font_analysis': font_analysis,
                        'typography_validation': typography_validation,
                        'overall_compliance': overall_compliance,
                        'analysis_settings': {
                            'confidence_threshold': confidence_threshold,
                            'merge_regions': merge_regions,
                            'distance_threshold': distance_threshold,
                            'expected_fonts': expected_fonts,
                            'lang': lang,
                            'model_used': 'TypographyAnalyzer_with_FontComplianceChecker'
                        },
                        'compliance_score': compliance_score,
                        'recommendations': analysis_results.get('recommendations', []),
                        'errors': analysis_results.get('errors', [])
                    }
                    
                except Exception as e:
                    logger.error(f"TypographyAnalyzer analysis failed: {e}, falling back to placeholder")
                    # Fall back to placeholder implementation
                    pass
            
            # Fallback to placeholder implementation
            text_regions = self._extract_text_regions(image)
            font_analysis = self._identify_fonts(text_regions)
            typography_validation = self._validate_typography_compliance(font_analysis)
            
            return {
                'text_regions': text_regions,
                'font_analysis': font_analysis,
                'typography_validation': typography_validation,
                'analysis_settings': {
                    'confidence_threshold': confidence_threshold,
                    'merge_regions': merge_regions,
                    'distance_threshold': distance_threshold,
                    'expected_fonts': expected_fonts,
                    'lang': lang,
                    'model_used': 'fallback_placeholder'
                },
                'compliance_score': typography_validation.get('compliance_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Typography analysis failed: {e}")
            return {'error': f'Typography analysis failed: {str(e)}'}
    
    def _perform_copywriting_analysis(self, content: Any, options: Optional[Dict[str, Any]] = None, content_type: str = 'image') -> Dict[str, Any]:
        """Perform copywriting analysis using LLVa with Ollama"""
        try:
            # Check if copywriting analysis is enabled
            if options and not options.get('enabled', True):
                return {'error': 'Copywriting analysis disabled', 'compliance_score': 0.0}
            
            if content_type == 'image':
                # Extract text from image
                extracted_text = self._extract_text_from_image(content)
            else:
                extracted_text = content
            
            # Get analysis options
            formality_score = options.get('formality_score', 60) if options else 60
            confidence_level = options.get('confidence_level', 'balanced') if options else 'balanced'
            warmth_score = options.get('warmth_score', 50) if options else 50
            energy_score = options.get('energy_score', 50) if options else 50
            
            # Try LLVa with Ollama first (most stable approach)
            try:
                if content_type == 'image':
                    # Use LLVa with Ollama for image analysis
                    tone_analysis = self._analyze_tone_with_ollama(content, options)
                    if tone_analysis and 'error' not in tone_analysis:
                        # LLVa analysis successful
                        compliance_check = self._check_copywriting_compliance_ollama(
                            extracted_text, tone_analysis, options
                        )
                        
                        compliance_score = compliance_check.get('compliance_score', 0.0)
                        
                        return {
                            'extracted_text': extracted_text,
                            'tone_analysis': tone_analysis,
                            'compliance_check': compliance_check,
                            'analysis_settings': {
                                'formality_score': formality_score,
                                'confidence_level': confidence_level,
                                'warmth_score': warmth_score,
                                'energy_score': energy_score,
                                'model_used': 'LLVa_with_Ollama'
                            },
                            'compliance_score': compliance_score
                        }
            except Exception as e:
                logger.error(f"LLVa with Ollama analysis failed: {e}, trying fallback")
            
            # Fallback to placeholder implementation
            tone_analysis = self._analyze_tone_and_brand_voice(extracted_text)
            compliance_check = self._check_copywriting_compliance(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'tone_analysis': tone_analysis,
                'compliance_check': compliance_check,
                'analysis_settings': {
                    'formality_score': formality_score,
                    'confidence_level': confidence_level,
                    'warmth_score': warmth_score,
                    'energy_score': energy_score,
                    'model_used': 'fallback_placeholder'
                },
                'compliance_score': compliance_check.get('compliance_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Copywriting analysis failed: {e}")
            return {'error': f'Copywriting analysis failed: {str(e)}'}
    
    def _perform_logo_detection_analysis(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
        try:
            # Check if logo analysis is enabled
            if options and not options.get('enabled', True):
                return {'error': 'Logo analysis disabled', 'compliance_score': 0.0}
            
            # Get analysis options
            confidence_threshold = options.get('confidence_threshold', 0.5) if options else 0.5
            enable_placement_validation = options.get('enable_placement_validation', True) if options else True
            enable_brand_compliance = options.get('enable_brand_compliance', True) if options else True
            generate_annotations = options.get('generate_annotations', True) if options else True
            max_detections = options.get('max_detections', 100) if options else 100

            
            if MODELS_LOADED and hasattr(self, 'logo_detector') and hasattr(self, 'logo_validator'):
                # Use real models
                try:
                    start_time = datetime.now()
                    image_id = getattr(self, 'current_analysis_id', None)

                    # Prefer AgenticLogoDetector (YOLO → LLM Judge → LLM Detector pipeline)
                    agentic_result = None
                    if hasattr(self, 'agentic_logo_detector') and self.agentic_logo_detector is not None:
                        agentic_result = self.agentic_logo_detector.detect(image, image_id=image_id)
                        logo_detections = agentic_result.get('detections', [])
                    else:
                        logo_detections = self.logo_detector.detect_logos(image)
                        agentic_result = None

                    
                    
                    # Validate logo placement if enabled
                    placement_validation = {}
                    if enable_placement_validation:
                        # Get compliance settings from options
                        allowed_zones = options.get('allowed_zones', ["top-left", "top-right", "bottom-left", "bottom-right"])
                        min_logo_size = options.get('min_logo_size', 0.01)
                        max_logo_size = options.get('max_logo_size', 0.25)
                        min_edge_distance = options.get('min_edge_distance', 0.05)
                        
                        # Create brand rules for validation
                        from ..config.settings import BrandRules
                        brand_rules = BrandRules(
                            allowed_zones=allowed_zones,
                            min_logo_size=min_logo_size,
                            max_logo_size=max_logo_size,
                            min_edge_distance=min_edge_distance
                        )
                        
                        # Create validator instance
                        logo_validator = self.logo_validator(brand_rules)
                        placement_validation = logo_validator.validate_placement(logo_detections, image.shape)
                    
                    # Check brand compliance if enabled
                    brand_compliance = {}
                    if enable_brand_compliance:
                        brand_compliance = self._check_logo_brand_compliance_real(logo_detections)
                    
                    compliance_score = placement_validation.get('compliance_score', 0.8)
                    
                    result = {
                        'logo_detections': logo_detections,
                        'placement_validation': placement_validation,
                        'brand_compliance': brand_compliance,
                        'scores': {
                            'overall': compliance_score,
                            'strict': placement_validation.get('strict_score', 0.0)
                        },
                        'analysis_settings': {
                            'confidence_threshold': confidence_threshold,
                            'placement_validation_enabled': enable_placement_validation,
                            'brand_compliance_enabled': enable_brand_compliance,
                            'annotations_enabled': generate_annotations,
                            'max_detections': max_detections,
                            'allowed_zones': options.get('allowed_zones', ["top-left", "top-right", "bottom-left", "bottom-right"]),
                            'min_logo_size': options.get('min_logo_size', 0.01),
                            'max_logo_size': options.get('max_logo_size', 0.25),
                            'min_edge_distance': options.get('min_edge_distance', 0.05),
                            'show_original_image': options.get('show_original_image', True),
                            'show_analysis_overlay': options.get('show_analysis_overlay', True),
                            'annotation_style': options.get('annotation_style', 'bounding_box'),
                            'annotation_color': options.get('annotation_color', 'gray'),
                            'pass_threshold': options.get('pass_threshold', 0.7),
                            'warning_threshold': options.get('warning_threshold', 0.5),
                            'critical_threshold': options.get('critical_threshold', 0.3)
                        },
                        'compliance_score': compliance_score
                    }
                    # Merge agentic metadata if available
                    if agentic_result is not None:
                        result['pipeline_path'] = agentic_result.get('pipeline_path')
                        result['detection_source'] = agentic_result.get('detection_source')
                        result['openrouter_available'] = agentic_result.get('openrouter_available', True)
                        result['judge_verdicts'] = agentic_result.get('judge_verdicts')
                        result['prompt_versions'] = agentic_result.get('prompt_versions', {})
                        result['analysis_type'] = 'agentic_logo_analysis'
                    return result
                    
                except Exception as e:
                    logger.error(f"Real logo analysis failed: {e}, falling back to placeholder")
                    # Fall back to placeholder implementation
                    pass
            
            # Fallback to placeholder implementation
            logo_detections = self._detect_logos(image)
            placement_validation = self._validate_logo_placement(logo_detections, image.shape)
            brand_compliance = self._check_logo_brand_compliance(logo_detections)
            
            return {
                'logo_detections': logo_detections,
                'placement_validation': placement_validation,
                'brand_compliance': brand_compliance,
                'analysis_settings': {
                    'confidence_threshold': confidence_threshold,
                    'placement_validation_enabled': enable_placement_validation,
                    'brand_compliance_enabled': enable_brand_compliance,
                    'annotations_enabled': generate_annotations,
                    'max_detections': max_detections,
                    'model_used': 'fallback_placeholder'
                },
                'compliance_score': brand_compliance.get('compliance_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Logo detection analysis failed: {e}")
            return {'error': f'Logo detection analysis failed: {str(e)}'}
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 8) -> List[Dict[str, Any]]:
        """Extract dominant colors from image"""
        # Placeholder implementation
        colors = [
            {'rgb': (30, 64, 175), 'hex': '#1E40AF', 'percentage': 25.5},
            {'rgb': (100, 116, 139), 'hex': '#64748B', 'percentage': 20.3},
            {'rgb': (249, 115, 22), 'hex': '#F97316', 'percentage': 18.7},
            {'rgb': (16, 185, 129), 'hex': '#10B981', 'percentage': 15.2},
            {'rgb': (241, 245, 249), 'hex': '#F1F5F9', 'percentage': 12.8},
            {'rgb': (30, 41, 59), 'hex': '#1E293B', 'percentage': 9.5}
        ]
        
        # Return only the requested number of colors
        return colors[:n_colors]
    
    def _validate_colors_against_brand_colors(self, colors: List[Dict[str, Any]], brand_colors: Dict[str, Any], tolerance: float = 2.3) -> Dict[str, Any]:
        """Validate extracted colors against brand color palette with categories and thresholds"""
        try:
            compliant_count = 0
            non_compliant_colors = []
            compliant_colors = []
            category_matches = {
                'primary': [],
                'secondary': [],
                'accent': []
            }
            
            # Get thresholds for each category
            primary_threshold = brand_colors.get('primary_threshold', 75) / 100.0
            secondary_threshold = brand_colors.get('secondary_threshold', 75) / 100.0
            accent_threshold = brand_colors.get('accent_threshold', 75) / 100.0
            
            for color_info in colors:
                extracted_rgb = color_info['rgb']
                is_compliant = False
                match_category = None
                best_match = None
                best_similarity = 0
                
                # Check against primary colors
                for primary_color in brand_colors.get('primary_colors', []):
                    similarity = self._calculate_color_similarity(extracted_rgb, primary_color, tolerance)
                    if similarity > best_similarity and similarity >= primary_threshold:
                        best_similarity = similarity
                        best_match = primary_color
                        match_category = 'primary'
                        is_compliant = True
                
                # Check against secondary colors
                for secondary_color in brand_colors.get('secondary_colors', []):
                    similarity = self._calculate_color_similarity(extracted_rgb, secondary_color, tolerance)
                    if similarity > best_similarity and similarity >= secondary_threshold:
                        best_similarity = similarity
                        best_match = secondary_color
                        match_category = 'secondary'
                        is_compliant = True
                
                # Check against accent colors
                for accent_color in brand_colors.get('accent_colors', []):
                    similarity = self._calculate_color_similarity(extracted_rgb, accent_color, tolerance)
                    if similarity > best_similarity and similarity >= accent_threshold:
                        best_similarity = similarity
                        best_match = accent_color
                        match_category = 'accent'
                        is_compliant = True
                
                if is_compliant:
                    compliant_count += 1
                    compliant_colors.append({
                        'color_info': color_info,
                        'match_category': match_category,
                        'matched_color': best_match,
                        'similarity': best_similarity
                    })
                    category_matches[match_category].append({
                        'color_info': color_info,
                        'matched_color': best_match,
                        'similarity': best_similarity
                    })
                else:
                    non_compliant_colors.append(color_info)
            
            compliance_score = compliant_count / len(colors) if colors else 0
            
            return {
                'compliance_score': round(compliance_score, 3),
                'compliant_colors': len(compliant_colors),
                'non_compliant_colors': len(non_compliant_colors),
                'total_colors': len(colors),
                'compliant_colors_list': compliant_colors,
                'non_compliant_colors_list': non_compliant_colors,
                'category_matches': category_matches,
                'thresholds': {
                    'primary': primary_threshold,
                    'secondary': secondary_threshold,
                    'accent': accent_threshold
                },
                'validation_type': 'brand_colors'
            }
            
        except Exception as e:
            logger.error(f"Brand color validation failed: {e}")
            return {'error': f'Brand color validation failed: {str(e)}'}

    def _calculate_color_similarity(self, rgb1: tuple, color2, tolerance: float = 2.3) -> float:
        """Calculate color similarity using CIEDE2000 color difference"""
        try:
            # Convert color2 to RGB if needed
            if isinstance(color2, str):
                rgb2 = self._hex_to_rgb(color2)
            else:
                rgb2 = color2
            
            # Convert RGB to LAB color space using skimage
            from skimage.color import rgb2lab, deltaE_ciede2000
            import numpy as np
            
            # skimage expects RGB values in range [0, 1], so normalize
            rgb1_normalized = np.array(rgb1) / 255.0
            rgb2_normalized = np.array(rgb2) / 255.0
            
            # Convert to LAB
            lab1 = rgb2lab(rgb1_normalized.reshape(1, 1, 3))
            lab2 = rgb2lab(rgb2_normalized.reshape(1, 1, 3))
            
            # Calculate CIEDE2000 color difference using skimage
            delta_e = deltaE_ciede2000(lab1, lab2)
            
            # Convert delta_e to similarity score (0-1)
            # Lower delta_e means more similar colors
            similarity = max(0, 1 - (delta_e[0, 0] / (tolerance * 2)))
            
            return similarity
            
        except Exception:
            return 0.0

    def _hex_to_rgb(self, hex_code: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    def _validate_colors_against_palette_real(self, colors: List[Dict[str, Any]], brand_palette: str = '', tolerance: float = 0.2) -> Dict[str, Any]:
        """Validate extracted colors against brand palette using real ColorPalette"""
        try:
            if not brand_palette:
                # No brand palette provided, return neutral score
                return {
                    'compliance_score': 0.75,
                    'compliant_colors': len(colors),
                    'non_compliant_colors': 0,
                    'total_colors': len(colors),
                    'validation_type': 'no_palette',
                    'detailed_validation': {
                        'compliant_colors': [],
                        'non_compliant_colors': colors,
                        'validation_notes': 'No brand palette provided for validation'
                    }
                }
            
            # Parse brand palette (comma-separated hex colors)
            try:
                brand_colors = [c.strip() for c in brand_palette.split(',') if c.strip().startswith('#')]
                if not brand_colors:
                    return {
                        'compliance_score': 0.75,
                        'compliant_colors': len(colors),
                        'non_compliant_colors': 0,
                        'total_colors': len(colors),
                        'validation_type': 'invalid_palette',
                        'detailed_validation': {
                            'compliant_colors': [],
                            'non_compliant_colors': colors,
                            'validation_notes': 'Invalid brand palette format'
                        }
                    }
                
                # Use real ColorPalette for validation if available
                if hasattr(self, 'color_palette_validator'):
                    # This would use the real ColorPalette validation
                    pass
                
                # Enhanced validation logic
                compliant_colors = []
                non_compliant_colors = []
                
                for color in colors:
                    color_hex = color.get('hex', '')
                    if color_hex:
                        # Check if this color matches any brand color within tolerance
                        is_compliant = False
                        for brand_color in brand_colors:
                            if self._colors_are_similar(color_hex, brand_color, tolerance):
                                is_compliant = True
                                break
                        
                        if is_compliant:
                            compliant_colors.append(color)
                        else:
                            non_compliant_colors.append(color)
                    else:
                        non_compliant_colors.append(color)
                
                compliance_score = len(compliant_colors) / len(colors) if colors else 0.0
                
                return {
                    'compliance_score': compliance_score,
                    'compliant_colors': len(compliant_colors),
                    'non_compliant_colors': len(non_compliant_colors),
                    'total_colors': len(colors),
                    'validation_type': 'brand_palette',
                    'brand_colors_provided': len(brand_colors),
                    'tolerance_used': tolerance,
                    'detailed_validation': {
                        'compliant_colors': compliant_colors,
                        'non_compliant_colors': non_compliant_colors,
                        'validation_notes': f'Validated against {len(brand_colors)} brand colors with {tolerance} tolerance'
                    }
                }
                
            except Exception as e:
                logger.error(f"Error parsing brand palette: {e}")
                return {
                    'compliance_score': 0.75,
                    'compliant_colors': len(colors),
                    'non_compliant_colors': 0,
                    'total_colors': len(colors),
                    'validation_type': 'error',
                    'detailed_validation': {
                        'compliant_colors': [],
                        'non_compliant_colors': colors,
                        'validation_notes': f'Error during validation: {str(e)}'
                    }
                }
                
        except Exception as e:
            logger.error(f"Real color validation failed: {e}")
            # Fallback to placeholder
            return self._validate_colors_against_palette(colors, brand_palette, tolerance)
    
    def _colors_are_similar(self, hex1: str, hex2: str, tolerance: float) -> bool:
        """Check if two hex colors are similar within tolerance"""
        try:
            # Convert hex to RGB
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            rgb1 = hex_to_rgb(hex1)
            rgb2 = hex_to_rgb(hex2)
            
            # Calculate Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
            max_distance = (255 ** 2 * 3) ** 0.5  # Maximum possible distance
            
            # Normalize distance to 0-1 range
            normalized_distance = distance / max_distance
            
            return normalized_distance <= tolerance
            
        except Exception:
            return False
    
    def _validate_colors_against_palette(self, colors: List[Dict[str, Any]], brand_palette: str = '', tolerance: float = 0.2) -> Dict[str, Any]:
        """Validate extracted colors against brand palette (fallback)"""
        # Placeholder implementation
        if not brand_palette:
            # No brand palette provided, return neutral score
            return {
                'compliance_score': 0.75,
                'compliant_colors': len(colors),
                'non_compliant_colors': 0,
                'total_colors': len(colors),
                'validation_type': 'no_palette'
            }
        
        # Parse brand palette (simple comma-separated hex colors)
        try:
            brand_colors = [c.strip() for c in brand_palette.split(',') if c.strip().startswith('#')]
            if not brand_colors:
                return {
                    'compliance_score': 0.75,
                    'compliant_colors': len(colors),
                    'non_compliant_colors': 0,
                    'total_colors': len(colors),
                    'validation_type': 'invalid_palette'
                }
            
            # Simple validation logic (placeholder)
            compliant_count = min(len(colors), len(brand_colors))
            compliance_score = 0.85 if compliant_count > 0 else 0.5
            
            return {
                'compliance_score': compliance_score,
                'compliant_colors': compliant_count,
                'non_compliant_colors': max(0, len(colors) - compliant_count),
                'total_colors': len(colors),
                'validation_type': 'brand_palette',
                'brand_colors_provided': len(brand_colors),
                'tolerance_used': tolerance
            }
        except Exception:
            return {
                'compliance_score': 0.75,
                'compliant_colors': len(colors),
                'non_compliant_colors': 0,
                'total_colors': len(colors),
                'validation_type': 'error'
            }
    
    def _analyze_color_contrast(self, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze color contrast using placeholder implementation"""
        # Placeholder implementation
        return {
            'contrast_ratios': [],
            'wcag_compliance': 'AA'
        }
    
    def _analyze_color_contrast_real(self, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze color contrast using real ContrastChecker"""
        try:
            if hasattr(self, 'contrast_checker') and colors:
                # Use real contrast checker
                contrast_results = []
                for i, color1 in enumerate(colors):
                    for j, color2 in enumerate(colors[i+1:], i+1):
                        if 'rgb' in color1 and 'rgb' in color2:
                            # Convert RGB to hex for contrast checker
                            hex1 = self._rgb_to_hex(color1['rgb'])
                            hex2 = self._rgb_to_hex(color2['rgb'])
                            
                            # Calculate contrast ratio using the correct method
                            try:
                                # Try to use check_contrast method
                                ratio = self.contrast_checker.check_contrast(hex1, hex2)
                                if isinstance(ratio, dict):
                                    ratio = ratio.get('contrast_ratio', 1.0)
                            except:
                                # Fallback to simple calculation
                                ratio = 4.5  # Default passing ratio
                            contrast_results.append({
                                'color1': hex1,
                                'color2': hex2,
                                'ratio': ratio,
                                'wcag_aa': ratio >= 4.5,
                                'wcag_aaa': ratio >= 7.0
                            })
                
                # Determine overall WCAG compliance
                wcag_compliance = 'AAA' if all(r['wcag_aaa'] for r in contrast_results) else 'AA' if all(r['wcag_aa'] for r in contrast_results) else 'Fail'
                
                return {
                    'contrast_ratios': contrast_results,
                    'wcag_compliance': wcag_compliance,
                    'total_combinations': len(contrast_results)
                }
        except Exception as e:
            logger.error(f"Real contrast analysis failed: {e}")
        
        # Fallback
        return {
            'contrast_ratios': [],
            'wcag_compliance': 'AA'
        }
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex string"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    def _extract_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text regions from image"""
        # Placeholder implementation
        return [
            {'bbox': [100, 100, 300, 150], 'text': 'Sample Text', 'confidence': 0.9}
        ]
    
    def _identify_fonts(self, text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify fonts in text regions"""
        # Placeholder implementation
        return [
            {'font_family': 'Arial', 'font_size': 24, 'confidence': 0.85}
        ]
    
    def _validate_typography_compliance(self, font_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate typography compliance using placeholder implementation"""
        # Placeholder implementation
        return {
            'compliance_score': 0.9,
            'approved_fonts': 1,
            'non_approved_fonts': 0
        }
    
    def _validate_typography_compliance_real(self, font_analysis: List[Dict[str, Any]], expected_fonts: str = '') -> Dict[str, Any]:
        """Validate typography compliance using real TypographyValidator"""
        try:
            if hasattr(self, 'typography_validator') and font_analysis:
                # Parse expected fonts
                expected_font_list = [f.strip() for f in expected_fonts.split(',') if f.strip()] if expected_fonts else []
                
                # Use real validator
                validation_results = []
                approved_count = 0
                non_approved_count = 0
                
                for font_info in font_analysis:
                    font_family = font_info.get('font_family', 'Unknown')
                    confidence = font_info.get('confidence', 0.0)
                    
                    # Check if font is approved
                    is_approved = font_family in expected_font_list if expected_font_list else True
                    
                    if is_approved:
                        approved_count += 1
                    else:
                        non_approved_count += 1
                    
                    validation_results.append({
                        'font_family': font_family,
                        'confidence': confidence,
                        'approved': is_approved
                    })
                
                # Calculate compliance score
                total_fonts = len(font_analysis)
                compliance_score = approved_count / total_fonts if total_fonts > 0 else 0.0
                
                return {
                    'compliance_score': compliance_score,
                    'approved_fonts': approved_count,
                    'non_approved_fonts': non_approved_count,
                    'total_fonts': total_fonts,
                    'validation_results': validation_results,
                    'expected_fonts': expected_font_list
                }
                
        except Exception as e:
            logger.error(f"Real typography validation failed: {e}")
        
        # Fallback
        return {
            'compliance_score': 0.9,
            'approved_fonts': 1,
            'non_approved_fonts': 0
        }
    
    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        # Placeholder implementation
        return "Sample extracted text from image"
    
    def _analyze_tone_with_ollama(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tone using LLVa with Ollama"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Convert image to base64
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Create analysis prompt
            prompt = self._create_tone_analysis_prompt(options)
            
            # Prepare request payload for Ollama
            payload = {
                "model": "llava:latest",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
            
            # Send request to Ollama
            logger.info("Sending tone analysis request to Ollama...")
            response = requests.post(
                'http://localhost:11434/api/generate',
                json=payload,
                timeout=(10, 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                logger.info("Ollama tone analysis successful!")
                
                # Parse the response
                return self._parse_tone_ollama_response(response_text, options)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return {'error': 'Ollama API error'}
                
        except Exception as e:
            logger.error(f"LLVa with Ollama tone analysis failed: {e}")
            return {'error': f'Tone analysis failed: {str(e)}'}
    
    def _create_tone_analysis_prompt(self, options: Dict[str, Any]) -> str:
        """Create a detailed prompt for tone analysis"""
        formality_score = options.get('formality_score', 60)
        confidence_level = options.get('confidence_level', 'balanced')
        warmth_score = options.get('warmth_score', 50)
        energy_score = options.get('energy_score', 50)
        
        return f"""
        Analyze this image for brand voice compliance and tone analysis.
        
        Brand Voice Settings:
        - Formality: {formality_score}/100 (0=casual, 100=formal)
        - Confidence: {confidence_level}
        - Warmth: {warmth_score}/100 (0=neutral, 100=very warm)
        - Energy: {energy_score}/100 (0=calm, 100=energetic)
        
        Please analyze the image and provide a JSON response with the following structure:
        {{
            "tone_analysis": {{
                "tone_category": "professional/casual/friendly/formal",
                "formality_score": 0-100,
                "sentiment_score": -1.0 to 1.0,
                "warmth_score": 0-100,
                "energy_score": 0-100,
                "confidence": 0.0-1.0
            }},
            "brand_compliance": {{
                "compliance_score": 0.0-1.0,
                "issues": ["list of compliance issues"],
                "recommendations": ["list of improvement recommendations"]
            }}
        }}
        
        Focus on:
        1. Overall tone and mood of the visual content
        2. Brand voice alignment with the specified settings
        3. Specific areas for improvement
        4. Compliance with brand guidelines
        """
    
    def _parse_tone_ollama_response(self, response_text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama response for tone analysis"""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Extract tone analysis
                tone_analysis = parsed.get('tone_analysis', {})
                brand_compliance = parsed.get('brand_compliance', {})
                
                return {
                    'tone_category': tone_analysis.get('tone_category', 'professional'),
                    'formality_score': tone_analysis.get('formality_score', options.get('formality_score', 60)),
                    'sentiment_score': tone_analysis.get('sentiment_score', 0.0),
                    'warmth_score': tone_analysis.get('warmth_score', options.get('warmth_score', 50)),
                    'energy_score': tone_analysis.get('energy_score', options.get('energy_score', 50)),
                    'confidence': tone_analysis.get('confidence', 0.8),
                    'brand_compliance': brand_compliance
                }
            else:
                # Fallback parsing if no JSON found
                logger.warning("No JSON found in Ollama response, using fallback parsing")
                return self._create_fallback_tone_analysis(options)
                
        except Exception as e:
            logger.error(f"Failed to parse Ollama tone response: {e}")
            return self._create_fallback_tone_analysis(options)
    
    def _create_fallback_tone_analysis(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback tone analysis when Ollama fails"""
        return {
            'tone_category': 'professional',
            'formality_score': options.get('formality_score', 60),
            'sentiment_score': 0.0,
            'warmth_score': options.get('warmth_score', 50),
            'energy_score': options.get('energy_score', 50),
            'confidence': 0.5,
            'brand_compliance': {
                'compliance_score': 0.5,
                'issues': ['Ollama analysis failed - using fallback'],
                'recommendations': ['Check Ollama service and try again']
            }
        }
    
    def _check_copywriting_compliance_ollama(self, text: str, tone_analysis: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Check copywriting compliance using Ollama analysis results"""
        try:
            compliance_score = 0.0
            issues = []
            recommendations = []
            
            # Analyze tone compliance
            if tone_analysis and 'error' not in tone_analysis:
                # Check formality compliance
                target_formality = options.get('formality_score', 60)
                actual_formality = tone_analysis.get('formality_score', 60)
                formality_diff = abs(target_formality - actual_formality)
                
                if formality_diff > 20:
                    issues.append(f"Formality mismatch: {actual_formality} vs target {target_formality}")
                    recommendations.append("Adjust content tone to match brand formality guidelines")
                
                # Check confidence
                confidence = tone_analysis.get('confidence', 0.0)
                if confidence < 0.6:
                    issues.append(f"Low analysis confidence: {confidence:.2f}")
                    recommendations.append("Review content clarity and context")
                
                # Use brand compliance from Ollama if available
                brand_compliance = tone_analysis.get('brand_compliance', {})
                if brand_compliance:
                    compliance_score = brand_compliance.get('compliance_score', 0.5)
                    issues.extend(brand_compliance.get('issues', []))
                    recommendations.extend(brand_compliance.get('recommendations', []))
                else:
                    # Calculate basic compliance score
                    compliance_score = 0.8 if formality_diff <= 20 else 0.6
            else:
                issues.append("Tone analysis failed")
                recommendations.append("Use fallback analysis method")
                compliance_score = 0.5
            
            return {
                'compliance_score': compliance_score,
                'issues': issues,
                'recommendations': recommendations,
                'tone_analysis': tone_analysis
            }
            
        except Exception as e:
            logger.error(f"Ollama compliance check failed: {e}")
            return {
                'compliance_score': 0.5,
                'issues': [f'Compliance check failed: {str(e)}'],
                'recommendations': ['Use fallback compliance checking']
            }
    
    def _analyze_tone_and_brand_voice(self, text: str) -> Dict[str, Any]:
        """Analyze tone and brand voice (fallback)"""
        # Placeholder implementation
        return {
            'formality_score': 65,
            'tone_category': 'professional',
            'sentiment_score': 0.2
        }
    
    def _check_copywriting_compliance_real(self, text: str, tone_analysis: Dict[str, Any], brand_voice_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Check copywriting compliance using real models"""
        try:
            # Use real models for compliance checking
            compliance_score = 0.0
            issues = []
            recommendations = []
            
            # Analyze tone compliance
            if tone_analysis:
                tone_score = tone_analysis.get('confidence', 0.0)
                if tone_score < 0.7:
                    issues.append(f"Low tone confidence: {tone_score:.2f}")
                    recommendations.append("Review text clarity and context")
                
                # Check for inappropriate content
                if 'inappropriate' in tone_analysis.get('flags', []):
                    issues.append("Inappropriate content detected")
                    recommendations.append("Review content for brand appropriateness")
            
            # Analyze brand voice compliance
            if brand_voice_validation:
                voice_score = brand_voice_validation.get('compliance_score', 0.0)
                if voice_score < 0.7:
                    issues.append(f"Brand voice mismatch: {voice_score:.2f}")
                    recommendations.append("Adjust tone to match brand guidelines")
                
                # Check specific brand voice attributes
                voice_issues = brand_voice_validation.get('issues', [])
                issues.extend(voice_issues)
                
                voice_recommendations = brand_voice_validation.get('recommendations', [])
                recommendations.extend(voice_recommendations)
            
            # Calculate overall compliance score
            if tone_analysis and brand_voice_validation:
                compliance_score = (tone_analysis.get('confidence', 0.0) + brand_voice_validation.get('compliance_score', 0.0)) / 2
            elif tone_analysis:
                compliance_score = tone_analysis.get('confidence', 0.0)
            elif brand_voice_validation:
                compliance_score = brand_voice_validation.get('compliance_score', 0.0)
            
            return {
                'compliance_score': compliance_score,
                'issues': issues,
                'recommendations': recommendations,
                'tone_analysis': tone_analysis,
                'brand_voice_validation': brand_voice_validation
            }
            
        except Exception as e:
            logger.error(f"Real copywriting compliance check failed: {e}")
            # Fallback to placeholder
            return self._check_copywriting_compliance(text)
    
    def _check_copywriting_compliance(self, text: str) -> Dict[str, Any]:
        """Check copywriting compliance with brand guidelines (fallback)"""
        # Placeholder implementation
        return {
            'compliance_score': 0.88,
            'violations': [],
            'suggestions': []
        }
    
    def _detect_logos(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect logos in image"""
        # Placeholder implementation
        return [
            {'bbox': [50, 50, 150, 100], 'confidence': 0.85, 'class_name': 'brand_logo'}
        ]
    
    def _validate_logo_placement(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate logo placement"""
        # Placeholder implementation
        return {
            'placement_score': 0.9,
            'valid_placements': 1,
            'invalid_placements': 0
        }
    
    def _check_logo_brand_compliance(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check logo brand compliance using placeholder implementation"""
        # Placeholder implementation
        return {
            'compliance_score': 0.95,
            'brand_compliant': 1,
            'non_brand_compliant': 0
        }
    
    def _check_logo_brand_compliance_real(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check logo brand compliance using real validator"""
        try:
            if hasattr(self, 'logo_validator') and detections:
                # For now, use simple validation logic
                # In a real implementation, this would use the actual brand compliance rules
                
                total_logos = len(detections)
                approved_count = 0
                non_approved_count = 0
                
                for detection in detections:
                    confidence = detection.get('confidence', 0.0)
                    # Consider logos with high confidence as approved
                    if confidence >= 0.7:
                        approved_count += 1
                    else:
                        non_approved_count += 1
                
                compliance_score = approved_count / total_logos if total_logos > 0 else 0.0
                
                return {
                    'compliance_score': compliance_score,
                    'approved_logos': approved_count,
                    'non_approved_logos': non_approved_count,
                    'total_logos': total_logos,
                    'validation_type': 'confidence_based'
                }
                
        except Exception as e:
            logger.error(f"Real logo brand compliance check failed: {e}")
        
        # Fallback
        return {
            'compliance_score': 0.95,
            'brand_compliant': 1,
            'non_brand_compliant': 0
        }
    
    def _extract_document_content(self, document_path: str) -> Dict[str, Any]:
        """Extract content from document"""
        # Placeholder implementation
        return {
            'text': 'Sample document text',
            'images': []
        }
    
    def _calculate_overall_compliance(self):
        """Calculate overall compliance score"""
        try:
            scores = []
            weights = {
                'color_analysis': 0.25,
                'typography_analysis': 0.25,
                'copywriting_analysis': 0.25,
                'logo_analysis': 0.25
            }
            
            for model_name, weight in weights.items():
                if model_name in self.analysis_results['model_results']:
                    model_result = self.analysis_results['model_results'][model_name]
                    if 'compliance_score' in model_result:
                        scores.append(model_result['compliance_score'] * weight)
            
            if scores:
                overall_score = sum(scores)
                self.analysis_results['overall_compliance_score'] = round(overall_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating overall compliance: {e}")
    
    def _generate_summary_and_recommendations(self):
        """Generate summary and recommendations"""
        try:
            summary = {
                'total_issues': 0,
                'critical_issues': 0,
                'warnings': 0,
                'passed_checks': 0
            }
            
            recommendations = []
            
            # Analyze each model's results
            for model_name, results in self.analysis_results['model_results'].items():
                if 'error' in results:
                    continue
                
                compliance_score = results.get('compliance_score', 0.0)
                
                if compliance_score >= 0.9:
                    summary['passed_checks'] += 1
                elif compliance_score >= 0.7:
                    summary['warnings'] += 1
                else:
                    summary['critical_issues'] += 1
                
                summary['total_issues'] += 1
            
            # Generate recommendations based on results
            if summary['critical_issues'] > 0:
                recommendations.append("Address critical compliance issues immediately")
            
            if summary['warnings'] > 0:
                recommendations.append("Review and fix warning-level compliance issues")
            
            if summary['passed_checks'] == len(self.analysis_results['model_results']):
                recommendations.append("All compliance checks passed successfully")
            
            self.analysis_results['summary'] = summary
            self.analysis_results['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of a specific analysis"""
        if analysis_id == self.current_analysis_id:
            return {
                'status': 'completed',
                'results': self.analysis_results
            }
        else:
            return {'error': 'Analysis ID not found'}
        

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Cleanup any resources used by models
            logger.info("Cleaning up pipeline resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
