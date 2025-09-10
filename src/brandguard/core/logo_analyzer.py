"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description: Logo Detection Analysis Module
Handles logo detection and validation
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LogoAnalyzer:
    """Handles logo detection and analysis functionality"""
    
    def __init__(self, settings, imported_models):
        self.settings = settings
        self.imported_models = imported_models
        self.logo_detector = None
        self.logo_validator = None
        
        # Initialize logo detection components
        self._init_logo_models()
    
    def _init_logo_models(self):
        """Initialize logo detection models"""
        try:
            LogoDetector = self.imported_models.get('LogoDetector')
            LogoValidator = self.imported_models.get('LogoPlacementValidator')
            
            if LogoDetector is not None:
                # Create logo detector configuration - Using fine-tuned logo detection model
                # Use default values if settings is None
                if self.settings and hasattr(self.settings, 'logo_detection'):
                    logo_config = {
                        'type': 'yolos',  # Using YOLOS model fine-tuned for logo detection
                        'confidence_threshold': self.settings.logo_detection.confidence_threshold,
                        'use_yolo': self.settings.logo_detection.use_yolo,
                        'use_qwen': self.settings.logo_detection.use_qwen,
                        'path': 'ellabettison/Logo-Detection-finetune',  # Fine-tuned logo detection model
                        'qwen_api_url': self.settings.logo_detection.qwen_api_url,
                        'qwen_model': self.settings.logo_detection.qwen_model
                    }
                else:
                    # Default configuration
                    logo_config = {
                        'type': 'yolos',
                        'confidence_threshold': 0.5,
                        'use_yolo': True,
                        'use_qwen': True,
                        'path': 'ellabettison/Logo-Detection-finetune',
                        'qwen_api_url': 'http://localhost:11434/v1',
                        'qwen_model': 'Qwen/Qwen2.5-VL-3B-Instruct'
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
                # LogoPlacementValidator requires BrandRules, not logo_config
                from dataclasses import dataclass
                
                @dataclass
                class BrandRules:
                    """Brand compliance rules"""
                    allowed_zones: list = None
                    min_logo_size: float = 0.01
                    max_logo_size: float = 0.25
                    min_edge_distance: float = 0.05
                    aspect_ratio_tolerance: float = 0.2
                    
                    def __post_init__(self):
                        if self.allowed_zones is None:
                            self.allowed_zones = ["top-left", "top-right", "bottom-left", "bottom-right"]
                
                brand_rules = BrandRules()
                self.logo_validator = LogoValidator(brand_rules)
                logger.info("✅ LogoValidator initialized with BrandRules")
            else:
                logger.warning("⚠️ LogoValidator not available, using fallback")
                
        except Exception as e:
            logger.error(f"Logo detection initialization failed: {e}")
            self.logo_detector = None
            self.logo_validator = None
    
    def analyze_logos(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform logo detection analysis on an image"""
        try:
            if not options:
                options = {}
            
            # Check if logo analysis is enabled
            if not options.get('enabled', True):
                return {
                    'logo_detections': [],
                    'placement_validation': {},
                    'brand_compliance': {},
                    'analysis_type': 'disabled'
                }
            
            # Use real models if available
            if self.logo_detector and self.logo_validator:
                return self._analyze_with_real_models(image, options)
            else:
                return self._analyze_with_fallback(image, options)
                
        except Exception as e:
            logger.error(f"Logo analysis failed: {e}")
            return {
                'error': f'Logo analysis failed: {str(e)}',
                'logo_detections': [],
                'placement_validation': {},
                'brand_compliance': {},
                'analysis_type': 'error'
            }
    
    def _analyze_with_real_models(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze logos using real models"""
        try:
            start_time = datetime.now()
            
            # Detect logos
            logo_detections = self.logo_detector.detect_logos(image)

            # Validate logo placement if enabled
            placement_validation = {}
            if options.get('enable_placement_validation', True):
                placement_validation = self._validate_logo_placement_real(
                    logo_detections, 
                    image.shape,
                    options
                )
            
            # Check brand compliance if enabled
            brand_compliance = {}
            if options.get('enable_brand_compliance', True):
                brand_compliance = self._check_logo_brand_compliance_real(logo_detections)
            
            # Calculate compliance score
            compliance_score = self._calculate_logo_compliance_score(
                logo_detections, 
                placement_validation, 
                brand_compliance
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                'logo_detections': logo_detections,
                'placement_validation': placement_validation,
                'brand_compliance': brand_compliance,
                'scores': {
                    'overall': compliance_score,
                    'placement': placement_validation.get('compliance_score', 0),
                    'brand': brand_compliance.get('compliance_score', 0)
                },
                'processing_time_ms': processing_time,
                'analysis_type': 'real_logo_analysis'
            }
            
        except Exception as e:
            logger.error(f"Real logo analysis failed: {e}")
            return self._analyze_with_fallback(image, options)
    
    def _analyze_with_fallback(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback logo analysis when real models are not available"""
        try:
            # Generate dummy detections for testing
            detections = self._detect_logos_fallback(image)
            
            # Validate placement
            placement_validation = self._validate_logo_placement_fallback(detections, image.shape)
            
            # Check brand compliance
            brand_compliance = self._check_logo_brand_compliance_fallback(detections)
            
            # Calculate compliance score
            compliance_score = self._calculate_logo_compliance_score(
                detections, 
                placement_validation, 
                brand_compliance
            )
            
            return {
                'logo_detections': detections,
                'placement_validation': placement_validation,
                'brand_compliance': brand_compliance,
                'scores': {
                    'overall': compliance_score,
                    'placement': placement_validation.get('compliance_score', 0),
                    'brand': brand_compliance.get('compliance_score', 0)
                },
                'analysis_type': 'fallback_logo_analysis'
            }
            
        except Exception as e:
            logger.error(f"Fallback logo analysis failed: {e}")
            return {
                'error': f'Fallback logo analysis failed: {str(e)}',
                'logo_detections': [],
                'placement_validation': {},
                'brand_compliance': {},
                'analysis_type': 'error'
            }
    
    def _validate_logo_placement_real(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...], options: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logo placement using real model"""
        try:
            if not self.logo_validator:
                return self._validate_logo_placement_fallback(detections, image_shape)
            
            # Use real model for placement validation
            # Ensure image_shape is a tuple of (height, width, channels)
            if len(image_shape) == 2:
                # If only height, width provided, add channels
                h, w = image_shape
                image_shape_tuple = (h, w, 3)  # Assume 3 channels
            else:
                image_shape_tuple = tuple(image_shape)
            
            validation_result = self.logo_validator.validate_placement(detections, image_shape_tuple)

            return validation_result
            
        except Exception as e:
            logger.error(f"Real placement validation failed: {e}, using fallback")
            return self._validate_logo_placement_fallback(detections, image_shape)
    
    def _validate_logo_placement_fallback(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Fallback logo placement validation"""
        try:
            if not detections:
                return {
                    'valid': True,
                    'message': 'No logos detected',
                    'compliance_score': 100.0,
                    'violations': []
                }
            
            height, width = image_shape[:2]
            violations = []
            
            for i, detection in enumerate(detections):
                # Check logo size
                bbox = detection.get('bbox', [0, 0, 0, 0])
                logo_width = bbox[2] - bbox[0]
                logo_height = bbox[3] - bbox[1]
                logo_area = logo_width * logo_height
                image_area = width * height
                
                if logo_area / image_area < 0.01:  # Less than 1% of image
                    violations.append(f"Logo {i+1} is too small")
                elif logo_area / image_area > 0.25:  # More than 25% of image
                    violations.append(f"Logo {i+1} is too large")
                
                # Check edge distance
                min_edge_distance = min(bbox[0], bbox[1], width - bbox[2], height - bbox[3])
                if min_edge_distance < width * 0.05:  # Less than 5% from edge
                    violations.append(f"Logo {i+1} is too close to edge")
            
            compliance_score = max(0, 100 - len(violations) * 20)  # -20 points per violation
            
            return {
                'valid': len(violations) == 0,
                'message': f'Found {len(violations)} placement violations',
                'compliance_score': compliance_score,
                'violations': violations
            }
            
        except Exception as e:
            logger.error(f"Placement validation failed: {e}")
            return {
                'valid': False,
                'message': f'Validation failed: {str(e)}',
                'compliance_score': 0.0,
                'violations': ['Validation error']
            }
    
    def _check_logo_brand_compliance_real(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check logo brand compliance using real model"""
        try:
            if not self.logo_validator:
                return self._check_logo_brand_compliance_fallback(detections)
            
            # Use real model for brand compliance
            # LogoPlacementValidator doesn't have check_brand_compliance method
            # Use placement validation as a proxy for brand compliance
            # Ensure proper image shape format
            compliance_result = self.logo_validator.validate_placement(detections, (1000, 1000, 3))
            return compliance_result
            
        except Exception as e:
            logger.error(f"Real brand compliance check failed: {e}, using fallback")
            return self._check_logo_brand_compliance_fallback(detections)
    
    def _check_logo_brand_compliance_fallback(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback logo brand compliance check"""
        try:
            if not detections:
                return {
                    'valid': True,
                    'message': 'No logos detected',
                    'compliance_score': 100.0,
                    'violations': []
                }
            
            violations = []
            for i, detection in enumerate(detections):
                confidence = detection.get('confidence', 0)
                if confidence < 0.5:
                    violations.append(f"Logo {i+1} has low confidence ({confidence:.2f})")
            
            compliance_score = max(0, 100 - len(violations) * 25)  # -25 points per violation
            
            return {
                'valid': len(violations) == 0,
                'message': f'Found {len(violations)} brand compliance violations',
                'compliance_score': compliance_score,
                'violations': violations
            }
            
        except Exception as e:
            logger.error(f"Brand compliance check failed: {e}")
            return {
                'valid': False,
                'message': f'Compliance check failed: {str(e)}',
                'compliance_score': 0.0,
                'violations': ['Compliance check error']
            }
    
    def _detect_logos_fallback(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback logo detection (dummy implementation)"""
        try:
            # Generate dummy detections for testing
            height, width = image.shape[:2]
            dummy_detections = [
                {
                    'bbox': [width * 0.1, height * 0.1, width * 0.3, height * 0.3],
                    'confidence': 0.85,
                    'class': 'logo',
                    'class_id': 0
                }
            ]
            return dummy_detections
            
        except Exception as e:
            logger.error(f"Fallback logo detection failed: {e}")
            return []
    
    def _calculate_logo_compliance_score(self, detections: List[Dict[str, Any]], placement_validation: Dict[str, Any], brand_compliance: Dict[str, Any]) -> float:
        """Calculate overall logo compliance score"""
        try:
            placement_score = placement_validation.get('compliance_score', 0)
            brand_score = brand_compliance.get('compliance_score', 0)
            
            # Weighted average (50% placement, 50% brand)
            overall_score = (placement_score * 0.5) + (brand_score * 0.5)
            
            return round(overall_score, 2)
            
        except Exception as e:
            logger.error(f"Compliance score calculation failed: {e}")
            return 0.0
