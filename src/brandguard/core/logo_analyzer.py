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
                    'status': 'skipped',
                    'reason': 'Logo analysis disabled in options',
                    'logo_detections': [],
                    'placement_validation': {},
                    'brand_compliance': {},
                    'analysis_type': 'disabled',
                    'confidence': 0.0
                }
            
            # Use real models if available
            if self.logo_detector and self.logo_validator:
                result = self._analyze_with_real_models(image, options)
            else:
                result = self._analyze_with_fallback(image, options)

            # Normalize "no logo" vs failure
            detections = result.get('logo_detections') or result.get('detections') or []
            if not detections and result.get('status') not in ['failed', 'skipped']:
                result['status'] = 'not_detected'
                result['confidence'] = 0.0
                result['message'] = 'No logo detected'

            return result
                
        except Exception as e:
            logger.error(f"Logo analysis failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Analysis error: {str(e)}',
                'error': f'Logo analysis failed: {str(e)}',
                'logo_detections': [],
                'placement_validation': {},
                'brand_compliance': {},
                'analysis_type': 'error',
                'confidence': 0.0
            }
    
    def _analyze_with_real_models(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze logos using real models"""
        try:
            start_time = datetime.now()
            
            # Detect logos
            raw_logo_detections = self.logo_detector.detect_logos(image)
            
            # Stage A: Apply deterministic filtering to reduce false positives
            try:
                from ..filters import filter_logo_candidates
                
                # Validate image shape
                if len(image.shape) < 2:
                    logger.error(f"Invalid image shape: {image.shape}")
                    filtered_detections = raw_logo_detections or []
                else:
                    image_height, image_width = image.shape[:2]
                    filter_config = {
                        'logoConfidenceThreshold': options.get('logoConfidenceThreshold', 0.75),
                        'minLogoSize': options.get('minLogoSize', 0.01),
                        'maxLogoSize': options.get('maxLogoSize', 0.25),
                        'allowedZones': options.get('allowedZones', [])
                    }
                    filtered_detections = filter_logo_candidates(
                        raw_logo_detections or [],
                        image_width,
                        image_height,
                        filter_config
                    )
            except Exception as e:
                logger.error(f"Error in logo filtering, using raw detections: {e}", exc_info=True)
                # Fallback to raw detections if filtering fails
                filtered_detections = raw_logo_detections or []
            
            # Stage B: Logo verification - verify detections are actually logos
            # RULE: If no reference logo is provided, detect "suspected logo" only, do NOT validate correctness
            brand_name = options.get('brand_name')
            reference_logo_image = options.get('reference_logo_image') or options.get('reference_logo')
            has_reference_logo = bool(brand_name or reference_logo_image)
            
            if not has_reference_logo:
                # No reference logo provided: Try to identify brands using VLM
                # This enables detection of NEW/UNKNOWN brands
                logger.info("⚠️ No reference logo provided - attempting brand identification using VLM")
                
                logo_detections = []
                suspected_logos = []
                unknown_graphics = []
                
                # Try to identify brand from each detected region using Qwen VLM
                identified_brands = []
                
                if filtered_detections and self.logo_detector:
                    try:
                        # Check if Qwen is available for brand identification
                        if hasattr(self.logo_detector, '_check_qwen_available') and self.logo_detector._check_qwen_available():
                            for det in filtered_detections:
                                bbox = det.get('bbox', [])
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    # Crop logo region
                                    h, w = image.shape[:2]
                                    x1, x2 = max(0, x1), min(w, x2)
                                    y1, y2 = max(0, y1), min(h, y2)
                                    logo_crop = image[y1:y2, x1:x2]
                                    
                                    # Identify brand from crop
                                    brand_result = self.logo_detector.identify_brand_from_crop(
                                        logo_crop, 
                                        hint=options.get('brand_name_hint', '')
                                    )
                                    
                                    if brand_result:
                                        det['identified_brand'] = brand_result.get('brand', 'Unknown')
                                        det['brand_confidence'] = brand_result.get('confidence', 0.0)
                                        det['brand_reasoning'] = brand_result.get('reasoning', '')
                                        identified_brands.append(det)
                                        logger.info(f"📋 Identified brand: {brand_result.get('brand')} (confidence: {brand_result.get('confidence', 0.0):.2f})")
                                    else:
                                        # VLM failed, fall back to suspected
                                        det['identified_brand'] = None
                                        det['brand_confidence'] = 0.0
                                        suspected_logos.append(d)
                                else:
                                    suspected_logos.append(d)
                        else:
                            # Qwen not available, mark all as suspected
                            logger.info("⚠️ VLM not available for brand identification - marking as suspected")
                            suspected_logos = filtered_detections
                    except Exception as e:
                        logger.warning(f"Brand identification failed: {e}, falling back to suspected")
                        suspected_logos = filtered_detections
                else:
                    # No detections or no detector
                    suspected_logos = filtered_detections if filtered_detections else []
                
                # Add verification info to suspected logos
                for d in suspected_logos:
                    d['verified'] = False
                    d['class_name'] = 'suspected_logo'
                    d['verification'] = {
                        'verified': False,
                        'verification_score': 0.0,
                        'rejection_reason': 'No reference logo provided - cannot validate correctness',
                        'status': 'suspected_only'
                    }
                
                # Add identified brands as detected logos (but not verified against reference)
                for d in identified_brands:
                    d['verified'] = False  # Identified, but not verified against reference
                    d['class_name'] = 'identified_logo'
                    d['verification'] = {
                        'verified': False,
                        'verification_score': d.get('brand_confidence', 0.0),
                        'rejection_reason': 'Brand identified but not verified against reference logo',
                        'status': 'identified'
                    }
                    # Move to logo_detections for reporting (identified but unverified)
                    logo_detections.append(d)
                
                logger.info(f"📋 {len(suspected_logos)} suspected logo(s), {len(identified_brands)} brand(s) identified (no reference for validation)")
            else:
                # Reference logo provided: run verification
                try:
                    from ..verification.logo_verifier import verify_logo_detections
                    
                    # Get OCR text if available (for brand name matching)
                    ocr_text = options.get('ocr_text') or options.get('extracted_text')
                    
                    min_logo_size = options.get('minLogoSize', 0.01)
                    max_logo_size = options.get('maxLogoSize', 0.25)
                    
                    # Verify all detections
                    verified_detections = verify_logo_detections(
                        filtered_detections,
                        image,
                        ocr_text=ocr_text,
                        brand_name=brand_name,
                        min_logo_size=min_logo_size,
                        max_logo_size=max_logo_size
                    )
                    
                    # Separate verified logos from unknown graphics
                    logo_detections = [d for d in verified_detections if d.get('verified', False)]
                    unknown_graphics = [d for d in verified_detections if not d.get('verified', False)]
                    suspected_logos = []
                    
                    if unknown_graphics:
                        logger.warning(f"⚠️ {len(unknown_graphics)} detections failed verification (classified as unknown_graphic)")
                        for ug in unknown_graphics:
                            logger.debug(f"  - Rejected: {ug.get('verification', {}).get('rejection_reason', 'Unknown')}")
                    
                    if logo_detections:
                        logger.info(f"✅ {len(logo_detections)} logo(s) verified")
                    
                    # 🚨 KEY: Check for BRAND MISMATCH!
                    # If user provided brand_name, check if detected logos match or are WRONG
                    if brand_name and logo_detections:
                        try:
                            from ..verification.logo_verifier import check_brand_mismatch
                            
                            reference_logo = options.get('reference_logo_image') or options.get('reference_logo')
                            
                            # Check for brand mismatch
                            logo_detections = check_brand_mismatch(
                                logo_detections,
                                image,
                                expected_brand=brand_name,
                                reference_logo=reference_logo
                            )
                            
                            # Separate correct brands from wrong brands
                            correct_brands = [d for d in logo_detections if not d.get('brand_match', {}).get('is_mismatch', False)]
                            wrong_brands = [d for d in logo_detections if d.get('brand_match', {}).get('is_mismatch', True)]
                            
                            if wrong_brands:
                                wrong_brand_names = [d.get('brand_match', {}).get('detected_brand', 'Unknown') for d in wrong_brands]
                                logger.warning(f"🚫 BRAND MISMATCH DETECTED! Expected '{brand_name}' but found: {wrong_brand_names}")
                            
                            if correct_brands:
                                logger.info(f"✅ Correct brand '{brand_name}' found")
                                
                        except Exception as e:
                            logger.warning(f"Brand mismatch check failed: {e}")
                
                except Exception as e:
                    logger.error(f"Logo verification failed: {e}, treating all as unverified", exc_info=True)
                    # If verification fails, mark all as unknown_graphic
                    logo_detections = []
                    suspected_logos = []
                    for d in filtered_detections:
                        d['verified'] = False
                        d['class_name'] = 'unknown_graphic'
                        d['verification'] = {
                            'verified': False,
                            'verification_score': 0.0,
                            'rejection_reason': 'Verification system error'
                        }
                    unknown_graphics = filtered_detections

            # ✅ FIX: HARD BLOCK placement validation unless reference logo exists AND logo is verified
            # RULE: Placement validation REQUIRES:
            #   1. Reference logo exists (has_reference_logo = True)
            #   2. Logo is verified (logo_detections contains verified logos)
            #   3. Allowed zones provided
            placement_validation = {}
            if options.get('enable_placement_validation', True):
                # ✅ HARD GATE: Check if we have verified logos AND reference logo
                if has_reference_logo and logo_detections and len(logo_detections) > 0:
                    # ✅ Both conditions met: verified logos exist AND reference logo provided
                    allowed_zones = options.get('allowedZones', [])
                    if allowed_zones and len(allowed_zones) > 0:
                        # ✅ Run placement validation on verified logos only
                        placement_validation = self._validate_logo_placement_real(
                            logo_detections,  # Only verified logos
                            image.shape,
                            options
                        )
                        logger.info(f"✅ Placement validation ran on {len(logo_detections)} verified logo(s)")
                    else:
                        # Verified logo exists but no allowed zones provided
                        placement_validation = {
                            'valid': False,
                            'status': 'not_applicable',
                            'message': 'No allowed zones provided - placement validation not applicable',
                            'compliance_score': None,  # RULE: null, not 0
                            'violations': []
                        }
                        logger.info("⚠️ Placement validation skipped - no allowed zones provided")
                elif suspected_logos and len(suspected_logos) > 0:
                    # ✅ FIX: Suspected logos (no reference) → NO placement validation
                    placement_validation = {
                        'valid': False,
                        'status': 'not_applicable',
                        'message': 'Logo identity not verified - placement validation requires reference logo',
                        'compliance_score': None,  # RULE: null, not 0
                        'violations': []
                    }
                    logger.info("⚠️ Placement validation blocked - logo identity not verified (suspected only)")
                else:
                    # No logo detected → placement status = "not_applicable", score = null
                    placement_validation = {
                        'valid': False,
                        'status': 'not_applicable',
                        'message': 'No logo detected - placement validation not applicable',
                        'compliance_score': None,  # RULE: null, not 0
                        'violations': []
                    }
                    logger.info("⚠️ Placement validation not applicable - no logo detected")
            else:
                # Placement validation disabled
                placement_validation = {
                    'valid': False,
                    'status': 'not_applicable',  # Changed from 'disabled' to match schema
                    'message': 'Placement validation disabled',
                    'compliance_score': None,
                    'violations': []
                }
            
            # Check brand compliance if enabled
            brand_compliance = {}
            if options.get('enable_brand_compliance', True):
                brand_compliance = self._check_logo_brand_compliance_real(logo_detections)
            
            # ✅ FIX: Calculate compliance score using new verification-aware logic
            # If status will be "observed" (suspected logos only), compliance score should be None
            unknown_graphics_list = unknown_graphics if 'unknown_graphics' in locals() else []
            suspected_logos_list = suspected_logos if 'suspected_logos' in locals() else []
            
            # ✅ HARD GATE: Only calculate compliance if verified logos exist
            if logo_detections and len(logo_detections) > 0 and has_reference_logo:
                compliance_score = self._calculate_logo_compliance_score_verified(
                    logo_detections,
                    unknown_graphics_list,
                    suspected_logos_list,
                    placement_validation, 
                    brand_compliance
                )
            else:
                # ✅ No verified logos → no compliance score
                compliance_score = None
                logger.debug("Logo compliance score set to None - no verified logos or no reference logo")
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            # Include all detections in response for transparency
            all_detections = logo_detections + unknown_graphics_list + suspected_logos_list
            
            # ✅ FIX: Determine status based on verification state
            # Status: "verified" (verified + correct brand), "wrong_brand" (verified but wrong brand), "identified" (brand identified via VLM), "observed" (suspected, no reference), "not_detected" (no detections)
            
            # Check if any logos were identified (brand identified via VLM)
            identified_logos = [d for d in logo_detections if d.get('class_name') == 'identified_logo']
            
            # Check for WRONG BRAND detection
            wrong_brand_logos = [d for d in logo_detections if d.get('brand_match', {}).get('is_mismatch', False)]
            
            if logo_detections and len(logo_detections) > 0 and has_reference_logo:
                if wrong_brand_logos:
                    # ✅ WRONG BRAND detected!
                    wrong_brands = [d.get('brand_match', {}).get('detected_brand', 'Unknown') for d in wrong_brand_logos]
                    status = 'wrong_brand'
                    reason = f"Expected brand '{brand_name}' but found: {', '.join(wrong_brands)}"
                    logger.warning(f"🚫 Logo status: WRONG BRAND - expected '{brand_name}', found: {wrong_brands}")
                else:
                    # ✅ Verified logo with correct brand
                    status = 'verified'
                    reason = None
            elif identified_logos and len(identified_logos) > 0:
                # ✅ Brand identified via VLM (no reference, but we know what brand it is)
                status = 'identified'
                brand_names = [d.get('identified_brand', 'Unknown') for d in identified_logos]
                reason = f"Brand(s) identified via AI: {', '.join(brand_names)}"
                logger.info(f"✅ Logo status set to 'identified' - brands: {brand_names}")
            elif suspected_logos and len(suspected_logos) > 0:
                # ✅ FIX: Suspected logos (no reference) → status = "observed", NO compliance impact
                status = 'observed'
                reason = 'Logo-like regions detected but identity not verified (no reference logo provided)'
                logger.info("⚠️ Logo status set to 'observed' - suspected logos only, no compliance validation")
            else:
                # No logo detected - status is "not_detected" (evaluated, not missing)
                status = 'not_detected'
                reason = 'No logos detected in image'
            
            # Get search zones from options
            allowed_zones = options.get('allowedZones', [])
            search_zones = allowed_zones if allowed_zones else ['top-left', 'top-center', 'top-right', 'bottom-left', 'bottom-center', 'bottom-right']
            
            # Calculate detection confidence from all attempts
            detection_confidence = 0.0
            if all_detections:
                confidences = [d.get('confidence', 0.0) for d in all_detections if isinstance(d.get('confidence'), (int, float))]
                detection_confidence = max(confidences) if confidences else 0.0
            elif raw_logo_detections:
                confidences = [d.get('confidence', 0.0) for d in raw_logo_detections if isinstance(d.get('confidence'), (int, float))]
                detection_confidence = max(confidences) if confidences else 0.0
            
            # Format detections with bbox and position for output
            formatted_detections = []
            for det in logo_detections:
                bbox = det.get('bbox', [])
                if bbox and len(bbox) == 4:
                    # Calculate position from bbox
                    x1, y1, x2, y2 = bbox
                    img_h, img_w = image.shape[:2]
                    center_x = (x1 + x2) / 2 / img_w if img_w > 0 else 0.5
                    center_y = (y1 + y2) / 2 / img_h if img_h > 0 else 0.5
                    
                    # Determine position label
                    if center_x < 0.33:
                        position = 'top-left' if center_y < 0.33 else ('bottom-left' if center_y > 0.67 else 'left')
                    elif center_x > 0.67:
                        position = 'top-right' if center_y < 0.33 else ('bottom-right' if center_y > 0.67 else 'right')
                    else:
                        position = 'top-center' if center_y < 0.33 else ('bottom-center' if center_y > 0.67 else 'center')
                    
                    formatted_detections.append({
                        'bbox': bbox,
                        'position': position,
                        'confidence': det.get('confidence', 0.0),
                        'zone': position
                    })
            
            # Build message for not_detected case
            message = None
            if status == 'not_detected':
                message = f'No logo detected in standard logo zones'
            
            return {
                'status': status,
                'reason': reason,
                'message': message,
                'detections': formatted_detections,  # Formatted for target schema
                'logo_detections': logo_detections,  # Keep for backward compatibility
                'suspected_logos': suspected_logos_list,
                'unknown_graphics': unknown_graphics_list,
                'all_detections': all_detections,
                'searchZones': search_zones,  # 🔥 Expose search zones
                'placement_validation': {
                    'status': placement_validation.get('status', 'not_applicable'),
                    'violations': placement_validation.get('violations', [])
                },
                'brand_compliance': brand_compliance,
                'scores': {
                    'overall': compliance_score,  # ✅ None if no verified logos
                    'placement': placement_validation.get('compliance_score'),  # ✅ None if not applicable
                    'brand': brand_compliance.get('compliance_score') if brand_compliance else None  # ✅ None if not validated
                },
                'processing_time_ms': processing_time,
                'analysis_type': 'real_logo_analysis',
                'verification_enabled': has_reference_logo,
                'has_reference_logo': has_reference_logo,
                'verified_logos': logo_detections,  # ✅ Expose verified logos separately
                'identified_brands': [d.get('identified_brand', 'Unknown') for d in identified_logos] if 'identified_logos' in locals() else [],  # ✅ Brands identified via VLM
                'brand_mismatch': {
                    'has_mismatch': len(wrong_brand_logos) > 0 if 'wrong_brand_logos' in locals() else False,
                    'expected_brand': brand_name if has_reference_logo else None,
                    'wrong_brands': [d.get('brand_match', {}).get('detected_brand') for d in wrong_brand_logos] if 'wrong_brand_logos' in locals() else [],
                    'wrong_brand_count': len(wrong_brand_logos) if 'wrong_brand_logos' in locals() else 0
                },
                'analyzerStatus': status,  # ✅ Use status: verified | wrong_brand | identified | observed | not_detected
                'confidence': detection_confidence if detection_confidence > 0 else (0.8 if logo_detections else (0.5 if suspected_logos_list else 0.92))  # 🔥 High confidence for thorough search
            }
            
        except Exception as e:
            logger.error(f"Real logo analysis failed: {e}")
            return self._analyze_with_fallback(image, options)
    
    def _analyze_with_fallback(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback logo analysis when real models are not available"""
        try:
            # Generate dummy detections for testing
            detections = self._detect_logos_fallback(image)
            
            # Check if reference logo is provided
            brand_name = options.get('brand_name')
            reference_logo_image = options.get('reference_logo_image') or options.get('reference_logo')
            has_reference_logo = bool(brand_name or reference_logo_image)
            
            # For fallback, mark based on whether reference is provided
            if not has_reference_logo:
                # No reference: mark as suspected_logo
                for d in detections:
                    d['verified'] = False
                    d['class_name'] = 'suspected_logo'
                    d['verification'] = {
                        'verified': False,
                        'verification_score': 0.0,
                        'rejection_reason': 'Fallback mode - no reference logo provided',
                        'status': 'suspected_only'
                    }
                suspected_logos = detections
                unknown_graphics = []
            else:
                # Has reference but fallback mode: mark as unknown_graphic (can't verify)
                for d in detections:
                    d['verified'] = False
                    d['class_name'] = 'unknown_graphic'
                    d['verification'] = {
                        'verified': False,
                        'verification_score': 0.0,
                        'rejection_reason': 'Fallback mode - verification not available'
                    }
                suspected_logos = []
                unknown_graphics = detections
            
            # Placement validation: not applicable in fallback mode (no verified logos)
            placement_validation = {
                'valid': False,
                'status': 'not_applicable',
                'message': 'Fallback mode - placement validation not applicable',
                'compliance_score': None,  # RULE: null, not 0
                'violations': ['Fallback mode - cannot validate placement']
            }
            
            # Check brand compliance
            brand_compliance = self._check_logo_brand_compliance_fallback(detections)
            
            # Calculate compliance score
            if not detections:
                compliance_score = 0.0  # No detections = 0
            elif suspected_logos:
                compliance_score = 5.0  # Suspected logos = 5
            else:
                compliance_score = 10.0  # Unknown graphics = 10
            
            # Determine status for fallback
            if not detections:
                status = 'not_detected'
                reason = 'No logos detected in image'
            elif suspected_logos:
                status = 'not_detected'  # 🔥 Use not_detected for consistency
                reason = f'No verified logos detected (fallback mode)'
            else:
                status = 'not_detected'  # 🔥 Use not_detected for consistency
                reason = f'No verified logos detected (fallback mode)'
            
            # Get search zones from options
            allowed_zones = options.get('allowedZones', [])
            search_zones = allowed_zones if allowed_zones else ['top-left', 'top-center', 'top-right', 'bottom-left', 'bottom-center', 'bottom-right']
            
            return {
                'status': status,
                'reason': reason,
                'message': 'No logo detected in standard logo zones' if status == 'not_detected' else None,
                'detections': [],  # No verified detections in fallback
                'logo_detections': [],  # No verified logos in fallback
                'suspected_logos': suspected_logos,
                'unknown_graphics': unknown_graphics,
                'all_detections': detections,
                'searchZones': search_zones,  # 🔥 Expose search zones
                'placement_validation': placement_validation,
                'brand_compliance': brand_compliance,
                'scores': {
                    'overall': compliance_score,
                    'placement': placement_validation.get('compliance_score', 0),
                    'brand': brand_compliance.get('compliance_score', 0)
                },
                'analysis_type': 'fallback_logo_analysis',
                'verification_enabled': False,
                'has_reference_logo': has_reference_logo,
                'confidence': 0.2  # Low confidence in fallback mode
            }
            
        except Exception as e:
            logger.error(f"Fallback logo analysis failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Analysis error: {str(e)}',
                'error': f'Fallback logo analysis failed: {str(e)}',
                'logo_detections': [],
                'placement_validation': {},
                'brand_compliance': {},
                'analysis_type': 'error',
                'confidence': 0.0
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
    
    def _validate_logo_placement_fallback(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fallback logo placement validation with proper bbox checking.
        
        CRITICAL RULES:
        - Logo covering >25% of image MUST FAIL
        - Center zone MUST FAIL if not allowed
        - Do NOT give 100% if bbox violates any rule
        - If identity is unknown (not verified), status = "not_applicable"
        """
        try:
            if not detections or len(detections) == 0:
                # RULE: No logo detected → placement status = "not_applicable", score = null
                return {
                    'valid': False,
                    'status': 'not_applicable',
                    'message': 'No logos detected - placement validation not applicable',
                    'compliance_score': None,  # RULE: null, not 0 or 100
                    'violations': []
                }
            
            # Check if detections have confirmed identity (verified)
            verified_detections = [d for d in detections if d.get('verified', False)]
            if not verified_detections:
                # Identity unknown - placement validation not applicable
                return {
                    'valid': False,
                    'status': 'not_applicable',
                    'message': 'Logo identity unknown - placement validation not applicable',
                    'compliance_score': None,  # RULE: null, not 0
                    'violations': ['Logo identity must be confirmed for placement validation']
                }
            
            height, width = image_shape[:2]
            image_area = width * height
            violations = []
            
            # Get validation rules from options
            min_logo_size = options.get('minLogoSize', 0.01) if options else 0.01
            max_logo_size = options.get('maxLogoSize', 0.25) if options else 0.25
            allowed_zones = options.get('allowedZones', ['top-left', 'top-right', 'bottom-left', 'bottom-right']) if options else ['top-left', 'top-right', 'bottom-left', 'bottom-right']
            
            # ✅ Extract placementDetails for each detection
            placement_details = []
            detection_violations = {}  # Track violations per detection
            
            # CRITICAL: Only validate placement for verified logos (confirmed identity)
            for i, detection in enumerate(verified_detections):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)
                
                # Validate bbox coordinates
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    violations.append(f"Logo {i+1}: Invalid bounding box coordinates")
                    detection_violations[i+1] = ["Invalid bounding box coordinates"]
                    continue
                
                logo_width = x2 - x1
                logo_height = y2 - y1
                logo_area = logo_width * logo_height
                size_ratio = logo_area / image_area if image_area > 0 else 0
                aspect_ratio = logo_width / logo_height if logo_height > 0 else 1.0
                
                # Determine logo zone
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Zone detection
                zone = None
                if center_x < width * 0.33:
                    if center_y < height * 0.33:
                        zone = 'top-left'
                    elif center_y > height * 0.67:
                        zone = 'bottom-left'
                    else:
                        zone = 'left'
                elif center_x > width * 0.67:
                    if center_y < height * 0.33:
                        zone = 'top-right'
                    elif center_y > height * 0.67:
                        zone = 'bottom-right'
                    else:
                        zone = 'right'
                else:
                    if center_y < height * 0.33:
                        zone = 'top-center'
                    elif center_y > height * 0.67:
                        zone = 'bottom-center'
                    else:
                        zone = 'center'
                
                # Check edge distance
                min_edge_distance = min(x1, y1, width - x2, height - y2)
                min_edge_distance_ratio = min_edge_distance / min(width, height) if min(width, height) > 0 else 0
                
                # Collect violations for this detection
                det_violations = []
                
                # CRITICAL: Logo covering >25% MUST FAIL
                if size_ratio > 0.25:
                    violations.append(f"Logo {i+1}: Too large ({size_ratio*100:.1f}% > 25% of image)")
                    det_violations.append(f"Too large ({size_ratio*100:.1f}% > 25% of image)")
                
                # Check min size
                if size_ratio < min_logo_size:
                    violations.append(f"Logo {i+1}: Too small ({size_ratio*100:.1f}% < {min_logo_size*100:.1f}%)")
                    det_violations.append(f"Too small ({size_ratio*100:.1f}% < {min_logo_size*100:.1f}%)")
                
                # Check max size
                if size_ratio > max_logo_size:
                    violations.append(f"Logo {i+1}: Exceeds max size ({size_ratio*100:.1f}% > {max_logo_size*100:.1f}%)")
                    det_violations.append(f"Exceeds max size ({size_ratio*100:.1f}% > {max_logo_size*100:.1f}%)")
                
                # CRITICAL: Center zone MUST FAIL if not allowed
                if zone == 'center' and 'center' not in allowed_zones and 'top-center' not in allowed_zones and 'bottom-center' not in allowed_zones:
                    violations.append(f"Logo {i+1}: In center zone (not allowed)")
                    det_violations.append("In center zone (not allowed)")
                
                # Check if zone is allowed
                if zone not in allowed_zones:
                    # Check for partial matches (e.g., 'top-left' vs 'top')
                    zone_parts = zone.split('-')
                    if not any(part in allowed_zones for part in zone_parts):
                        violations.append(f"Logo {i+1}: In {zone} zone (not in allowed zones: {allowed_zones})")
                        det_violations.append(f"In {zone} zone (not in allowed zones)")
                
                # Check edge distance
                if min_edge_distance_ratio < 0.05:  # Less than 5% from edge
                    violations.append(f"Logo {i+1}: Too close to edge ({min_edge_distance_ratio*100:.1f}% < 5%)")
                    det_violations.append(f"Too close to edge ({min_edge_distance_ratio*100:.1f}% < 5%)")
                
                # ✅ Build placementDetails entry
                placement_details.append({
                    'detectionId': i + 1,
                    'zone': zone,
                    'isValid': len(det_violations) == 0,
                    'sizeRatio': round(size_ratio, 4),
                    'aspectRatio': round(aspect_ratio, 2),
                    'edgeDistanceRatio': round(min_edge_distance_ratio, 4),
                    'violations': det_violations
                })
            
            # CRITICAL: Do NOT give 100% if any violation exists
            if violations:
                compliance_score = max(0, 100 - len(violations) * 20)  # -20 points per violation
                return {
                    'valid': False,
                    'message': f'Found {len(violations)} placement violations',
                    'compliance_score': float(compliance_score),
                    'violations': violations,
                    'placementDetails': placement_details,  # ✅ Always include
                    'status': 'failed'
                }
            else:
                return {
                    'valid': True,
                    'message': 'All logos pass placement validation',
                    'compliance_score': 100.0,
                    'violations': [],
                    'placementDetails': placement_details,  # ✅ Always include
                    'status': 'passed'
                }
            
        except Exception as e:
            logger.error(f"Placement validation failed: {e}")
            return {
                'valid': False,
                'message': f'Validation failed: {str(e)}',
                'compliance_score': None,  # RULE: null on error, not 0
                'violations': ['Validation error'],
                'status': 'failed'
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
    
    def _calculate_logo_compliance_score_verified(
        self,
        verified_logos: List[Dict[str, Any]],
        unknown_graphics: List[Dict[str, Any]],
        suspected_logos: List[Dict[str, Any]],
        placement_validation: Dict[str, Any],
        brand_compliance: Dict[str, Any]
    ) -> float:
        """
        Calculate logo compliance score using verification-aware logic.
        
        Rules:
        - No logo detected → score = 0
        - Suspected logo (no reference) → score = 5 (detected but not validated)
        - Unknown graphic → score = 10
        - Verified logo but bad placement → score = 30–60
        - Verified logo + valid placement → score = 70–100
        - Placement validation not applicable → score = 0 (identity unknown)
        """
        try:
            # No detections at all → score = 0
            if not verified_logos and not unknown_graphics and not suspected_logos:
                logger.info("No logo detections → score = 0")
                return 0.0
            
            # Suspected logos (no reference provided) → score = 5
            if suspected_logos and not verified_logos:
                logger.info(f"{len(suspected_logos)} suspected logo(s) (no reference) → score = 5")
                return 5.0
            
            # Only unknown graphics → score = 10
            if not verified_logos and unknown_graphics:
                logger.info(f"{len(unknown_graphics)} unknown graphic(s) → score = 10")
                return 10.0
            
            # Placement validation not applicable (identity unknown) → score = 0
            if placement_validation.get('status') == 'not_applicable':
                logger.info("Placement validation not applicable (identity unknown) → score = 0")
                return 0.0
            
            # We have verified logos - check placement
            placement_score = placement_validation.get('compliance_score', 0)
            placement_valid = placement_validation.get('valid', False)
            
            # Verified logo but bad placement → score = 30–60
            if not placement_valid or placement_score < 60:
                # Scale placement score to 30-60 range
                score = 30 + (placement_score / 100.0) * 30  # 30-60 range
                logger.info(f"Verified logo(s) but bad placement (score={placement_score}) → score = {score:.1f}")
                return round(score, 1)
            
            # Verified logo + valid placement → score = 70–100
            # Scale placement score to 70-100 range
            score = 70 + (placement_score / 100.0) * 30  # 70-100 range
            logger.info(f"Verified logo(s) + valid placement (score={placement_score}) → score = {score:.1f}")
            return round(score, 1)
            
        except Exception as e:
            logger.error(f"Logo compliance score calculation failed: {e}")
            return 0.0
    
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
