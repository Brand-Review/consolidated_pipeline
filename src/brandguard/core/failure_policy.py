"""
Critical Signal Failure Policy
Handles detection and reporting of critical signal failures.

Rules:
- If visible text exists but OCR fails → critical signal failure
- If critical signal failure → overallCompliance = null, bucket = "unknown"
- Every failure must explain why
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FailurePolicy:
    """
    Critical Signal Failure Policy
    
    Determines when analysis cannot proceed due to critical system errors.
    Critical failures prevent compliance scoring but allow signal extraction to continue.
    """
    
    def __init__(self):
        """Initialize failure policy"""
        pass
    
    def check_critical_failures(
        self,
        ocr_result: Optional[Dict[str, Any]] = None,
        vision_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check for critical signal failures across all analyzers.
        
        Critical failures occur when:
        1. Visible text is detected but OCR fails to extract it
        2. Vision analyzer fails when image is clearly valid
        3. Text analyzer fails when text was successfully extracted by OCR
        
        Args:
            ocr_result: OCR engine result with {hasText, text, failure, confidence}
            vision_result: Vision analyzer result with {status, visibleTextDetected, failure}
            text_result: Text analyzer result with {status, failure}
            
        Returns:
            Dictionary with failure information:
            {
                "criticalSignalFailure": bool,
                "failures": [str],  # List of failure descriptions
                "details": [str],  # Detailed failure information
                "recommendations": [str],  # Actionable recommendations
                "overallCompliance": null | float,  # null if critical failure
                "bucket": "unknown" | "approved" | "review" | "rejected"  # "unknown" if critical failure
            }
        """
        critical_failures = []
        failure_details = []
        recommendations = []
        
        # Check OCR failure with visible text (CRITICAL)
        ocr_failure = self._check_ocr_failure(ocr_result, vision_result)
        if ocr_failure["is_critical"]:
            critical_failures.append(ocr_failure["message"])
            failure_details.extend(ocr_failure["details"])
            recommendations.extend(ocr_failure["recommendations"])
            logger.error(f"🚨 CRITICAL SIGNAL FAILURE: {ocr_failure['message']}")
        
        # Check vision analyzer critical failure
        vision_failure = self._check_vision_failure(vision_result)
        if vision_failure["is_critical"]:
            critical_failures.append(vision_failure["message"])
            failure_details.extend(vision_failure["details"])
            recommendations.extend(vision_failure["recommendations"])
            logger.error(f"🚨 CRITICAL SIGNAL FAILURE: {vision_failure['message']}")
        
        # Check text analyzer critical failure
        text_failure = self._check_text_failure(text_result, ocr_result)
        if text_failure["is_critical"]:
            critical_failures.append(text_failure["message"])
            failure_details.extend(text_failure["details"])
            recommendations.extend(text_failure["recommendations"])
            logger.error(f"🚨 CRITICAL SIGNAL FAILURE: {text_failure['message']}")
        
        # Determine if critical signal failure occurred
        has_critical_failure = len(critical_failures) > 0
        
        # Build result
        result = {
            "criticalSignalFailure": has_critical_failure,
            "failures": critical_failures,
            "details": failure_details,
            "recommendations": recommendations,
            "overallCompliance": None if has_critical_failure else 0.0,  # null if critical failure
            "bucket": "unknown" if has_critical_failure else "pending",  # "unknown" if critical failure
            "timestamp": datetime.now().isoformat()
        }
        
        if has_critical_failure:
            logger.warning(f"⚠️ Critical signal failures detected: {len(critical_failures)} failures")
            logger.warning(f"   Failures: {', '.join(critical_failures)}")
        
        return result
    
    def _check_ocr_failure(
        self,
        ocr_result: Optional[Dict[str, Any]],
        vision_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for OCR failure when visible text exists (CRITICAL).
        
        CRITICAL RULES:
        - Low-confidence OCR with text extracted is NOT a failure
        - OCR failure only causes criticalSignalFailure if:
          1. Visible text detected with high confidence (>= 0.5)
          2. OCR extracted NO text (hasText = false AND text is empty)
          3. This indicates a system error, not missing text
        
        Low-confidence OCR that extracts partial text is acceptable and should NOT cause critical failure.
        """
        if not vision_result:
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        if not ocr_result:
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        # Check if vision detected visible text
        visible_text_detected = vision_result.get("visibleTextDetected", False)
        vision_confidence = vision_result.get("confidence", 0.0)
        
        # Check if OCR succeeded
        ocr_has_text = ocr_result.get("hasText", False)
        ocr_text = ocr_result.get("text", "")
        ocr_status = ocr_result.get("status", "")
        ocr_confidence = ocr_result.get("confidence", 0.0)
        ocr_failure = ocr_result.get("failure")
        
        # CRITICAL RULE: Low-confidence OCR with text is NOT a failure
        # If OCR extracted text (even with low confidence), it's not a critical failure
        has_extracted_text = bool(ocr_text and ocr_text.strip())
        is_low_confidence = ocr_status == "low_confidence" or ocr_confidence < 0.5
        
        if has_extracted_text:
            # Text was extracted - NOT a critical failure, even if low confidence
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        # CRITICAL: Visible text detected with high confidence but OCR extracted NO text
        # Only flag as critical if:
        # 1. Vision detected text with high confidence (>= 0.5)
        # 2. OCR extracted NO text (hasText = false AND text is empty)
        # 3. This indicates a system error, not missing text
        if visible_text_detected and vision_confidence >= 0.5 and not ocr_has_text and not has_extracted_text:
            # This is a critical system error
            text_regions_count = vision_result.get("textRegionsCount", 0)  # If available
            failure_reason = ocr_failure.get("reason", "OCR failed to extract text") if ocr_failure else "OCR failed to extract text"
            failure_type = ocr_failure.get("failure_type", "ocr_failure") if ocr_failure else "ocr_failure"
            
            details = [
                f"Vision analyzer detected visible text with high confidence ({vision_confidence:.2f})",
                f"OCR failed to extract ANY text: {failure_reason}",
                f"OCR failure type: {failure_type}",
                "This indicates a system error in OCR processing, not missing text",
                "CRITICAL: Visible text exists but OCR extracted nothing - system error"
            ]
            
            if text_regions_count > 0:
                details.insert(1, f"Vision detected {text_regions_count} text regions")
            
            recommendations = [
                "Verify OCR libraries are installed: pip install pytesseract (requires tesseract-ocr binary)",
                "Alternatively: pip install paddlepaddle paddleocr",
                "Check image quality and preprocessing settings",
                "Review OCR diagnostics in logs",
                "Verify image format is supported (PNG, JPG, etc.)",
                "Ensure image preprocessing (BGR→RGB, 2× upscaling) is working correctly"
            ]
            
            # Add OCR-specific recommendations if available
            if ocr_failure and "recommendations" in ocr_failure:
                recommendations.extend(ocr_failure["recommendations"][:3])  # Add first 3 recommendations
            
            return {
                "is_critical": True,
                "message": f"OCR failed to extract text despite visible text detected with high confidence ({vision_confidence:.2f})",
                "details": details,
                "recommendations": list(set(recommendations))  # Remove duplicates
            }
        
        # No critical failure - either:
        # 1. No visible text detected (expected)
        # 2. Low confidence vision detection (uncertain)
        # 3. OCR extracted text (even if low confidence)
        return {"is_critical": False, "message": "", "details": [], "recommendations": []}
    
    def _check_vision_failure(self, vision_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for vision analyzer critical failure.
        
        Rule: If vision analyzer status is "failed" with invalid_input or model_error,
              this is a critical failure (system cannot analyze images).
        """
        if not vision_result:
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        status = vision_result.get("status", "unknown")
        failure = vision_result.get("failure")
        
        # Only critical if status is "failed" and failure type is critical
        if status == "failed" and failure:
            failure_type = failure.get("failure_type", "")
            
            # Critical failure types
            critical_types = ["invalid_input", "model_error", "connection_error"]
            
            if failure_type in critical_types:
                reason = failure.get("reason", "Vision analyzer failed")
                recommendations = failure.get("recommendations", [])
                
                details = [
                    f"Vision analyzer failed: {reason}",
                    f"Failure type: {failure_type}",
                    "System cannot analyze images"
                ]
                
                if not recommendations:
                    recommendations = [
                        "Check OpenRouter API key (OPENROUTER_API_KEY environment variable)",
                        "Verify API key has sufficient credits",
                        "Check network connectivity",
                        "Verify image format is supported"
                    ]
                
                return {
                    "is_critical": True,
                    "message": f"Vision analyzer critical failure: {reason}",
                    "details": details,
                    "recommendations": recommendations
                }
        
        return {"is_critical": False, "message": "", "details": [], "recommendations": []}
    
    def _check_text_failure(
        self,
        text_result: Optional[Dict[str, Any]],
        ocr_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for text analyzer critical failure.
        
        Rule: If OCR successfully extracted text (hasText = true) but text analyzer failed,
              this is a critical failure (system cannot analyze text content).
        """
        if not text_result:
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        if not ocr_result:
            return {"is_critical": False, "message": "", "details": [], "recommendations": []}
        
        # Check if OCR succeeded
        ocr_has_text = ocr_result.get("hasText", False)
        ocr_text = ocr_result.get("text", "")
        
        # Check text analyzer status
        text_status = text_result.get("status", "unknown")
        text_failure = text_result.get("failure")
        
        # CRITICAL: OCR extracted text but text analyzer failed
        if ocr_has_text and ocr_text and len(ocr_text.split()) >= 3:
            # Text is available for analysis
            if text_status == "failed" and text_failure:
                failure_type = text_failure.get("failure_type", "")
                
                # Critical failure types
                critical_types = ["invalid_input", "model_error", "connection_error", "parse_error"]
                
                if failure_type in critical_types:
                    reason = text_failure.get("reason", "Text analyzer failed")
                    recommendations = text_failure.get("recommendations", [])
                    
                    details = [
                        f"OCR successfully extracted {len(ocr_text.split())} words",
                        f"Text analyzer failed: {reason}",
                        f"Failure type: {failure_type}",
                        "System cannot analyze text content"
                    ]
                    
                    if not recommendations:
                        recommendations = [
                            "Check OpenRouter API key (OPENROUTER_API_KEY environment variable)",
                            "Verify API key has sufficient credits",
                            "Check network connectivity",
                            "Review text analyzer logs for details"
                        ]
                    
                    return {
                        "is_critical": True,
                        "message": f"Text analyzer critical failure: {reason}",
                        "details": details,
                        "recommendations": recommendations
                    }
        
        return {"is_critical": False, "message": "", "details": [], "recommendations": []}
    
    def apply_failure_policy(
        self,
        analysis_results: Dict[str, Any],
        failure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply failure policy to analysis results.
        
        Effects of critical signal failure:
        - overallCompliance = null (cannot be scored)
        - bucket = "unknown" (cannot be categorized)
        - summary = failure explanation
        - recommendations = failure recommendations
        
        Args:
            analysis_results: Complete analysis results dictionary
            failure_info: Failure information from check_critical_failures()
            
        Returns:
            Modified analysis results with failure policy applied
        """
        if failure_info.get("criticalSignalFailure", False):
            # Apply critical failure effects
            analysis_results["criticalSignalFailure"] = True
            analysis_results["overallCompliance"] = None
            analysis_results["bucket"] = "unknown"
            
            # Set summary
            failures = failure_info.get("failures", [])
            if failures:
                analysis_results["summary"] = f"Analysis could not be completed due to critical system errors: {', '.join(failures[:2])}"
            else:
                analysis_results["summary"] = "Analysis could not be completed due to critical system errors"
            
            # Add recommendations
            existing_recommendations = analysis_results.get("recommendations", [])
            failure_recommendations = failure_info.get("recommendations", [])
            
            # Merge recommendations (avoid duplicates)
            all_recommendations = existing_recommendations.copy()
            for rec in failure_recommendations:
                if rec not in all_recommendations:
                    all_recommendations.append(rec)
            
            analysis_results["recommendations"] = all_recommendations
            
            # Add failure details
            failure_details = failure_info.get("details", [])
            if failure_details:
                analysis_results["failureDetails"] = failure_details
            
            logger.info("✅ Failure policy applied: overallCompliance=null, bucket=unknown")
        
        else:
            # No critical failure - proceed normally
            analysis_results["criticalSignalFailure"] = False
        
        return analysis_results

