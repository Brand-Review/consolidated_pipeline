"""
Error Handler with Kill Switch
Handles AI provider failures gracefully
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Kill switch environment variable
DISABLE_AI_ANALYSIS = os.getenv("DISABLE_AI_ANALYSIS", "false").lower() == "true"


class AnalysisErrorHandler:
    """Handles errors and provides safe fallbacks"""
    
    @staticmethod
    def is_ai_disabled() -> bool:
        """Check if AI analysis is disabled via kill switch"""
        return DISABLE_AI_ANALYSIS
    
    @staticmethod
    def handle_provider_error(
        error: Exception,
        provider: str = "unknown",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle provider errors gracefully
        
        Args:
            error: The exception that occurred
            provider: Provider name (openrouter, vllm, ollama)
            context: Additional context about the error
            
        Returns:
            Safe fallback response
        """
        logger.error(f"Provider error ({provider}): {str(error)}")
        if context:
            logger.error(f"Context: {context}")
        
        if AnalysisErrorHandler.is_ai_disabled():
            return AnalysisErrorHandler.get_disabled_response()
        
        return AnalysisErrorHandler.get_fallback_response(provider)
    
    @staticmethod
    def get_disabled_response() -> Dict[str, Any]:
        """
        Return response when AI is disabled via kill switch
        
        Returns:
            Safe response indicating analysis unavailable
        """
        return {
            "success": False,
            "error": {
                "message": "Analysis temporarily unavailable",
                "code": "AI_DISABLED",
                "user_message": "Analysis service is currently unavailable. Please try again later."
            },
            "scores": {
                "grammar": 50,
                "tone": 50,
                "brand": 50,
                "overall": 50.0,
                "bucket": "review"
            },
            "explanation": ["Analysis service temporarily unavailable"],
            "metadata": {
                "ai_disabled": True,
                "reason": "Kill switch activated"
            }
        }
    
    @staticmethod
    def get_fallback_response(provider: str) -> Dict[str, Any]:
        """
        Return safe fallback response when provider fails
        
        Args:
            provider: Provider name that failed
            
        Returns:
            Conservative fallback response
        """
        return {
            "success": False,
            "error": {
                "message": f"Analysis provider ({provider}) failed",
                "code": "PROVIDER_ERROR",
                "user_message": "Analysis temporarily unavailable. Please try again later."
            },
            "scores": {
                "grammar": 50,
                "tone": 50,
                "brand": 50,
                "overall": 50.0,
                "bucket": "review"  # Conservative: never approve on error
            },
            "explanation": ["Analysis service temporarily unavailable"],
            "metadata": {
                "provider": provider,
                "fallback": True
            }
        }
    
    @staticmethod
    def validate_response(response: Dict[str, Any]) -> bool:
        """
        Validate that response has required structure
        
        Args:
            response: Response to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["scores", "explanation"]
        if "scores" in response:
            required_score_keys = ["grammar", "tone", "brand", "overall", "bucket"]
            if not all(key in response["scores"] for key in required_score_keys):
                return False
        
        return all(key in response for key in required_keys)
    
    @staticmethod
    def ensure_safe_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure response is safe (never approve on error)
        
        Args:
            response: Response to sanitize
            
        Returns:
            Safe response
        """
        if not AnalysisErrorHandler.validate_response(response):
            logger.warning("Invalid response structure, using fallback")
            return AnalysisErrorHandler.get_fallback_response("unknown")
        
        # If error occurred, ensure bucket is never "approve"
        if response.get("success") is False or "error" in response:
            if "scores" in response and response["scores"].get("bucket") == "approve":
                logger.warning("Error response had approve bucket, correcting to review")
                response["scores"]["bucket"] = "review"
                response["scores"]["overall"] = min(response["scores"]["overall"], 84.0)
        
        return response

