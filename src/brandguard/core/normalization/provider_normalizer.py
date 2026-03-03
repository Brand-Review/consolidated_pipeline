"""
Provider Normalization Layer
Ensures consistent output format across OpenRouter, vLLM, and Ollama
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ProviderNormalizer:
    """Normalizes outputs from different providers to canonical format"""
    
    # Canonical contract structure
    CANONICAL_CONTRACT = {
        "version": "analysis_v1",
        "detected": {},
        "observations": {},
        "flags": [],
        "raw_metrics": {},
        "confidence": 0.0
    }
    
    # Allowed enum values
    ALLOWED_TONES = ["energetic", "neutral", "calm", "unknown"]
    ALLOWED_SENTIMENTS = ["positive", "neutral", "negative"]
    ALLOWED_CONFIDENCE_LEVELS = ["low", "balanced", "high"]
    
    def __init__(self):
        """Initialize normalizer"""
        pass
    
    def normalize(self, raw_response: Dict[str, Any], provider: str = "unknown") -> Dict[str, Any]:
        """
        Normalize provider response to canonical format
        
        Args:
            raw_response: Raw response from provider
            provider: Provider name (openrouter, vllm, ollama)
            
        Returns:
            Normalized response in canonical format
        """
        try:
            # Start with canonical structure
            normalized = self.CANONICAL_CONTRACT.copy()
            
            # Extract version if present
            normalized["version"] = raw_response.get("version", "analysis_v1")
            
            # Normalize detected fields
            normalized["detected"] = self._normalize_detected(raw_response, provider)
            
            # Normalize observations
            normalized["observations"] = self._normalize_observations(raw_response, provider)
            
            # Normalize flags
            normalized["flags"] = self._normalize_flags(raw_response, provider)
            
            # Normalize raw_metrics
            normalized["raw_metrics"] = self._normalize_raw_metrics(raw_response, provider)
            
            # Normalize confidence
            normalized["confidence"] = self._normalize_confidence(raw_response, provider)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization failed for {provider}: {e}")
            return self.CANONICAL_CONTRACT.copy()
    
    def _normalize_detected(self, raw: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Normalize detected fields"""
        detected = {}
        
        # Text extraction
        if "text" in raw or "extracted_text" in raw:
            detected["text"] = raw.get("text") or raw.get("extracted_text", "")
        
        # Text metrics
        text_metrics = raw.get("text_metrics", {})
        if text_metrics:
            detected["sentence_count"] = text_metrics.get("sentence_count", 0)
            detected["word_count"] = text_metrics.get("word_count", 0)
        
        # Colors (for brand extraction)
        if "colors" in raw:
            detected["colors"] = raw["colors"]
        elif "primaryColor" in raw:
            # Handle brand extractor format
            primary = raw.get("primaryColor", {})
            if isinstance(primary, dict):
                detected["colors"] = [primary.get("hex", "")]
            elif isinstance(primary, str):
                detected["colors"] = [primary]
        
        # Fonts
        if "fonts" in raw:
            detected["fonts"] = raw["fonts"]
        
        # Texts from OCR
        if "texts" in raw:
            detected["texts"] = raw["texts"]
        
        return detected
    
    def _normalize_observations(self, raw: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Normalize observations"""
        observations = {}
        
        # Grammar observations
        grammar_analysis = raw.get("grammar_analysis", {})
        if grammar_analysis:
            observations["grammar_errors"] = grammar_analysis.get("grammar_errors", [])
            observations["spelling_errors"] = grammar_analysis.get("spelling_errors", [])
            observations["punctuation_issues"] = grammar_analysis.get("punctuation_issues", [])
        
        # Tone observations
        tone_analysis = raw.get("tone_analysis", {})
        if tone_analysis:
            # Normalize tone enum
            tone = tone_analysis.get("tone") or tone_analysis.get("tone_category")
            if tone:
                observations["tone"] = self._normalize_enum(tone, self.ALLOWED_TONES, "neutral")
            
            # Normalize sentiment enum
            sentiment = tone_analysis.get("sentiment") or tone_analysis.get("overall_sentiment")
            if sentiment:
                observations["sentiment"] = self._normalize_enum(sentiment, self.ALLOWED_SENTIMENTS, "neutral")
            
            # Normalize confidence level
            confidence_level = tone_analysis.get("confidence_level")
            if confidence_level:
                observations["confidence_level"] = self._normalize_enum(
                    confidence_level, self.ALLOWED_CONFIDENCE_LEVELS, "balanced"
                )
        
        # Direct observations (if provider already uses canonical format)
        if "observations" in raw:
            observations.update(raw["observations"])
        
        return observations
    
    def _normalize_flags(self, raw: Dict[str, Any], provider: str) -> List[str]:
        """Normalize flags"""
        flags = []
        
        # Extract flags from various sources
        if "flags" in raw:
            flags.extend(raw["flags"])
        
        # Extract flags from grammar errors
        grammar_analysis = raw.get("grammar_analysis", {})
        if grammar_analysis.get("grammar_errors"):
            flags.append("grammar_error_detected")
        
        if grammar_analysis.get("spelling_errors"):
            flags.append("spelling_error_detected")
        
        # Extract flags from compliance failures
        compliance = raw.get("compliance", {})
        if compliance.get("failures"):
            flags.append("compliance_failure_detected")
        
        # Special rule: "Sing up" error
        detected_text = raw.get("text") or raw.get("extracted_text", "")
        if "Sing up" in detected_text or ("sing up" in detected_text.lower() and "Sign up" not in detected_text):
            flags.append("known_capitalization_error")
        
        return list(set(flags))  # Remove duplicates
    
    def _normalize_raw_metrics(self, raw: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Normalize raw metrics"""
        metrics = {}
        
        # Text block count from OCR
        if "text_block_count" in raw:
            metrics["text_block_count"] = raw["text_block_count"]
        
        # Font count
        if "fonts" in raw:
            metrics["font_count"] = len(raw["fonts"])
        
        # Error counts
        grammar_analysis = raw.get("grammar_analysis", {})
        if grammar_analysis:
            metrics["grammar_error_count"] = len(grammar_analysis.get("grammar_errors", []))
            metrics["spelling_error_count"] = len(grammar_analysis.get("spelling_errors", []))
        
        # Direct raw_metrics
        if "raw_metrics" in raw:
            metrics.update(raw["raw_metrics"])
        
        return metrics
    
    def _normalize_confidence(self, raw: Dict[str, Any], provider: str) -> float:
        """Normalize confidence score"""
        # Try various confidence fields
        confidence = raw.get("confidence", 0.0)
        
        if confidence == 0.0:
            # Try tone analysis confidence
            tone_analysis = raw.get("tone_analysis", {})
            if "confidence" in tone_analysis:
                confidence = tone_analysis["confidence"]
        
        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, float(confidence)))
    
    def _normalize_enum(self, value: str, allowed: List[str], default: str) -> str:
        """Normalize enum value to allowed set"""
        if not value:
            return default
        
        value_lower = value.lower()
        
        # Try exact match
        for allowed_val in allowed:
            if value_lower == allowed_val.lower():
                return allowed_val
        
        # Try partial match
        for allowed_val in allowed:
            if allowed_val.lower() in value_lower or value_lower in allowed_val.lower():
                return allowed_val
        
        # Return default if no match
        return default
    
    def fill_missing_fields(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill missing fields in response to ensure canonical contract
        
        Args:
            response: Response to fill
            
        Returns:
            Response with all required fields
        """
        normalized = self.CANONICAL_CONTRACT.copy()
        normalized.update(response)
        
        # Ensure all top-level fields exist
        if "detected" not in normalized:
            normalized["detected"] = {}
        if "observations" not in normalized:
            normalized["observations"] = {}
        if "flags" not in normalized:
            normalized["flags"] = []
        if "raw_metrics" not in normalized:
            normalized["raw_metrics"] = {}
        if "confidence" not in normalized:
            normalized["confidence"] = 0.0
        
        return normalized

