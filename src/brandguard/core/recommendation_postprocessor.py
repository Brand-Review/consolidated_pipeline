"""
Recommendation Post-Processor

Post-processes recommendations to:
1. Deduplicate messages
2. Tag each recommendation with source (system | user | config)
3. Remove user-blaming messages when analyzer status = failed
4. Force overallCompliance = null if criticalSignalFailure = true
"""

import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)


class RecommendationPostProcessor:
    """
    Post-processes recommendations to ensure quality and consistency.
    """
    
    # User-blaming patterns (messages that blame the user for system failures)
    USER_BLAMING_PATTERNS = [
        r'check image quality',
        r'verify.*is visible',
        r'ensure.*is provided',
        r'improve.*quality',
        r'check.*configuration',
        r'verify.*settings',
        r'ensure.*correct',
        r'fix.*issue',
        r'address.*problem',
        r'review.*for.*issues',
    ]
    
    # System error indicators (when analyzer failed due to system issues)
    SYSTEM_ERROR_INDICATORS = [
        'failed',
        'error',
        'exception',
        'unavailable',
        'not installed',
        'not available',
        'timeout',
        'connection error',
    ]
    
    def __init__(self):
        """Initialize recommendation post-processor"""
        self.user_blaming_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.USER_BLAMING_PATTERNS]
    
    def process(
        self,
        analysis_results: Dict[str, Any],
        analyzer_statuses: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Post-process analysis results to clean up recommendations.
        
        Args:
            analysis_results: Full analysis results dictionary
            analyzer_statuses: Optional dict mapping analyzer names to their statuses
            
        Returns:
            Modified analysis_results with processed recommendations
        """
        try:
            # Extract recommendations
            recommendations = analysis_results.get('recommendations', [])
            if not recommendations:
                return analysis_results
            
            # Step 1: Deduplicate messages
            recommendations = self._deduplicate(recommendations)
            
            # Step 2: Tag each recommendation with source
            recommendations = self._tag_with_source(recommendations, analysis_results)
            
            # Step 3: Remove user-blaming messages when analyzer status = failed
            if analyzer_statuses:
                recommendations = self._remove_user_blaming(
                    recommendations,
                    analyzer_statuses,
                    analysis_results
                )
            
            # Step 4: Force overallCompliance = null if criticalSignalFailure = true
            if analysis_results.get('criticalSignalFailure', False):
                analysis_results['overallCompliance'] = None
                logger.info("[PostProcessor] Forced overallCompliance = null due to criticalSignalFailure")
            
            # Update recommendations in results
            analysis_results['recommendations'] = recommendations
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"[PostProcessor] Error processing recommendations: {e}", exc_info=True)
            return analysis_results
    
    def _deduplicate(self, recommendations: List[Any]) -> List[Any]:
        """
        Deduplicate recommendation messages.
        
        Args:
            recommendations: List of recommendations (strings or dicts)
            
        Returns:
            Deduplicated list
        """
        seen = set()
        deduplicated = []
        
        for rec in recommendations:
            # Extract text for comparison
            if isinstance(rec, dict):
                text = rec.get('message', rec.get('text', str(rec)))
            else:
                text = str(rec)
            
            # Normalize text for comparison (lowercase, strip whitespace)
            normalized = text.lower().strip()
            
            if normalized not in seen:
                seen.add(normalized)
                deduplicated.append(rec)
        
        if len(deduplicated) < len(recommendations):
            logger.info(f"[PostProcessor] Deduplicated {len(recommendations)} -> {len(deduplicated)} recommendations")
        
        return deduplicated
    
    def _tag_with_source(self, recommendations: List[Any], analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Tag each recommendation with source (system | user | config).
        
        Args:
            recommendations: List of recommendations
            analysis_results: Full analysis results
            
        Returns:
            List of tagged recommendations (all as dicts)
        """
        tagged = []
        
        for rec in recommendations:
            # Convert to dict if needed
            if isinstance(rec, dict):
                rec_dict = rec.copy()
                message = rec_dict.get('message', rec_dict.get('text', str(rec)))
            else:
                rec_dict = {}
                message = str(rec)
            
            # Determine source
            source = self._determine_source(message, analysis_results)
            
            # Ensure message field exists
            if 'message' not in rec_dict and 'text' not in rec_dict:
                rec_dict['message'] = message
            
            # Add source tag
            rec_dict['source'] = source
            
            tagged.append(rec_dict)
        
        return tagged
    
    def _determine_source(self, message: str, analysis_results: Dict[str, Any]) -> str:
        """
        Determine recommendation source (system | user | config).
        
        Rules:
        - "system": System errors, technical issues, missing dependencies
        - "user": User action required (fix content, improve quality, etc.)
        - "config": Configuration issues (missing brand palette, font guidelines, etc.)
        """
        message_lower = message.lower()
        
        # System indicators
        if any(indicator in message_lower for indicator in self.SYSTEM_ERROR_INDICATORS):
            return 'system'
        
        # Config indicators
        config_keywords = [
            'not provided',
            'not configured',
            'missing.*palette',
            'missing.*guidelines',
            'missing.*config',
            'brand.*palette.*not',
            'font.*guidelines.*not',
            'observed only',
        ]
        if any(re.search(pattern, message_lower) for pattern in config_keywords):
            return 'config'
        
        # Default to user (actionable recommendations)
        return 'user'
    
    def _remove_user_blaming(
        self,
        recommendations: List[Dict[str, Any]],
        analyzer_statuses: Dict[str, str],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Remove user-blaming messages when analyzer status = failed.
        
        Args:
            recommendations: List of tagged recommendations
            analyzer_statuses: Dict mapping analyzer names to statuses
            analysis_results: Full analysis results
            
        Returns:
            Filtered recommendations
        """
        filtered = []
        removed_count = 0
        
        for rec in recommendations:
            message = rec.get('message', rec.get('text', ''))
            source = rec.get('source', 'user')
            
            # Check if this is a user-blaming message
            is_user_blaming = any(regex.search(message) for regex in self.user_blaming_regex)
            
            # Check if any analyzer failed
            has_failed_analyzer = any(
                status in ['failed', 'fail', 'error'] 
                for status in analyzer_statuses.values()
            )
            
            # Remove if user-blaming AND analyzer failed (system error, not user fault)
            if is_user_blaming and has_failed_analyzer and source == 'user':
                removed_count += 1
                logger.debug(f"[PostProcessor] Removed user-blaming message (analyzer failed): {message[:50]}...")
                continue
            
            filtered.append(rec)
        
        if removed_count > 0:
            logger.info(f"[PostProcessor] Removed {removed_count} user-blaming messages (analyzers failed)")
        
        return filtered
    
    def extract_analyzer_statuses(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract analyzer statuses from analysis results.
        
        Args:
            analysis_results: Full analysis results
            
        Returns:
            Dict mapping analyzer names to statuses
        """
        statuses = {}
        
        # Check model_results for analyzer statuses
        model_results = analysis_results.get('model_results', {})
        
        for key, value in model_results.items():
            if isinstance(value, dict):
                status = value.get('analyzerStatus', value.get('status', 'unknown'))
                statuses[key] = status
        
        # Also check top-level analyzer statuses
        for key in ['color_analysis', 'logo_analysis', 'copywriting_analysis', 'typography_analysis', 'ocr_result']:
            if key in analysis_results:
                value = analysis_results[key]
                if isinstance(value, dict):
                    status = value.get('analyzerStatus', value.get('status', 'unknown'))
                    statuses[key] = status
        
        return statuses

