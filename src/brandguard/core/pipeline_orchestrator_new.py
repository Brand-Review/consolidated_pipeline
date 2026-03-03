"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description: Main orchestrator that coordinates all BrandReview models
"""

from .base_orchestrator import BasePipelineOrchestrator
import logging

logger = logging.getLogger(__name__)

class PipelineOrchestrator(BasePipelineOrchestrator):
    """
    Main Pipeline Orchestrator


    Inherits from BasePipelineOrchestrator and can be extended with additional functionality
    """
    
    def __init__(self, settings):
        super().__init__(settings)
    
    def analyze_content(self, 
                       input_source: str, 
                       source_type: str = 'image',
                       analysis_options = None):
        """
        Override analyze_content to add defensive routing for image URLs.
        CRITICAL: Image URLs must NEVER reach url_analysis.
        """
        # CRITICAL: Re-validate source_type for image URLs/files
        # This is a defensive check in case source_type was set incorrectly upstream
        if input_source:
            input_lower = str(input_source).lower()
            url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
            is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                      any(url_path.endswith(ext) for ext in image_extensions)
            
            if is_image and source_type != 'image':
                # CRITICAL: Force image URLs to use image analysis, regardless of source_type
                logger.warning(
                    f"[Routing] CRITICAL: Image URL/file '{input_source[:100]}' was routed to "
                    f"'{source_type}' but should be 'image'. Forcing correction."
                )
                source_type = 'image'
        
        # Call parent method with corrected source_type
        return super().analyze_content(input_source, source_type, analysis_options)
