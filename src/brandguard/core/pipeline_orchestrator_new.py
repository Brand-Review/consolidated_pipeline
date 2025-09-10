"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description: Main orchestrator that coordinates all BrandReview models
"""

from .base_orchestrator import BasePipelineOrchestrator

class PipelineOrchestrator(BasePipelineOrchestrator):
    """
    Main Pipeline Orchestrator


    Inherits from BasePipelineOrchestrator and can be extended with additional functionality
    """
    
    def __init__(self, settings):
        super().__init__(settings)
