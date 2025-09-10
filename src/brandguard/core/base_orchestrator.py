"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description:Base Pipeline Orchestrator 
Core functionality for coordinating all BrandGuard models
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .model_imports import import_all_models, get_imported_models, is_models_loaded
from .color_analyzer import ColorAnalyzer
from .logo_analyzer import LogoAnalyzer
from .typography_analyzer import TypographyAnalyzer
from .copywriting_analyzer import CopywritingAnalyzer

logger = logging.getLogger(__name__)

class BasePipelineOrchestrator:
    """
    Base orchestrator that coordinates all BrandGuard models
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.analysis_results = {}
        self.current_analysis_id = None
        
        # Import and initialize models
        self.MODELS_LOADED = import_all_models()
        self.imported_models = get_imported_models()
        
        # Initialize analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize all analyzer components"""
        try:
            # Initialize color analyzer
            self.color_analyzer = ColorAnalyzer(self.settings, self.imported_models)
            
            # Initialize logo analyzer
            self.logo_analyzer = LogoAnalyzer(self.settings, self.imported_models)

            # Initialize typography analyzer
            self.typography_analyzer = TypographyAnalyzer(self.settings, self.imported_models)

            # Initialize copywriting analyzer
            self.copywriting_analyzer = CopywritingAnalyzer(self.settings, self.imported_models)
            
            logger.info("All analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            import traceback
            logger.error(f"Analyzer initialization traceback: {traceback.format_exc()}")
    
    def analyze_content(self, 
                       input_source: str, 
                       source_type: str = 'image',
                       analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method that coordinates all models
        
        Args:
            input_source: Path to file, text content, or URL
            source_type: Type of input ('image', 'document', 'text', 'url')
            analysis_options: Configuration options for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Generate analysis ID
            self.current_analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Initialize results structure
            results = {
                'analysis_id': self.current_analysis_id,
                'timestamp': datetime.now().isoformat(),
                'input_source': input_source,
                'source_type': source_type,
                'model_results': {},
                'overall_compliance': 0.0,
                'summary': '',
                'recommendations': []
            }
            
            # Route to appropriate analysis method
            if source_type == 'image':
                analysis_result = self._analyze_image(input_source, analysis_options)
            elif source_type == 'document':
                analysis_result = self._analyze_document(input_source, analysis_options)
            elif source_type == 'text':
                analysis_result = self._analyze_text(input_source, analysis_options)
            elif source_type == 'url':
                analysis_result = self._analyze_url(input_source, analysis_options)
            else:
                return {'error': f'Unsupported source type: {source_type}'}
            
            # Merge results
            results.update(analysis_result)
            
            # Calculate overall compliance
            results['overall_compliance'] = self._calculate_overall_compliance(results)
            
            # Generate summary and recommendations
            summary_data = self._generate_summary_and_recommendations(results)
            results.update(summary_data)
            
            # Store results
            self.analysis_results[self.current_analysis_id] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_image(self, image_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze image content"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            # Initialize results
            model_results = {}
            
            # Perform color analysis if enabled
            if analysis_options and analysis_options.get('color_analysis', {}).get('enabled', True):
                color_options = analysis_options.get('color_analysis', {})
                model_results['color_analysis'] = self.color_analyzer.analyze_colors(image, color_options)
            
            # Perform logo analysis if enabled
            if analysis_options and analysis_options.get('logo_analysis', {}).get('enabled', True):
                logo_options = analysis_options.get('logo_analysis', {})
                model_results['logo_analysis'] = self.logo_analyzer.analyze_logos(image, logo_options)
            
            # # Perform typography analysis if enabled
            # if analysis_options and analysis_options.get('typography_analysis', {}).get('enabled', True):
            #     typography_options = analysis_options.get('typography_analysis', {})
            #     # Typography analyzer expects (image, text_regions) not (image, options)
            #     model_results['typography_analysis'] = self.typography_analyzer.analyze_typography(image, None)
            
            # # Perform copywriting analysis if enabled
            # if analysis_options and analysis_options.get('copywriting_analysis', {}).get('enabled', True):
            #     copywriting_options = analysis_options.get('copywriting_analysis', {})
            #     model_results['copywriting_analysis'] = self.copywriting_analyzer.analyze_copywriting(image, copywriting_options)
            
            return {
                'model_results': model_results,
                'analysis_type': 'image_analysis'
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _analyze_document(self, document_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            # TODO: Implement document analysis
            return {
                'model_results': {},
                'analysis_type': 'document_analysis',
                'message': 'Document analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {'error': f'Document analysis failed: {str(e)}'}
    
    def _analyze_text(self, text_content: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            # TODO: Implement text analysis
            return {
                'model_results': {},
                'analysis_type': 'text_analysis',
                'message': 'Text analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {'error': f'Text analysis failed: {str(e)}'}
    
    def _analyze_url(self, url: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze URL content"""
        try:
            # TODO: Implement URL analysis
            return {
                'model_results': {},
                'analysis_type': 'url_analysis',
                'message': 'URL analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return {'error': f'URL analysis failed: {str(e)}'}
    
    def _calculate_overall_compliance(self, results: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        try:
            model_results = results.get('model_results', {})
            if not model_results:
                return 0.0
            
            scores = []
            
            # Color analysis score
            if 'color_analysis' in model_results:
                color_score = model_results['color_analysis'].get('brand_validation', {}).get('compliance_score', 0)
                scores.append(color_score)
            
            # Logo analysis score
            if 'logo_analysis' in model_results:
                logo_score = model_results['logo_analysis'].get('scores', {}).get('overall', 0)
                scores.append(logo_score)
            
            # TODO: Add other analysis scores
            
            if scores:
                return round(sum(scores) / len(scores), 2)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Compliance calculation failed: {e}")
            return 0.0
    
    def _generate_summary_and_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and recommendations"""
        try:
            model_results = results.get('model_results', {})
            overall_compliance = results.get('overall_compliance', 0)
            
            # Generate summary
            if overall_compliance >= 80:
                summary = "Excellent brand compliance"
            elif overall_compliance >= 60:
                summary = "Good brand compliance with minor issues"
            elif overall_compliance >= 40:
                summary = "Moderate brand compliance with several issues"
            else:
                summary = "Poor brand compliance requiring attention"
            
            # Generate recommendations
            recommendations = []
            
            # Color recommendations
            if 'color_analysis' in model_results:
                color_validation = model_results['color_analysis'].get('brand_validation', {})
                if not color_validation.get('valid', True):
                    recommendations.append("Review color palette to better align with brand guidelines")
            
            # Logo recommendations
            if 'logo_analysis' in model_results:
                logo_validation = model_results['logo_analysis'].get('placement_validation', {})
                if not logo_validation.get('valid', True):
                    recommendations.append("Adjust logo placement and sizing for better compliance")
            
            return {
                'summary': summary,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                'summary': 'Analysis completed with errors',
                'recommendations': ['Review analysis results for details']
            }
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of a specific analysis"""
        try:
            if analysis_id in self.analysis_results:
                return {
                    'analysis_id': analysis_id,
                    'status': 'completed',
                    'results': self.analysis_results[analysis_id]
                }
            else:
                return {'error': 'Analysis not found'}
                
        except Exception as e:
            logger.error(f"Failed to get analysis status: {e}")
            return {'error': f'Failed to get analysis status: {str(e)}'}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear analysis results
            self.analysis_results.clear()
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
