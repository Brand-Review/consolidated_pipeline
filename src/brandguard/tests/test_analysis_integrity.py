"""
Regression tests for analysis integrity
Ensures zero output never produces fake compliance scores
"""

import unittest
from unittest.mock import Mock, MagicMock
from src.brandguard.core.base_orchestrator import BasePipelineOrchestrator


class TestAnalysisIntegrity(unittest.TestCase):
    """Test that analysis integrity guards work correctly"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal orchestrator with mocked analyzers
        self.orchestrator = BasePipelineOrchestrator.__new__(BasePipelineOrchestrator)
        self.orchestrator.current_analysis_id = 'test_analysis_001'
        
        # Mock analyzers to prevent initialization errors
        self.orchestrator.color_analyzer = Mock()
        self.orchestrator.logo_analyzer = Mock()
        self.orchestrator.typography_analyzer = Mock()
        self.orchestrator.copywriting_analyzer = Mock()
        self.orchestrator.analysis_results = {}
    
    def test_no_model_outputs_does_not_score(self):
        """Test that empty model results produces appropriate summary, not fake 'Poor compliance'"""
        result = self.orchestrator._generate_summary_and_recommendations({
            'model_results': {},
            'overall_compliance': 0
        })
        
        # Should NOT return "Poor brand compliance requiring attention"
        assert result['summary'] != "Poor brand compliance requiring attention"
        # Should return appropriate failure message
        assert 'could not be completed' in result['summary'].lower() or 'not be completed' in result['summary'].lower()
        assert len(result.get('recommendations', [])) > 0
    
    def test_normalization_prevents_zero_score(self):
        """Test that normalization preserves valid scores"""
        raw_results = {
            'logo_analysis': {
                'scores': {'overall': 72}
            }
        }
        
        normalized = self.orchestrator._normalize_model_results(raw_results)
        
        assert normalized['logo_analysis']['scores']['overall'] == 72
        assert 'logo_analysis' in normalized
    
    def test_normalization_handles_empty_results(self):
        """Test that normalization handles empty/missing results gracefully"""
        raw_results = {}
        
        normalized = self.orchestrator._normalize_model_results(raw_results)
        
        assert isinstance(normalized, dict)
        assert len(normalized) == 0
    
    def test_normalization_handles_color_analysis(self):
        """Test that color analysis scores are normalized correctly"""
        raw_results = {
            'color_analysis': {
                'brand_validation': {
                    'compliance_score': 85
                }
            }
        }
        
        normalized = self.orchestrator._normalize_model_results(raw_results)
        
        assert normalized['color_analysis']['brand_validation']['compliance_score'] == 85
    
    def test_normalization_handles_logo_analysis(self):
        """Test that logo analysis scores are normalized correctly"""
        raw_results = {
            'logo_analysis': {
                'scores': {'overall': 70},
                'placement_validation': {'valid': True}
            }
        }
        
        normalized = self.orchestrator._normalize_model_results(raw_results)
        
        assert normalized['logo_analysis']['scores']['overall'] == 70
        assert normalized['logo_analysis']['placement_validation']['valid'] == True
    
    def test_normalization_handles_partial_results(self):
        """Test that normalization handles partial results (some modules missing)"""
        raw_results = {
            'logo_analysis': {
                'scores': {'overall': 75}
            }
            # color_analysis is missing
        }
        
        normalized = self.orchestrator._normalize_model_results(raw_results)
        
        assert 'logo_analysis' in normalized
        assert 'color_analysis' not in normalized
        assert normalized['logo_analysis']['scores']['overall'] == 75
    
    def test_calculate_compliance_with_normalized_results(self):
        """Test that compliance calculation works with normalized results"""
        results = {
            'model_results': {
                'color_analysis': {
                    'brand_validation': {
                        'compliance_score': 80
                    }
                },
                'logo_analysis': {
                    'scores': {
                        'overall': 70
                    }
                }
            }
        }
        
        score = self.orchestrator._calculate_overall_compliance(results)
        
        # Should average the two scores: (80 + 70) / 2 = 75
        assert score == 75.0
    
    def test_calculate_compliance_with_zero_scores_excluded(self):
        """Test that zero scores are excluded from compliance calculation"""
        results = {
            'model_results': {
                'color_analysis': {
                    'brand_validation': {
                        'compliance_score': 0  # Zero score should be excluded
                    }
                },
                'logo_analysis': {
                    'scores': {
                        'overall': 80  # Only valid score
                    }
                }
            }
        }
        
        score = self.orchestrator._calculate_overall_compliance(results)
        
        # Should only use logo score: 80
        assert score == 80.0


if __name__ == '__main__':
    unittest.main()

