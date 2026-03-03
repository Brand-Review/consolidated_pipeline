"""
Regression tests for scoring integrity.
Ensures scoring logic never regresses and handles edge cases correctly.
"""
import unittest
from unittest.mock import Mock
from src.brandguard.core.base_orchestrator import BasePipelineOrchestrator


class TestScoringIntegrity(unittest.TestCase):
    """Test that scoring logic is correct and deterministic"""
    
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
    
    def test_logo_alone_cannot_force_100(self):
        """Test that logo alone cannot force 100% overall score"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "logo_analysis": {"scores": {"overall": 100}},
                "color_analysis": {"brand_validation": {"compliance_score": 0}},
                "typography_analysis": {"typographyScore": 0.4},
                "copywriting_analysis": {"copywritingScore": 0.25},
            }
        })
        
        # With weights: color=0.3, logo=0.3, typography=0.2, copywriting=0.2
        # Expected: (0 * 0.3) + (100 * 0.3) + (40 * 0.2) + (25 * 0.2) = 0 + 30 + 8 + 5 = 43
        # Should be < 60
        assert result < 60, f"Logo alone should not force high score, got {result}"
        assert result == 43.0, f"Expected 43.0, got {result}"
    
    def test_zero_color_penalizes_score(self):
        """Test that zero color score penalizes overall score"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "color_analysis": {"brand_validation": {"compliance_score": 0}},
                "logo_analysis": {"scores": {"overall": 100}},
                "typography_analysis": {"typographyScore": 1.0},
                "copywriting_analysis": {"copywritingScore": 1.0},
            }
        })
        
        # With weights: color=0.3, logo=0.3, typography=0.2, copywriting=0.2
        # Expected: (0 * 0.3) + (100 * 0.3) + (100 * 0.2) + (100 * 0.2) = 0 + 30 + 20 + 20 = 70
        # Should be < 70 (actually 70, but let's verify it's not inflated)
        assert result == 70.0, f"Expected 70.0 with zero color, got {result}"
        assert result < 100, f"Zero color should prevent perfect score, got {result}"
    
    def test_weighted_average_correct(self):
        """Test that weighted average is calculated correctly"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "color_analysis": {"brand_validation": {"compliance_score": 50}},
                "logo_analysis": {"scores": {"overall": 80}},
                "typography_analysis": {"typographyScore": 0.6},
                "copywriting_analysis": {"copywritingScore": 0.7},
            }
        })
        
        # Expected: (50 * 0.3) + (80 * 0.3) + (60 * 0.2) + (70 * 0.2) = 15 + 24 + 12 + 14 = 65
        assert result == 65.0, f"Expected 65.0, got {result}"
    
    def test_missing_modules_default_to_zero(self):
        """Test that missing modules default to zero score"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "color_analysis": {"brand_validation": {"compliance_score": 100}},
                # Missing logo, typography, copywriting
            }
        })
        
        # Expected: (100 * 0.3) + (0 * 0.3) + (0 * 0.2) + (0 * 0.2) = 30
        assert result == 30.0, f"Expected 30.0 with only color, got {result}"
    
    def test_empty_model_results_returns_zero(self):
        """Test that empty model results returns zero"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {}
        })
        
        assert result == 0.0, f"Expected 0.0 for empty results, got {result}"
    
    def test_no_model_results_returns_zero(self):
        """Test that no model results returns zero"""
        result = self.orchestrator._calculate_overall_compliance({
            # No model_results key
        })
        
        assert result == 0.0, f"Expected 0.0 for no model_results, got {result}"
    
    def test_typography_score_conversion(self):
        """Test that typography score (0-1) is correctly converted to 0-100"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "color_analysis": {"brand_validation": {"compliance_score": 0}},
                "logo_analysis": {"scores": {"overall": 0}},
                "typography_analysis": {"typographyScore": 0.75},
                "copywriting_analysis": {"copywritingScore": 0},
            }
        })
        
        # Expected: (0 * 0.3) + (0 * 0.3) + (75 * 0.2) + (0 * 0.2) = 15
        assert result == 15.0, f"Expected 15.0 with typography=0.75, got {result}"
    
    def test_copywriting_score_conversion(self):
        """Test that copywriting score (0-1) is correctly converted to 0-100"""
        result = self.orchestrator._calculate_overall_compliance({
            "model_results": {
                "color_analysis": {"brand_validation": {"compliance_score": 0}},
                "logo_analysis": {"scores": {"overall": 0}},
                "typography_analysis": {"typographyScore": 0},
                "copywriting_analysis": {"copywritingScore": 0.85},
            }
        })
        
        # Expected: (0 * 0.3) + (0 * 0.3) + (0 * 0.2) + (85 * 0.2) = 17
        assert result == 17.0, f"Expected 17.0 with copywriting=0.85, got {result}"


if __name__ == '__main__':
    unittest.main()

