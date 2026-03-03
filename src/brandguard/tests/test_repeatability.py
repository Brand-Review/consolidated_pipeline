"""
Repeatability Test Suite
Tests that same input produces same bucket decision across multiple runs
"""

import unittest
import logging
from typing import Dict, Any, List
import statistics

logger = logging.getLogger(__name__)


class RepeatabilityTest(unittest.TestCase):
    """Test repeatability of analysis results"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from brandguard.core.scoring import ScoringEngine
            from brandguard.core.normalization import ProviderNormalizer
            self.scoring_engine = ScoringEngine()
            self.normalizer = ProviderNormalizer()
        except ImportError:
            self.skipTest("Scoring engine or normalizer not available")
    
    def test_grammar_scoring_repeatability(self):
        """Test that grammar scoring is deterministic"""
        observations = {
            "grammar_errors": ["Subject-verb disagreement"],
            "spelling_errors": ["teh"],
            "punctuation_issues": ["Missing period"],
            "detected_text": "This is a test."
        }
        
        # Run 10 times
        scores = []
        for _ in range(10):
            score = self.scoring_engine.score_grammar(observations)
            scores.append(score)
        
        # All scores should be identical
        self.assertEqual(len(set(scores)), 1, f"Scores varied: {scores}")
        self.assertGreater(scores[0], 0)
        self.assertLessEqual(scores[0], 100)
    
    def test_bucket_decision_repeatability(self):
        """Test that bucket decisions are deterministic"""
        test_cases = [
            (85, "approve"),
            (60, "review"),
            (30, "reject"),
            (100, "approve"),
            (0, "reject")
        ]
        
        for score, expected_bucket in test_cases:
            buckets = []
            for _ in range(10):
                bucket = self.scoring_engine.get_bucket(score)
                buckets.append(bucket)
            
            # All buckets should be identical
            self.assertEqual(len(set(buckets)), 1, f"Buckets varied for score {score}: {buckets}")
            self.assertEqual(buckets[0], expected_bucket, f"Expected {expected_bucket}, got {buckets[0]}")
    
    def test_special_rule_sing_up(self):
        """Test special rule for 'Sing up' capitalization error"""
        observations = {
            "grammar_errors": [],
            "spelling_errors": [],
            "punctuation_issues": [],
            "detected_text": "Sing up for our newsletter"
        }
        
        score = self.scoring_engine.score_grammar(observations)
        
        # Should be capped at 60 for this error
        self.assertLessEqual(score, 60, f"Score should be <= 60 for 'Sing up' error, got {score}")
    
    def test_compliance_score_calculation(self):
        """Test compliance score calculation is deterministic"""
        grammar_obs = {
            "grammar_errors": ["Error 1"],
            "spelling_errors": [],
            "punctuation_issues": [],
            "detected_text": "Test text"
        }
        
        tone_obs = {
            "tone": "neutral",
            "sentiment": "positive"
        }
        
        # Run 10 times
        results = []
        for _ in range(10):
            result = self.scoring_engine.calculate_final_score(
                grammar_observations=grammar_obs,
                tone_observations=tone_obs
            )
            results.append(result)
        
        # All compliance scores should be identical
        compliance_scores = [r["compliance_score"] for r in results]
        self.assertEqual(len(set(compliance_scores)), 1, f"Compliance scores varied: {compliance_scores}")
        
        # All buckets should be identical
        buckets = [r["bucket"] for r in results]
        self.assertEqual(len(set(buckets)), 1, f"Buckets varied: {buckets}")
    
    def test_normalization_consistency(self):
        """Test that normalization produces consistent output"""
        raw_responses = [
            {
                "version": "analysis_v1",
                "detected": {"text": "Test"},
                "observations": {"grammar_errors": ["Error"]},
                "flags": [],
                "raw_metrics": {},
                "confidence": 0.5
            },
            {
                "grammar_analysis": {
                    "grammar_errors": ["Error"]
                },
                "text": "Test"
            }
        ]
        
        normalized_results = []
        for raw in raw_responses:
            normalized = self.normalizer.normalize(raw, provider="test")
            normalized = self.normalizer.fill_missing_fields(normalized)
            normalized_results.append(normalized)
        
        # All normalized results should have same structure
        for result in normalized_results:
            self.assertIn("version", result)
            self.assertIn("detected", result)
            self.assertIn("observations", result)
            self.assertIn("flags", result)
            self.assertIn("raw_metrics", result)
            self.assertIn("confidence", result)
    
    def test_score_variance_threshold(self):
        """Test that score variance is within acceptable threshold"""
        observations = {
            "grammar_errors": ["Error"],
            "spelling_errors": [],
            "punctuation_issues": [],
            "detected_text": "Test text with error"
        }
        
        # Run 10 times
        scores = []
        for _ in range(10):
            score = self.scoring_engine.score_grammar(observations)
            scores.append(score)
        
        # Calculate variance
        if len(set(scores)) > 1:
            variance = statistics.variance(scores)
            self.assertLessEqual(variance, 5, f"Score variance {variance} exceeds threshold of 5. Scores: {scores}")
        else:
            # Perfect repeatability
            self.assertEqual(len(set(scores)), 1)


if __name__ == '__main__':
    unittest.main()

