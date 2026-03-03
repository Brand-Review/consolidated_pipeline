"""
Integration Test Suite for Scoring Engine
Tests the complete scoring pipeline with realistic normalized inputs
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brandguard.core.scoring import ScoringEngine
from brandguard.core.normalization import ProviderNormalizer


class ScoringIntegrationTest(unittest.TestCase):
    """Test complete scoring pipeline with normalized inputs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scoring_engine = ScoringEngine()
        self.normalizer = ProviderNormalizer()
    
    def test_perfect_content(self):
        """Test perfect content gets approve bucket"""
        normalized = {
            "grammar": {
                "errors": [],
                "spelling": [],
                "punctuation": [],
                "flags": []
            },
            "tone": {
                "tone": "energetic",
                "sentiment": "positive",
                "confidence_level": "high"
            },
            "brand": {
                "logo": {
                    "present": True,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": True,
                    "flags": []
                }
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        self.assertEqual(result["bucket"], "approve")
        self.assertGreaterEqual(result["final_score"], 85)
        self.assertEqual(result["grammar_score"], 100)
        self.assertGreater(result["tone_score"], 50)
        self.assertEqual(result["brand_score"], 100)
    
    def test_grammar_errors_reject(self):
        """Test grammar errors lead to reject bucket"""
        normalized = {
            "grammar": {
                "errors": ["Error 1", "Error 2", "Error 3", "Error 4"],
                "spelling": ["speling"],
                "punctuation": [],
                "flags": []
            },
            "tone": {
                "tone": "neutral",
                "sentiment": "neutral",
                "confidence_level": "balanced"
            },
            "brand": {
                "logo": {
                    "present": True,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": True,
                    "flags": []
                }
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        # Grammar score should be low
        self.assertLess(result["grammar_score"], 60)
        # Final score should be reject
        self.assertEqual(result["bucket"], "reject")
    
    def test_critical_grammar_flag(self):
        """Test critical grammar flag caps score at 60"""
        normalized = {
            "grammar": {
                "errors": [],
                "spelling": [],
                "punctuation": [],
                "flags": ["critical_grammar_error"]
            },
            "tone": {
                "tone": "neutral",
                "sentiment": "neutral",
                "confidence_level": "balanced"
            },
            "brand": {
                "logo": {
                    "present": True,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": True,
                    "flags": []
                }
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        # Grammar score should be capped at 60
        self.assertLessEqual(result["grammar_score"], 60)
    
    def test_missing_logo_reject(self):
        """Test missing logo leads to reject"""
        normalized = {
            "grammar": {
                "errors": [],
                "spelling": [],
                "punctuation": [],
                "flags": []
            },
            "tone": {
                "tone": "neutral",
                "sentiment": "neutral",
                "confidence_level": "balanced"
            },
            "brand": {
                "logo": {
                    "present": False,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": True,
                    "flags": []
                }
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        # Brand score should be low (100 - 40 = 60)
        self.assertEqual(result["brand_score"], 60)
        # Final score should be review or reject
        self.assertIn(result["bucket"], ["review", "reject"])
    
    def test_tone_misalignment(self):
        """Test tone misalignment affects score"""
        normalized = {
            "grammar": {
                "errors": [],
                "spelling": [],
                "punctuation": [],
                "flags": []
            },
            "tone": {
                "tone": "unknown",
                "sentiment": "negative",
                "confidence_level": "low"
            },
            "brand": {
                "logo": {
                    "present": True,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": True,
                    "flags": []
                }
            }
        }
        
        # Test without brand rules (should be conservative)
        result = self.scoring_engine.score_all(normalized)
        self.assertLess(result["tone_score"], 60)
        
        # Test with brand rules (tone mismatch)
        brand_rules = {
            "target_tone": "energetic",
            "target_sentiment": "positive"
        }
        result_with_rules = self.scoring_engine.score_all(normalized, brand_rules)
        self.assertLess(result_with_rules["tone_score"], 50)
    
    def test_partial_data_conservative(self):
        """Test missing sections get conservative scores"""
        # Only grammar provided
        normalized = {
            "grammar": {
                "errors": [],
                "spelling": [],
                "punctuation": [],
                "flags": []
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        # Should still produce valid scores
        self.assertIn("grammar_score", result)
        self.assertIn("tone_score", result)
        self.assertIn("brand_score", result)
        self.assertIn("final_score", result)
        self.assertIn("bucket", result)
        
        # Missing sections should be conservative (50 for tone, 0 for brand)
        self.assertEqual(result["tone_score"], 50)
        self.assertEqual(result["brand_score"], 0)
    
    def test_explanation_generation(self):
        """Test explanation builder generates reasons"""
        normalized = {
            "grammar": {
                "errors": ["Error 1", "Error 2"],
                "spelling": [],
                "punctuation": [],
                "flags": []
            },
            "tone": {
                "tone": "unknown",
                "sentiment": "neutral",
                "confidence_level": "low"
            },
            "brand": {
                "logo": {
                    "present": False,
                    "size_ok": True,
                    "placement_ok": True,
                    "clear_space_ok": True,
                    "flags": []
                },
                "colors": {
                    "primary_match": False,
                    "flags": []
                }
            }
        }
        
        result = self.scoring_engine.score_all(normalized)
        
        # Should have explanations
        self.assertIn("explanations", result)
        self.assertIsInstance(result["explanations"], list)
        self.assertGreater(len(result["explanations"]), 0)
        
        # Should mention grammar issues
        explanation_text = " ".join(result["explanations"]).lower()
        self.assertIn("grammar", explanation_text)
        self.assertIn("brand", explanation_text)
    
    def test_bucket_thresholds(self):
        """Test bucket thresholds are correct"""
        test_cases = [
            (100, "approve"),
            (85, "approve"),
            (84.9, "review"),
            (60, "review"),
            (59.9, "reject"),
            (0, "reject"),
        ]
        
        for score, expected_bucket in test_cases:
            bucket = self.scoring_engine.get_bucket(score)
            self.assertEqual(
                bucket,
                expected_bucket,
                f"Score {score} should map to {expected_bucket}, got {bucket}"
            )
    
    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        weights = self.scoring_engine.FINAL_WEIGHTS
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

