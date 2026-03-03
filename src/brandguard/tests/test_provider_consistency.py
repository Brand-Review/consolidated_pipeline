"""
Provider Consistency Test
Tests that same input produces same bucket across different providers
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brandguard.core.scoring import ScoringEngine
from brandguard.core.normalization import ProviderNormalizer


class ProviderConsistencyTest(unittest.TestCase):
    """Test that different providers produce same bucket"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scoring_engine = ScoringEngine()
        self.normalizer = ProviderNormalizer()
    
    def test_openrouter_vs_vllm_same_bucket(self):
        """Test OpenRouter and vLLM produce same bucket"""
        # Simulate OpenRouter response
        openrouter_response = {
            "grammar_analysis": {
                "grammar_errors": ["Error 1"],
                "spelling_errors": [],
                "punctuation_issues": []
            },
            "tone_analysis": {
                "tone": "energetic",
                "sentiment": "positive"
            },
            "text": "Test text"
        }
        
        # Simulate vLLM response (different structure, same content)
        vllm_response = {
            "version": "analysis_v1",
            "detected": {
                "text": "Test text"
            },
            "observations": {
                "grammar_errors": ["Error 1"],
                "spelling_errors": [],
                "punctuation_issues": [],
                "tone": "energetic",
                "sentiment": "positive"
            },
            "flags": ["grammar_error_detected"],
            "raw_metrics": {},
            "confidence": 0.8
        }
        
        # Normalize both
        normalized_openrouter = self.normalizer.normalize(openrouter_response, provider="openrouter")
        normalized_vllm = self.normalizer.normalize(vllm_response, provider="vllm")
        
        # Convert to scoring format
        def to_scoring_format(normalized):
            return {
                "grammar": {
                    "errors": normalized.get("observations", {}).get("grammar_errors", []),
                    "spelling": normalized.get("observations", {}).get("spelling_errors", []),
                    "punctuation": normalized.get("observations", {}).get("punctuation_issues", []),
                    "flags": normalized.get("flags", [])
                },
                "tone": {
                    "tone": normalized.get("observations", {}).get("tone", "unknown"),
                    "sentiment": normalized.get("observations", {}).get("sentiment", "neutral"),
                    "confidence_level": normalized.get("observations", {}).get("confidence_level", "balanced")
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
        
        scoring_openrouter = to_scoring_format(normalized_openrouter)
        scoring_vllm = to_scoring_format(normalized_vllm)
        
        # Score both
        result_openrouter = self.scoring_engine.score_all(scoring_openrouter)
        result_vllm = self.scoring_engine.score_all(scoring_vllm)
        
        # Buckets must be identical
        bucket_openrouter = result_openrouter['scores']['bucket']
        bucket_vllm = result_vllm['scores']['bucket']
        
        self.assertEqual(
            bucket_openrouter,
            bucket_vllm,
            f"Buckets differ: OpenRouter={bucket_openrouter}, vLLM={bucket_vllm}"
        )
        
        # Scores can drift slightly, but should be close
        score_openrouter = result_openrouter['scores']['overall']
        score_vllm = result_vllm['scores']['overall']
        score_diff = abs(score_openrouter - score_vllm)
        
        self.assertLessEqual(
            score_diff,
            5,
            f"Score drift too large: OpenRouter={score_openrouter}, vLLM={score_vllm}, diff={score_diff}"
        )
    
    def test_normalization_produces_same_structure(self):
        """Test that normalization produces same structure regardless of provider"""
        providers = ["openrouter", "vllm", "ollama"]
        
        # Different raw formats from different providers
        raw_responses = {
            "openrouter": {
                "grammar_analysis": {
                    "grammar_errors": ["Error"],
                    "spelling_errors": []
                },
                "tone_analysis": {
                    "tone": "neutral"
                }
            },
            "vllm": {
                "observations": {
                    "grammar_errors": ["Error"],
                    "spelling_errors": [],
                    "tone": "neutral"
                }
            },
            "ollama": {
                "grammar": {
                    "errors": ["Error"]
                },
                "tone": {
                    "tone": "neutral"
                }
            }
        }
        
        normalized_results = {}
        for provider in providers:
            if provider in raw_responses:
                normalized = self.normalizer.normalize(raw_responses[provider], provider=provider)
                normalized = self.normalizer.fill_missing_fields(normalized)
                normalized_results[provider] = normalized
        
        # All should have same canonical structure
        for provider, normalized in normalized_results.items():
            self.assertIn("version", normalized)
            self.assertIn("detected", normalized)
            self.assertIn("observations", normalized)
            self.assertIn("flags", normalized)
            self.assertIn("raw_metrics", normalized)
            self.assertIn("confidence", normalized)


if __name__ == '__main__':
    unittest.main(verbosity=2)

