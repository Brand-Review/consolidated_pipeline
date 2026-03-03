"""
Test suite for logo candidate filtering
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from brandguard.core.filters import filter_logo_candidates


class LogoFilterTest(unittest.TestCase):
    """Test logo candidate filtering"""
    
    def test_confidence_filter(self):
        """Test confidence filter removes low-confidence detections"""
        # Use larger bboxes to pass area ratio filter
        # Place in top-left to avoid center zone position bias
        detections = [
            {'bbox': [10, 10, 60, 60], 'confidence': 0.9},  # Keep (50x50 = 0.01 area ratio, top-left)
            {'bbox': [70, 70, 120, 120], 'confidence': 0.5},  # Remove (< 0.75)
            {'bbox': [10, 70, 60, 120], 'confidence': 0.8},  # Keep (top-left, not center)
            {'bbox': [70, 10, 120, 60], 'confidence': 0.6},  # Remove
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        self.assertEqual(len(filtered), 2)
        # Sort by confidence to ensure order
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        self.assertEqual(filtered[0]['confidence'], 0.9)
        self.assertEqual(filtered[1]['confidence'], 0.8)
    
    def test_area_ratio_filter(self):
        """Test area ratio filter"""
        # Very small detection (should be removed)
        detections = [
            {'bbox': [10, 10, 15, 15], 'confidence': 0.9},  # Too small
            {'bbox': [10, 10, 100, 100], 'confidence': 0.9},  # Good size
            {'bbox': [10, 10, 400, 400], 'confidence': 0.9},  # Too large (> 25%)
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        # Should only keep the middle one
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(filtered[0]['size_ratio'], (90 * 90) / (500 * 500), places=3)
    
    def test_aspect_ratio_filter(self):
        """Test aspect ratio filter"""
        detections = [
            {'bbox': [10, 10, 50, 100], 'confidence': 0.9},  # aspect = 0.4 (too narrow)
            {'bbox': [10, 10, 90, 50], 'confidence': 0.9},  # aspect = 1.6 (good, 80/50)
            {'bbox': [10, 10, 400, 50], 'confidence': 0.9},  # aspect = 8.0 (too wide)
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        # Should only keep the middle one (aspect ratio 0.8-3.5)
        self.assertEqual(len(filtered), 1)
        # Check it's in the valid range
        self.assertGreaterEqual(filtered[0]['aspect_ratio'], 0.8)
        self.assertLessEqual(filtered[0]['aspect_ratio'], 3.5)
    
    def test_nms_filter(self):
        """Test Non-Maximum Suppression removes overlapping detections"""
        # Two overlapping detections - should keep highest confidence
        detections = [
            {'bbox': [10, 10, 100, 100], 'confidence': 0.8},  # Lower confidence
            {'bbox': [20, 20, 110, 110], 'confidence': 0.9},  # Higher confidence, overlaps
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        # Should keep only the higher confidence one
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['confidence'], 0.9)
    
    def test_position_bias(self):
        """Test position bias reduces confidence for center zone"""
        # Center detection
        detections = [
            {'bbox': [200, 200, 300, 300], 'confidence': 0.9},  # Center of 500x500
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['zone'], 'center')
        # Confidence should be reduced by 0.7
        self.assertAlmostEqual(filtered[0]['confidence'], 0.9 * 0.7, places=2)
        self.assertEqual(filtered[0]['confidence_original'], 0.9)
    
    def test_zone_detection(self):
        """Test zone detection works correctly"""
        detections = [
            {'bbox': [50, 50, 150, 150], 'confidence': 0.9},  # Top-left
            {'bbox': [350, 50, 450, 150], 'confidence': 0.9},  # Top-right
            {'bbox': [50, 350, 150, 450], 'confidence': 0.9},  # Bottom-left
            {'bbox': [350, 350, 450, 450], 'confidence': 0.9},  # Bottom-right
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        self.assertEqual(len(filtered), 4)
        zones = [d['zone'] for d in filtered]
        self.assertIn('top-left', zones)
        self.assertIn('top-right', zones)
        self.assertIn('bottom-left', zones)
        self.assertIn('bottom-right', zones)
    
    def test_complete_filtering_pipeline(self):
        """Test complete filtering pipeline with all filters"""
        detections = [
            # Good logo candidate
            {'bbox': [10, 10, 100, 100], 'confidence': 0.9},
            # Low confidence (removed)
            {'bbox': [110, 110, 200, 200], 'confidence': 0.5},
            # Too small (removed)
            {'bbox': [210, 210, 215, 215], 'confidence': 0.9},
            # Too large (removed)
            {'bbox': [10, 10, 400, 400], 'confidence': 0.9},
            # Bad aspect ratio (removed)
            {'bbox': [10, 10, 500, 50], 'confidence': 0.9},
            # Overlapping with first (removed by NMS)
            {'bbox': [20, 20, 110, 110], 'confidence': 0.85},
        ]
        
        filtered = filter_logo_candidates(detections, 500, 500)
        
        # Should only keep the first one (best candidate)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['confidence'], 0.9)
        self.assertIn('size_ratio', filtered[0])
        self.assertIn('aspect_ratio', filtered[0])
        self.assertIn('zone', filtered[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)

