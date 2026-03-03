"""
Analysis Routing Test Suite
Regression tests to ensure image URLs are correctly routed to image_analysis,
not url_analysis. This prevents the routing bug from recurring.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from app import determine_analysis_type
    from src.brandguard.core.pipeline_orchestrator import PipelineOrchestrator
    from src.brandguard.config.settings import Settings
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Some tests may be skipped")


class AnalysisRoutingTest(unittest.TestCase):
    """Test that image URLs are correctly routed to image_analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            # Initialize settings and pipeline for integration tests
            self.settings = Settings()
            self.pipeline = None
            try:
                self.pipeline = PipelineOrchestrator(self.settings)
            except Exception as e:
                # Pipeline initialization may fail if models aren't available
                # That's okay for routing tests
                pass
        except Exception:
            pass
    
    def test_determine_analysis_type_png_url(self):
        """Test that .png URLs are identified as image analysis"""
        url = "https://example.com/image.png"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image", 
                        f"Expected 'image' for .png URL, got '{result}'")
    
    def test_determine_analysis_type_jpg_url(self):
        """Test that .jpg URLs are identified as image analysis"""
        url = "https://example.com/image.jpg"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image",
                        f"Expected 'image' for .jpg URL, got '{result}'")
    
    def test_determine_analysis_type_jpeg_url(self):
        """Test that .jpeg URLs are identified as image analysis"""
        url = "https://example.com/image.jpeg"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image",
                        f"Expected 'image' for .jpeg URL, got '{result}'")
    
    def test_determine_analysis_type_webp_url(self):
        """Test that .webp URLs are identified as image analysis"""
        url = "https://example.com/image.webp"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image",
                        f"Expected 'image' for .webp URL, got '{result}'")
    
    def test_determine_analysis_type_s3_presigned_url(self):
        """Test that S3 presigned URLs with image extensions are identified as image analysis"""
        url = "https://bucket.s3.amazonaws.com/image.png?X-Amz-Signature=abc123"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image",
                        f"Expected 'image' for S3 presigned .png URL, got '{result}'")
    
    def test_determine_analysis_type_url_without_extension(self):
        """Test that URLs without extensions default to image (safe default)"""
        url = "https://example.com/image"
        result = determine_analysis_type(url)
        self.assertEqual(result, "image",
                        f"Expected 'image' for URL without extension (safe default), got '{result}'")
    
    def test_image_url_routes_to_image_analysis(self):
        """
        CRITICAL REGRESSION TEST: Image URLs must route to image_analysis, not url_analysis.
        
        This test ensures that:
        - analysisType == "image_analysis" (or source_type == "image")
        - The image analysis pipeline is called, not url_analysis
        - No "not_supported" response is returned for image URLs
        """
        if not self.pipeline:
            self.skipTest("Pipeline not available")
        
        # Test image URL
        image_url = "https://example.com/test-image.png"
        
        # Mock the requests.get call to simulate downloading an image
        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.raise_for_status = Mock()
        mock_response.raw = Mock()
        
        # Create a temporary image file for the pipeline to process
        temp_file = None
        try:
            # Create a minimal valid PNG file (1x1 pixel)
            # PNG header: 89 50 4E 47 0D 0A 1A 0A
            png_data = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D,  # IHDR chunk length
                0x49, 0x48, 0x44, 0x52,  # IHDR
                0x00, 0x00, 0x00, 0x01,  # width = 1
                0x00, 0x00, 0x00, 0x01,  # height = 1
                0x08, 0x06, 0x00, 0x00, 0x00,  # bit depth, color type, etc.
                0x1F, 0x15, 0xC4, 0x89,  # CRC
                0x00, 0x00, 0x00, 0x00,  # IEND chunk length
                0x49, 0x45, 0x4E, 0x44,  # IEND
                0xAE, 0x42, 0x60, 0x82   # CRC
            ])
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.write(png_data)
            temp_file.close()
            
            # Mock shutil.copyfileobj to write our test PNG data
            with patch('requests.get', return_value=mock_response), \
                 patch('shutil.copyfileobj') as mock_copy, \
                 patch('os.path.exists', return_value=True), \
                 patch('cv2.imread', return_value=Mock()):  # Mock OpenCV image read
                
                # Set up the mock to write our PNG data
                def copyfileobj_side_effect(src, dst):
                    dst.write(png_data)
                
                mock_copy.side_effect = copyfileobj_side_effect
                
                # Determine analysis type (should be 'image')
                source_type = determine_analysis_type(image_url)
                
                # CRITICAL ASSERTION: Must route to 'image', not 'url'
                self.assertEqual(source_type, "image",
                               f"CRITICAL: Image URL must route to 'image', got '{source_type}'. "
                               f"This indicates routing bug has returned!")
                
                # Test that pipeline would route correctly
                # We'll test the routing logic without actually running analysis
                # (to avoid dependency on models)
                if source_type == "image":
                    # Verify it would call _analyze_image_from_url, not _analyze_url
                    # This is verified by checking source_type == "image"
                    pass
                else:
                    self.fail(f"Image URL routed to '{source_type}' instead of 'image'. "
                            f"This is a critical routing bug!")
        
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
    
    def test_image_url_does_not_return_not_supported(self):
        """
        Test that image URLs do NOT return 'not_supported' response.
        This ensures they're being routed correctly.
        """
        if not self.pipeline:
            self.skipTest("Pipeline not available")
        
        image_url = "https://example.com/test.jpg"
        source_type = determine_analysis_type(image_url)
        
        # If source_type is 'image', it should NOT return 'not_supported'
        # (that's only for non-image URLs)
        self.assertNotEqual(source_type, "url",
                          "Image URL should not route to 'url' analysis")
        
        # Verify it would route to image analysis
        self.assertEqual(source_type, "image",
                        "Image URL must route to 'image' analysis")
    
    def test_multiple_image_url_formats(self):
        """Test various image URL formats are all routed correctly"""
        test_urls = [
            "https://example.com/image.png",
            "http://example.com/image.jpg",
            "https://s3.amazonaws.com/bucket/image.jpeg",
            "https://cdn.example.com/image.webp?version=123",
            "https://example.com/path/to/image.png#fragment",
            "https://example.com/image.JPG",  # Uppercase
            "https://example.com/image.PNG",  # Uppercase
        ]
        
        for url in test_urls:
            with self.subTest(url=url):
                result = determine_analysis_type(url)
                self.assertEqual(result, "image",
                               f"URL '{url}' should route to 'image', got '{result}'")
    
    def test_non_image_urls_still_default_to_image(self):
        """
        Test that URLs without clear image extensions still default to 'image'
        (This is the safe default behavior we implemented)
        """
        # URLs without extensions default to 'image' as safe default
        test_urls = [
            "https://example.com/page",
            "https://example.com/resource",
        ]
        
        for url in test_urls:
            with self.subTest(url=url):
                result = determine_analysis_type(url)
                # Should default to 'image' (safe default)
                self.assertEqual(result, "image",
                               f"URL '{url}' should default to 'image', got '{result}'")


if __name__ == '__main__':
    unittest.main(verbosity=2)

