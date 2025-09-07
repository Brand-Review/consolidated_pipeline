#!/usr/bin/env python3
"""
Test color analysis with real models
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, 'src')

from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
from brandguard.config.settings import Settings

def create_test_image():
    """Create a test image with known colors"""
    # Create a 200x200 image with specific colors
    img_array = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add different colored regions
    img_array[0:100, 0:100] = [255, 0, 0]      # Red
    img_array[0:100, 100:200] = [0, 255, 0]    # Green
    img_array[100:200, 0:100] = [0, 0, 255]    # Blue
    img_array[100:200, 100:200] = [255, 255, 0] # Yellow
    
    return img_array

def test_color_analysis():
    """Test color analysis functionality"""
    print("🔍 Testing Color Analysis with Real Models...")
    
    try:
        # Create test settings
        settings = Settings()
        
        # Initialize pipeline
        pipeline = PipelineOrchestrator(settings)
        print("✅ Pipeline initialized successfully")
        
        # Create test image
        test_image = create_test_image()
        print("✅ Test image created")
        
        # Test color analysis
        print("\n🎨 Running color analysis...")
        color_options = {
            'enabled': True,
            'n_colors': 8,
            'color_tolerance': 0.2,
            'enable_contrast_check': True,
            'brand_palette': '#FF0000,#00FF00,#0000FF'  # Red, Green, Blue
        }
        
        result = pipeline._perform_color_analysis(test_image, color_options)
        
        if 'error' in result:
            print(f"❌ Color analysis failed: {result['error']}")
            return
        
        print("✅ Color analysis completed successfully!")
        print(f"📊 Compliance Score: {(result['compliance_score'] * 100):.1f}%")
        
        # Display results
        if 'dominant_colors' in result:
            print(f"\n🎨 Dominant Colors ({len(result['dominant_colors'])}):")
            for i, color in enumerate(result['dominant_colors']):
                print(f"  Color {i+1}: {color['hex']} (RGB: {color['rgb']})")
        
        if 'palette_validation' in result:
            validation = result['palette_validation']
            print(f"\n✅ Palette Validation:")
            print(f"  Compliant: {validation['compliant_colors']}/{validation['total_colors']}")
            print(f"  Non-compliant: {validation['non_compliant_colors']}")
            print(f"  Score: {(validation['compliance_score'] * 100):.1f}%")
        
        if 'analysis_settings' in result:
            settings = result['analysis_settings']
            print(f"\n⚙️ Analysis Settings:")
            print(f"  Model Used: {settings['model_used']}")
            print(f"  Colors Extracted: {settings['n_colors_extracted']}")
            print(f"  Color Tolerance: {settings['color_tolerance_used']}")
        
        print("\n🎯 Color Analysis Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_color_analysis()
