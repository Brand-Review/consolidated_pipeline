#!/usr/bin/env python3
"""
Simple test for consolidated pipeline with and without images
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_simple_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    text = "SALE 50% OFF"
    font = cv2.FONT_HERSHEY_BOLD
    cv2.putText(img, text, (50, 200), font, 2, (0, 0, 255), 3)
    
    # Add more text
    text2 = "Limited Time Only!"
    cv2.putText(img, text2, (80, 300), font, 1, (255, 0, 0), 2)
    
    return img

def test_text_only():
    """Test copywriting analysis with text only"""
    print("\n" + "="*60)
    print("TEST 1: Text-Only Analysis")
    print("="*60)
    
    try:
        from brandguard.core.copywriting_analyzer import CopywritingAnalyzer
        from brandguard.core.model_imports import import_all_models, imported_models
        
        print("📦 Importing models...")
        import_all_models()
        
        print("🔧 Initializing CopywritingAnalyzer...")
        settings = {}
        analyzer = CopywritingAnalyzer(settings, imported_models)
        
        print("📝 Testing text analysis...")
        test_text = "Get 50% off now! Limited time offer."
        
        # Create a dummy image for the interface (required by analyze_copywriting)
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        print(f"   Text: '{test_text}'")
        print(f"⏳ Analyzing... (this may take 30-60 seconds)")
        
        result = analyzer.analyze_copywriting(dummy_image, text_content=test_text)
        
        print("\n✅ Analysis Complete!")
        print(f"   Tone: {result.get('tone_analysis', {})}")
        print(f"   Score: {result.get('copywriting_score', 0):.2f}")
        print(f"   Recommendations: {result.get('recommendations', [])[:2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_analysis():
    """Test copywriting analysis with image"""
    print("\n" + "="*60)
    print("TEST 2: Image Analysis")
    print("="*60)
    
    try:
        from brandguard.core.copywriting_analyzer import CopywritingAnalyzer
        from brandguard.core.model_imports import import_all_models, imported_models
        
        print("📦 Importing models (if not already imported)...")
        import_all_models()
        
        print("🔧 Initializing CopywritingAnalyzer...")
        settings = {}
        analyzer = CopywritingAnalyzer(settings, imported_models)
        
        print("🖼️  Creating test image...")
        test_image = create_simple_test_image()
        
        # Save temporarily to show what we're testing
        temp_path = "/tmp/test_copywriting_image.png"
        cv2.imwrite(temp_path, test_image)
        print(f"   Saved test image to: {temp_path}")
        print(f"   Image contains: 'SALE 50% OFF' and 'Limited Time Only!'")
        
        print(f"⏳ Analyzing image... (this may take 30-90 seconds)")
        
        result = analyzer.analyze_copywriting(test_image)
        
        print("\n✅ Analysis Complete!")
        print(f"   Extracted Text: '{result.get('extracted_text', 'N/A')}'")
        print(f"   Tone: {result.get('tone_analysis', {})}")
        print(f"   Score: {result.get('copywriting_score', 0):.2f}")
        print(f"   Recommendations: {result.get('recommendations', [])[:2]}")
        
        if 'backend_used' in result:
            print(f"   🎯 Backend Used: {result['backend_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with a real image if available"""
    print("\n" + "="*60)
    print("TEST 3: Real Image Analysis (Optional)")
    print("="*60)
    
    # Look for test images
    test_dirs = [
        "uploads",
        "../CopywritingToneChecker/uploads",
        "../CopywritingToneChecker/demo"
    ]
    
    real_image_path = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            image_files = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
            if image_files:
                real_image_path = str(image_files[0])
                break
    
    if not real_image_path:
        print("⚠️  No real test images found, skipping this test")
        return None
    
    try:
        from brandguard.core.copywriting_analyzer import CopywritingAnalyzer
        from brandguard.core.model_imports import import_all_models, imported_models
        
        print(f"🖼️  Found test image: {real_image_path}")
        
        print("📦 Importing models (if not already imported)...")
        import_all_models()
        
        print("🔧 Initializing CopywritingAnalyzer...")
        settings = {}
        analyzer = CopywritingAnalyzer(settings, imported_models)
        
        print("📸 Loading image...")
        image = cv2.imread(real_image_path)
        if image is None:
            print(f"❌ Could not load image: {real_image_path}")
            return False
        
        print(f"⏳ Analyzing real image... (this may take 30-90 seconds)")
        
        result = analyzer.analyze_copywriting(image)
        
        print("\n✅ Analysis Complete!")
        print(f"   Extracted Text: '{result.get('extracted_text', 'N/A')}'")
        print(f"   Score: {result.get('copywriting_score', 0):.2f}")
        print(f"   Recommendations: {result.get('recommendations', [])[:2]}")
        
        if 'backend_used' in result:
            print(f"   🎯 Backend Used: {result['backend_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Simple Consolidated Pipeline Test")
    print("="*60)
    print("This will test the pipeline with:")
    print("  1. Text-only analysis")
    print("  2. Generated test image")
    print("  3. Real image (if available)")
    print("="*60)
    
    results = {
        'text_only': False,
        'test_image': False,
        'real_image': None
    }
    
    # Test 1: Text only
    results['text_only'] = test_text_only()
    
    # Test 2: Generated test image
    results['test_image'] = test_image_analysis()
    
    # Test 3: Real image (optional)
    results['real_image'] = test_with_real_image()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Text-Only Analysis: {'✅ PASSED' if results['text_only'] else '❌ FAILED'}")
    print(f"Test Image Analysis: {'✅ PASSED' if results['test_image'] else '❌ FAILED'}")
    if results['real_image'] is not None:
        print(f"Real Image Analysis: {'✅ PASSED' if results['real_image'] else '❌ FAILED'}")
    else:
        print(f"Real Image Analysis: ⚠️  SKIPPED")
    
    print("\n💡 Tips:")
    print("   • Look for '🚀 Using backend:' in logs to see which inference backend was used")
    print("   • Analysis takes 30-90 seconds per image with Transformers backend")
    print("   • vLLM backend would be faster if server is running")
    print("="*60)

if __name__ == "__main__":
    main()





