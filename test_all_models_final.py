#!/usr/bin/env python3
"""
Test ALL analysis models including working copywriting analysis with LLVa + Ollama
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the src directory to the path
sys.path.insert(0, 'src')

from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
from brandguard.config.settings import Settings

def create_test_image():
    """Create a test image with text and colors for comprehensive testing"""
    # Create a 400x300 image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add colored regions
    draw.rectangle([0, 0, 200, 150], fill='red')
    draw.rectangle([200, 0, 400, 150], fill='blue')
    draw.rectangle([0, 150, 200, 300], fill='green')
    draw.rectangle([200, 150, 400, 300], fill='yellow')
    
    # Add text (this will be used for typography and copywriting analysis)
    try:
        # Try to use a common font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Professional business text for copywriting analysis
    draw.text((50, 50), "Professional Business", fill='white', font=font)
    draw.text((250, 50), "Brand Guidelines", fill='white', font=font)
    draw.text((50, 200), "Compliance Check", fill='black', font=font)
    draw.text((250, 200), "Quality Assurance", fill='black', font=font)
    
    # Convert to numpy array
    img_array = np.array(img)
    return img_array

def test_color_analysis(pipeline):
    """Test color analysis functionality"""
    print("\n🎨 Testing Color Analysis...")
    
    try:
        color_options = {
            'enabled': True,
            'n_colors': 8,
            'color_tolerance': 0.2,
            'enable_contrast_check': True,
            'brand_palette': '#FF0000,#00FF00,#0000FF,#FFFF00'  # Red, Green, Blue, Yellow
        }
        
        test_image = create_test_image()
        result = pipeline._perform_color_analysis(test_image, color_options)
        
        if 'error' in result:
            print(f"❌ Color analysis failed: {result['error']}")
            return False
        
        print(f"✅ Color analysis completed! Score: {(result['compliance_score'] * 100):.1f}%")
        print(f"   Colors detected: {len(result['dominant_colors'])}")
        print(f"   Model used: {result['analysis_settings']['model_used']}")
        return True
        
    except Exception as e:
        print(f"❌ Color analysis test failed: {e}")
        return False

def test_typography_analysis(pipeline):
    """Test typography analysis functionality"""
    print("\n📝 Testing Typography Analysis...")
    
    try:
        typography_options = {
            'enabled': True,
            'confidence_threshold': 0.7,
            'merge_regions': True,
            'distance_threshold': 20,
            'expected_fonts': 'Arial,Helvetica'
        }
        
        test_image = create_test_image()
        result = pipeline._perform_typography_analysis(test_image, typography_options)
        
        if 'error' in result:
            print(f"❌ Typography analysis failed: {result['error']}")
            return False
        
        print(f"✅ Typography analysis completed! Score: {(result['compliance_score'] * 100):.1f}%")
        print(f"   Text regions: {len(result['text_regions'])}")
        print(f"   Fonts identified: {len(result['font_analysis'])}")
        print(f"   Model used: {result['analysis_settings']['model_used']}")
        return True
        
    except Exception as e:
        print(f"❌ Typography analysis test failed: {e}")
        return False

def test_copywriting_analysis(pipeline):
    """Test copywriting analysis functionality with LLVa + Ollama"""
    print("\n✍️ Testing Copywriting Analysis (LLVa + Ollama)...")
    
    try:
        copywriting_options = {
            'enabled': True,
            'formality_score': 80,  # High formality
            'confidence_level': 'high',
            'warmth_score': 30,     # Low warmth (professional)
            'energy_score': 40      # Low energy (calm)
        }
        
        test_image = create_test_image()
        result = pipeline._perform_copywriting_analysis(test_image, copywriting_options, 'image')
        
        if 'error' in result:
            print(f"❌ Copywriting analysis failed: {result['error']}")
            return False
        
        print(f"✅ Copywriting analysis completed! Score: {(result['compliance_score'] * 100):.1f}%")
        print(f"   Text length: {len(result['extracted_text'])} characters")
        print(f"   Model used: {result['analysis_settings']['model_used']}")
        
        # Show detailed results
        if 'tone_analysis' in result:
            tone = result['tone_analysis']
            print(f"   Tone Category: {tone.get('tone_category', 'N/A')}")
            print(f"   Formality Score: {tone.get('formality_score', 'N/A')}")
            print(f"   Confidence: {tone.get('confidence', 'N/A')}")
        
        if 'compliance_check' in result:
            compliance = result['compliance_check']
            print(f"   Issues Found: {len(compliance.get('issues', []))}")
            print(f"   Recommendations: {len(compliance.get('recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Copywriting analysis test failed: {e}")
        return False

def test_logo_analysis(pipeline):
    """Test logo analysis functionality"""
    print("\n🏢 Testing Logo Analysis...")
    
    try:
        logo_options = {
            'enabled': True,
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'min_logo_size': 20,
            'max_logo_size': 500,
            'enable_llva_ollama': False
        }
        
        test_image = create_test_image()
        result = pipeline._perform_logo_detection_analysis(test_image, logo_options)
        
        if 'error' in result:
            print(f"❌ Logo analysis failed: {result['error']}")
            return False
        
        print(f"✅ Logo analysis completed! Score: {(result['compliance_score'] * 100):.1f}%")
        print(f"   Logos detected: {len(result['logo_detections'])}")
        print(f"   Model used: {result['analysis_settings']['model_used']}")
        return True
        
    except Exception as e:
        print(f"❌ Logo analysis test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔬 Testing ALL Analysis Models with Real Implementations")
    print("=" * 60)
    print("🎯 Including: Color, Typography, Copywriting (LLVa+Ollama), Logo")
    print("=" * 60)
    
    try:
        # Create test settings
        settings = Settings()
        
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = PipelineOrchestrator(settings)
        print("✅ Pipeline initialized successfully!")
        
        # Test each analysis model
        results = {}
        
        results['color'] = test_color_analysis(pipeline)
        results['typography'] = test_typography_analysis(pipeline)
        results['copywriting'] = test_copywriting_analysis(pipeline)
        results['logo'] = test_logo_analysis(pipeline)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for model, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{model.upper():12} : {status}")
        
        passed = sum(results.values())
        total = len(results)
        print(f"\n🎯 Overall: {passed}/{total} models working ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL MODELS ARE NOW WORKING PERFECTLY!")
            print("🚀 Your brand compliance pipeline is fully operational!")
        else:
            print("⚠️  Some models need attention")
        
        print("\n💡 Key Features:")
        print("   🎨 Color Analysis: Real ColorPaletteExtractor with palette validation")
        print("   📝 Typography: Real FontIdentifier with ML-based font detection")
        print("   ✍️ Copywriting: LLVa with Ollama for AI-powered tone analysis")
        print("   🏢 Logo Detection: YOLOv8 with advanced placement validation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
