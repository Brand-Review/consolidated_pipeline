#!/usr/bin/env python3
"""
Test copywriting analysis with LLVa and Ollama integration
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
    """Create a test image with text for copywriting analysis"""
    # Create a 400x300 image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add colored background
    draw.rectangle([0, 0, 400, 300], fill='#f0f0f0')
    
    # Add text (this will be used for copywriting analysis)
    try:
        # Try to use a common font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Add professional business text
    draw.text((50, 50), "Professional Business", fill='#333333', font=font)
    draw.text((50, 100), "Communication", fill='#333333', font=font)
    draw.text((50, 150), "Brand Guidelines", fill='#333333', font=font)
    draw.text((50, 200), "Compliance Check", fill='#333333', font=font)
    
    # Convert to numpy array
    img_array = np.array(img)
    return img_array

def test_copywriting_analysis():
    """Test copywriting analysis with LLVa and Ollama"""
    print("✍️ Testing Copywriting Analysis with LLVa + Ollama...")
    
    try:
        # Create test settings
        settings = Settings()
        
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = PipelineOrchestrator(settings)
        print("✅ Pipeline initialized successfully!")
        
        # Test copywriting analysis
        copywriting_options = {
            'enabled': True,
            'formality_score': 80,  # High formality
            'confidence_level': 'high',
            'warmth_score': 30,     # Low warmth (professional)
            'energy_score': 40      # Low energy (calm)
        }
        
        # Test with image
        test_image = create_test_image()
        print(f"📝 Testing with professional business image")
        
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
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔬 Testing Copywriting Analysis with LLVa + Ollama")
    print("=" * 60)
    
    success = test_copywriting_analysis()
    
    if success:
        print("\n🎉 Copywriting analysis test completed successfully!")
        print("💡 LLVa with Ollama integration is working!")
    else:
        print("\n❌ Copywriting analysis test failed!")
        print("💡 Check if Ollama is running and llava model is available")

if __name__ == "__main__":
    main()
