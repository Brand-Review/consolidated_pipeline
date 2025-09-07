#!/usr/bin/env python3
"""
Simple test to isolate copywriting analysis issue
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, 'src')

from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
from brandguard.config.settings import Settings

def test_copywriting_only():
    """Test only copywriting analysis"""
    print("✍️ Testing Copywriting Analysis Only...")
    
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
            'formality_score': 60,
            'confidence_level': 'balanced',
            'warmth_score': 50,
            'energy_score': 50
        }
        
        # Test with simple text
        test_text = "Hello world."
        print(f"📝 Testing with text: '{test_text}'")
        
        result = pipeline._perform_copywriting_analysis(test_text, copywriting_options, 'text')
        
        if 'error' in result:
            print(f"❌ Copywriting analysis failed: {result['error']}")
            return False
        
        print(f"✅ Copywriting analysis completed! Score: {(result['compliance_score'] * 100):.1f}%")
        print(f"   Text length: {len(result['extracted_text'])} characters")
        print(f"   Model used: {result['analysis_settings']['model_used']}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_copywriting_only()
    if success:
        print("\n🎉 Copywriting test completed successfully!")
    else:
        print("\n❌ Copywriting test failed!")
