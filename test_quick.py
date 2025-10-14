#!/usr/bin/env python3
"""
Quick test - just check if inference backend is working
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🧪 Quick Pipeline Test")
print("="*60)

# Test 1: Check if InferenceManager is available
print("\n1️⃣ Checking InferenceManager availability...")
try:
    from brandguard.core.model_imports import import_all_models, imported_models
    import_all_models()
    
    if 'InferenceManager' in imported_models and imported_models['InferenceManager']:
        print("   ✅ InferenceManager: Available")
        
        # Try to initialize it
        if 'ConfigManager' in imported_models and imported_models['ConfigManager']:
            config_manager = imported_models['ConfigManager']()
            config = config_manager.get_config()
            manager = imported_models['InferenceManager'](config)
            
            status = manager.get_status()
            available = status.get('available_backends', [])
            
            if available:
                print(f"   ✅ Available backends: {', '.join(available)}")
                print(f"   🎯 System ready for inference!")
            else:
                print(f"   ⚠️  No backends available")
        else:
            print("   ⚠️  ConfigManager not available")
    else:
        print("   ❌ InferenceManager: Not available")
        print("   💡 Will fall back to legacy methods")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Quick text analysis
print("\n2️⃣ Testing quick text analysis...")
try:
    from brandguard.core.copywriting_analyzer import CopywritingAnalyzer
    
    print("   🔧 Initializing analyzer...")
    settings = {}
    analyzer = CopywritingAnalyzer(settings, imported_models)
    
    # Check what analyzer was initialized
    if analyzer.vllm_analyzer:
        if hasattr(analyzer.vllm_analyzer, 'get_status'):
            # It's InferenceManager
            print("   ✅ Using InferenceManager (multi-backend)")
        else:
            # It's old VLLMToneAnalyzer
            print("   ✅ Using VLLMToneAnalyzer (single backend)")
    elif analyzer.tone_analyzer:
        print("   ✅ Using ToneAnalyzer (fallback)")
    else:
        print("   ⚠️  Using basic fallback")
    
    print("\n✅ Pipeline is initialized and ready!")
    print("\n💡 The system is working. For full test, run:")
    print("   python test_simple_pipeline.py")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)





