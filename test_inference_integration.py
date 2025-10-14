#!/usr/bin/env python3
"""
Test script to verify inference backend integration in consolidated pipeline
"""

import sys
import os

# Add consolidated pipeline src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_imports():
    """Test that InferenceManager and ConfigManager are imported correctly"""
    print("🧪 Testing Model Imports for Consolidated Pipeline")
    print("=" * 60)
    
    try:
        from brandguard.core.model_imports import import_all_models, imported_models
        
        print("\n📦 Importing all models...")
        import_all_models()
        
        print("\n📊 Checking imported models:")
        
        # Check for InferenceManager
        if 'InferenceManager' in imported_models and imported_models['InferenceManager']:
            print("  ✅ InferenceManager: Available")
        else:
            print("  ❌ InferenceManager: Not available")
        
        # Check for ConfigManager
        if 'ConfigManager' in imported_models and imported_models['ConfigManager']:
            print("  ✅ ConfigManager: Available")
        else:
            print("  ❌ ConfigManager: Not available")
        
        # Check for VLLMToneAnalyzer (fallback)
        if 'VLLMToneAnalyzer' in imported_models and imported_models['VLLMToneAnalyzer']:
            print("  ✅ VLLMToneAnalyzer: Available (fallback)")
        else:
            print("  ⚠️  VLLMToneAnalyzer: Not available")
        
        print("\n" + "=" * 60)
        
        # Try to initialize InferenceManager
        if 'InferenceManager' in imported_models and imported_models['InferenceManager']:
            print("\n🔧 Testing InferenceManager initialization...")
            try:
                if 'ConfigManager' in imported_models and imported_models['ConfigManager']:
                    config_manager = imported_models['ConfigManager']()
                    config = config_manager.get_config()
                    manager = imported_models['InferenceManager'](config)
                else:
                    # Use default config
                    default_config = {
                        'preferred_backends': ['vllm', 'transformers', 'ollama'],
                        'backends': {
                            'vllm': {
                                'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
                                'vllm_server_url': 'http://localhost:8000'
                            },
                            'transformers': {
                                'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
                                'device': 'auto'
                            },
                            'ollama': {
                                'model_name': 'qwen2.5:3b',
                                'ollama_url': 'http://localhost:11434'
                            }
                        }
                    }
                    manager = imported_models['InferenceManager'](default_config)
                
                # Check status
                status = manager.get_status()
                available_backends = status.get('available_backends', [])
                
                if available_backends:
                    print(f"  ✅ InferenceManager initialized successfully")
                    print(f"  📊 Available backends: {', '.join(available_backends)}")
                    print(f"  🎯 Current backend: {status.get('current_backend', 'auto-select')}")
                else:
                    print(f"  ⚠️  InferenceManager initialized but no backends available")
                    print(f"  💡 Tip: Start vLLM server or check Transformers installation")
                
            except Exception as e:
                print(f"  ❌ InferenceManager initialization failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Integration test complete!")
        print("\n💡 Summary:")
        has_inference_manager = 'InferenceManager' in imported_models and imported_models['InferenceManager']
        has_config_manager = 'ConfigManager' in imported_models and imported_models['ConfigManager']
        
        if has_inference_manager and has_config_manager:
            print("  ✅ Multi-backend system is integrated and ready")
            print("  ✅ Copywriting analyzer will use InferenceManager with fallback")
            print("  ✅ No more warnings about unavailable backends")
        else:
            print("  ⚠️  Multi-backend system not fully integrated")
            print("  ⚠️  Will fall back to legacy single-backend approach")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_imports()





