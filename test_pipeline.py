#!/usr/bin/env python3
"""
Test script for BrandGuard Consolidated Pipeline
Tests the core components without starting the full server
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from brandguard.config.settings import Settings
        print("✅ Settings imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Settings: {e}")
        return False
    
    try:
        from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
        print("✅ PipelineOrchestrator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import PipelineOrchestrator: {e}")
        return False
    
    return True

def test_settings():
    """Test settings configuration"""
    print("\n⚙️ Testing settings...")
    
    try:
        from brandguard.config.settings import Settings
        settings = Settings()
        print("✅ Settings loaded successfully")
        print(f"   - Color palette: {settings.color_palette.name}")
        print(f"   - Typography rules: {len(settings.typography_rules.approved_fonts)} approved fonts")
        print(f"   - Brand voice: {settings.brand_voice.formality_score} formality score")
        print(f"   - Logo detection: {settings.logo_detection.confidence_threshold} confidence threshold")
        return True
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline orchestrator initialization"""
    print("\n🚀 Testing pipeline initialization...")
    
    try:
        from brandguard.config.settings import Settings
        from brandguard.core.pipeline_orchestrator import PipelineOrchestrator
        
        settings = Settings()
        pipeline = PipelineOrchestrator(settings)
        print("✅ Pipeline orchestrator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_config_files():
    """Test if configuration files exist"""
    print("\n📁 Testing configuration files...")
    
    config_dir = Path('configs')
    required_configs = [
        'color_palette.yaml',
        'typography_rules.yaml', 
        'brand_voice.yaml',
        'logo_detection.yaml'
    ]
    
    all_exist = True
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            all_exist = False
    
    return all_exist

def test_directories():
    """Test if required directories exist"""
    print("\n📂 Testing directories...")
    
    required_dirs = ['configs', 'uploads', 'results', 'models']
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🎨 BrandGuard Pipeline - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Settings Tests", test_settings),
        ("Pipeline Tests", test_pipeline_initialization),
        ("Config Files", test_config_files),
        ("Directories", test_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to run.")
        print("\nTo start the server:")
        print("  python app.py")
        print("  # or")
        print("  python run.py --port 5001")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
