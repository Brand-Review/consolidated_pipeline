#!/usr/bin/env python3
"""
Test script for LLVa with Ollama integration in consolidated pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from brandguard.config.settings import Settings
from brandguard.core.pipeline_orchestrator import PipelineOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llva_integration():
    """Test LLVa with Ollama integration"""
    try:
        logger.info("🚀 Starting LLVa integration test...")
        
        # Initialize settings and pipeline
        settings = Settings()
        pipeline = PipelineOrchestrator(settings)
        
        # Create a test image (simple colored rectangle with text)
        test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        test_image[50:150, 50:250] = [255, 0, 0]  # Red rectangle (logo area)
        test_image[200:300, 300:550] = [0, 255, 0]  # Green rectangle (text area)
        
        logger.info("📊 Created test image with simulated logo and text areas")
        
        # Test 1: YOLOv8 only
        logger.info("\n🔍 Test 1: YOLOv8 Only Analysis")
        yolo_options = {
            'logo_analysis': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'enable_placement_validation': True,
                'enable_brand_compliance': True,
                'generate_annotations': True,
                'max_detections': 10,
                'allowed_zones': ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                'min_logo_size': 0.01,
                'max_logo_size': 0.25,
                'min_edge_distance': 0.05,
                'enable_llva_ollama': False
            }
        }
        
        yolo_results = pipeline.analyze_content(
            input_source=test_image,
            source_type='image',
            analysis_options=yolo_options
        )
        
        print_analysis_results("YOLOv8 Only", yolo_results)
        
        # Test 2: YOLOv8 + LLVa Combined
        logger.info("\n🤖 Test 2: YOLOv8 + LLVa Combined Analysis")
        combined_options = {
            'logo_analysis': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'enable_placement_validation': True,
                'enable_brand_compliance': True,
                'generate_annotations': True,
                'max_detections': 10,
                'allowed_zones': ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                'min_logo_size': 0.01,
                'max_logo_size': 0.25,
                'min_edge_distance': 0.05,
                'enable_llva_ollama': True,
                'llva_analysis_focus': 'comprehensive'
            }
        }
        
        combined_results = pipeline.analyze_content(
            input_source=test_image,
            source_type='image',
            analysis_options=combined_options
        )
        
        print_analysis_results("YOLOv8 + LLVa", combined_results)
        
        # Test 3: LLVa Focus Options
        logger.info("\n🎯 Test 3: LLVa Analysis Focus Options")
        
        for focus in ['logo_only', 'context_only', 'comprehensive']:
            logger.info(f"Testing focus: {focus}")
            focus_options = combined_options.copy()
            focus_options['logo_analysis']['llva_analysis_focus'] = focus
            
            focus_results = pipeline.analyze_content(
                input_source=test_image,
                source_type='image',
                analysis_options=focus_options
            )
            
            print_focus_results(focus, focus_results)
        
        logger.info("\n✅ LLVa integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLVa integration test failed: {e}")
        return False

def print_analysis_results(test_name: str, results: dict):
    """Print formatted analysis results"""
    print(f"\n📋 {test_name} Results:")
    print("=" * 50)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    if 'model_results' in results and 'logo_analysis' in results['model_results']:
        logo_results = results['model_results']['logo_analysis']
        
        # Model information
        if 'model' in logo_results:
            model_info = logo_results['model']
            print(f"🔧 Model: {model_info.get('name', 'Unknown')}")
            print(f"📁 File: {model_info.get('file', 'Unknown')}")
            print(f"⚙️  Backend: {model_info.get('backend', 'Unknown')}")
            print(f"🎯 Analysis Type: {model_info.get('analysis_type', 'Unknown')}")
        
        # Timing information
        if 'timings' in logo_results:
            timings = logo_results['timings']
            print(f"⏱️  Detection Time: {timings.get('detection_ms', 0):.2f}ms")
            print(f"🤖 LLVa Time: {timings.get('llva_ms', 0):.2f}ms")
            print(f"⏰ Total Time: {timings.get('total_ms', 0):.2f}ms")
        
        # Detection results
        detections = logo_results.get('logo_detections', [])
        print(f"🎯 Logos Detected: {len(detections)}")
        
        # Compliance scores
        if 'scores' in logo_results:
            scores = logo_results['scores']
            print(f"📊 Overall Score: {scores.get('overall', 0):.2f}")
            print(f"📏 Strict Score: {scores.get('strict', 0):.2f}")
        
        # LLVa analysis
        if 'llva_analysis' in logo_results and logo_results['llva_analysis']:
            llva = logo_results['llva_analysis']
            print(f"🧠 LLVa Success: {llva.get('success', False)}")
            print(f"🎭 Analysis Focus: {llva.get('analysis_focus', 'N/A')}")
            
            if 'parsed_analysis' in llva:
                parsed = llva['parsed_analysis']
                print(f"🏷️  Logos Identified by LLVa: {len(parsed.get('logos_identified', []))}")
                print(f"🎯 LLVa Confidence: {parsed.get('confidence_score', 0):.2f}")

def print_focus_results(focus: str, results: dict):
    """Print results for specific focus test"""
    print(f"  📍 {focus.upper()} Focus:")
    
    if 'error' in results:
        print(f"    ❌ Error: {results['error']}")
        return
    
    if 'model_results' in results and 'logo_analysis' in results['model_results']:
        logo_results = results['model_results']['logo_analysis']
        
        if 'llva_analysis' in logo_results and logo_results['llva_analysis']:
            llva = logo_results['llva_analysis']
            success = llva.get('success', False)
            focus_used = llva.get('analysis_focus', 'N/A')
            print(f"    ✅ Success: {success} | Focus: {focus_used}")
            
            if 'parsed_analysis' in llva:
                parsed = llva['parsed_analysis']
                confidence = parsed.get('confidence_score', 0)
                print(f"    🎯 Confidence: {confidence:.2f}")
        else:
            print(f"    ❌ No LLVa analysis found")

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama service is running")
            models = response.json().get('models', [])
            llava_models = [m for m in models if 'llava' in m.get('name', '').lower()]
            if llava_models:
                logger.info(f"✅ LLVa models available: {[m['name'] for m in llava_models]}")
                return True
            else:
                logger.warning("⚠️  No LLVa models found in Ollama")
                return False
        else:
            logger.warning("⚠️  Ollama service not responding properly")
            return False
    except Exception as e:
        logger.warning(f"⚠️  Ollama service check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 BrandGuard LLVa Integration Test Suite")
    print("=" * 60)
    
    # Check Ollama service
    ollama_available = check_ollama_service()
    if not ollama_available:
        print("⚠️  Note: LLVa analysis will use fallback mode (Ollama not available)")
    
    # Run tests
    success = test_llva_integration()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
