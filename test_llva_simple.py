#!/usr/bin/env python3
"""
Simple test for LLVa integration
"""

import requests
import base64
import time
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a simple test image"""
    # Create a 100x100 test image with some colors
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[20:80, 20:80] = [255, 0, 0]  # Red square
    img_array[40:60, 40:60] = [0, 255, 0]  # Green square inside
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_b64

def test_llva_direct():
    """Test LLVa API directly"""
    print("🧪 Testing LLVa API directly...")
    
    # Create test image
    img_b64 = create_test_image()
    print(f"✅ Created test image, base64 length: {len(img_b64)}")
    
    # Test prompt
    prompt = "Describe what you see in this image in one sentence."
    
    try:
        print("📤 Sending request to Ollama...")
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llava:latest',
                'prompt': prompt,
                'images': [img_b64],
                'stream': False
            },
            timeout=(10, 60)  # 10s connect, 60s read
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️  Request completed in {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            analysis = result.get('response', 'No response')
            print(f"✅ LLVa Success!")
            print(f"📝 Response: {analysis}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout as e:
        print(f"⏰ Timeout: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_llva_status():
    """Check LLVa model status"""
    print("\n🔍 Checking LLVa model status...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            llava_models = [m for m in models if 'llava' in m.get('name', '').lower()]
            
            if llava_models:
                print("✅ LLVa models available:")
                for model in llava_models:
                    print(f"   - {model['name']} ({model.get('details', {}).get('parameter_size', 'Unknown')})")
                return True
            else:
                print("❌ No LLVa models found")
                return False
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 LLVa Integration Test")
    print("=" * 40)
    
    # Check status first
    status_ok = test_llva_status()
    
    if status_ok:
        print("\n🚀 Running LLVa test...")
        success = test_llva_direct()
        
        if success:
            print("\n🎉 LLVa integration test PASSED!")
        else:
            print("\n💥 LLVa integration test FAILED!")
    else:
        print("\n⚠️  Cannot test LLVa - service not available")
