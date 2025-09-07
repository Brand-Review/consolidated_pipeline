#!/usr/bin/env python3
"""
Test LLVa logo detection capabilities
"""

import requests
import base64
import time
from PIL import Image
import numpy as np
import io

def create_agencyhandy_mockup():
    """Create a mockup similar to the AgencyHandy promotional image"""
    # Create a larger image (400x600) to better represent the promotional graphic
    img_array = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Background gradient (light blue to purple)
    for y in range(400):
        # Create gradient from top to bottom
        blue_ratio = 1 - (y / 400)
        purple_ratio = y / 400
        
        # Light blue at top, purple at bottom
        blue = int(173 + (255 - 173) * blue_ratio)  # Light blue
        green = int(216 + (0 - 216) * blue_ratio)
        red = int(230 + (128 - 230) * blue_ratio)
        
        img_array[y, :] = [blue, green, red]
    
    # Add logo area in top-left (purple shapes)
    # Logo shapes (simplified version of the interlocking curves)
    img_array[50:120, 50:150] = [128, 0, 128]  # Purple logo area
    
    # Add "AgencyHandy" text area (black text on white background)
    img_array[40:80, 160:320] = [255, 255, 255]  # White text background
    img_array[50:70, 170:310] = [0, 0, 0]  # Black text area
    
    # Add UI windows (white rectangles with rounded corners effect)
    # Main dashboard window
    img_array[100:300, 50:250] = [255, 255, 255]  # White window
    
    # Overlapping windows
    img_array[120:280, 80:280] = [240, 240, 240]  # Light gray window
    img_array[140:260, 100:300] = [250, 250, 250]  # Very light gray window
    
    # Right side promotional text area
    img_array[150:350, 400:580] = [128, 0, 128]  # Purple text area
    
    return img_array

def test_llva_logo_detection():
    """Test LLVa's logo detection capabilities"""
    print("🔍 Testing LLVa Logo Detection...")
    
    # Create AgencyHandy mockup
    img_array = create_agencyhandy_mockup()
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print(f"✅ Created AgencyHandy mockup image, base64 length: {len(img_b64)}")
    
    # Test prompts for logo detection
    logo_prompts = [
        "Can you identify any logos or brand marks in this image? If yes, describe their location and appearance.",
        "Are there any company logos visible in this image? Please describe what you see.",
        "Look for any logos, brand names, or company identifiers in this image. What do you find?",
        "This appears to be a promotional image. Can you spot any logos or brand elements?",
        "Focus on the top-left area of this image. Do you see any logos or brand symbols there?"
    ]
    
    for i, prompt in enumerate(logo_prompts, 1):
        print(f"\n🧪 Test {i}: Logo Detection")
        print(f"📝 Prompt: {prompt}")
        
        try:
            print("📤 Sending request to LLVa...")
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llava:latest',
                    'prompt': prompt,
                    'images': [img_b64],
                    'stream': False
                },
                timeout=(10, 60)
            )
            
            elapsed = time.time() - start_time
            print(f"⏱️  Response time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', 'No response')
                print(f"✅ LLVa Response:")
                print(f"📄 {analysis}")
                
                # Check if logo was detected
                logo_keywords = ['logo', 'brand', 'symbol', 'mark', 'agencyhandy', 'agency handy']
                detected = any(keyword in analysis.lower() for keyword in logo_keywords)
                
                if detected:
                    print("🎯 LOGO DETECTED! ✅")
                else:
                    print("❌ No logo mentioned in response")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

def test_llva_brand_analysis():
    """Test LLVa's ability to analyze brand elements"""
    print("\n🎨 Testing LLVa Brand Analysis...")
    
    # Create the same image
    img_array = create_agencyhandy_mockup()
    img = Image.fromarray(img_array)
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    brand_prompts = [
        "Analyze this image for brand identity elements. What company or product is being promoted?",
        "What is the main brand name visible in this image? Where is it located?",
        "Describe the visual branding elements you can see in this promotional image.",
        "This appears to be marketing material. What brand or company is featured?",
        "Look at the top-left corner of this image. What brand or company name do you see there?"
    ]
    
    for i, prompt in enumerate(brand_prompts, 1):
        print(f"\n🏷️  Brand Analysis Test {i}")
        print(f"📝 Prompt: {prompt}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llava:latest',
                    'prompt': prompt,
                    'images': [img_b64],
                    'stream': False
                },
                timeout=(10, 60)
            )
            
            elapsed = time.time() - start_time
            print(f"⏱️  Response time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', 'No response')
                print(f"✅ LLVa Response:")
                print(f"📄 {analysis}")
                
                # Check for brand detection
                brand_detected = 'agencyhandy' in analysis.lower() or 'agency' in analysis.lower()
                
                if brand_detected:
                    print("🏷️  BRAND DETECTED! ✅")
                else:
                    print("❌ Brand not clearly identified")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    print("🔬 LLVa Logo Detection Test for AgencyHandy Image")
    print("=" * 70)
    
    # Test logo detection
    test_llva_logo_detection()
    
    # Test brand analysis
    test_llva_brand_analysis()
    
    print("\n🎯 Logo Detection Test Complete!")
    print("This test simulates asking LLVa to find logos in the AgencyHandy promotional image.")
