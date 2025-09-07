#!/usr/bin/env python3
"""
Test real models integration
"""

import sys
import os
import numpy as np
from PIL import Image

# Add paths to individual model directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

color_path = os.path.join(parent_dir, 'ColorPaletteChecker', 'src')
typography_path = os.path.join(parent_dir, 'FontTypographyChecker', 'src')
copywriting_path = os.path.join(parent_dir, 'CopywritingToneChecker', 'src')

print(f"Color path: {color_path}")
print(f"Typography path: {typography_path}")
print(f"Copywriting path: {copywriting_path}")

# Test ColorPaletteExtractor
print("\n🔍 Testing ColorPaletteExtractor...")
try:
    sys.path.insert(0, color_path)
    from brandguard.core.color_palette import ColorPaletteExtractor
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Initialize extractor
    extractor = ColorPaletteExtractor(n_colors=8)
    
    # Extract colors
    colors = extractor.extract_colors(test_image)
    
    print(f"✅ ColorPaletteExtractor working! Extracted {len(colors)} colors")
    for i, color in enumerate(colors):
        print(f"  Color {i+1}: {color['hex']} (RGB: {color['rgb']})")
        
except Exception as e:
    print(f"❌ ColorPaletteExtractor failed: {e}")

# Test FontIdentifier
print("\n🔍 Testing FontIdentifier...")
try:
    sys.path.insert(0, typography_path)
    from brandguard.core.font_identifier import FontIdentifier
    
    # Initialize identifier
    identifier = FontIdentifier()
    
    # Test with sample text
    test_text = "Sample text for font identification"
    font_info = identifier.identify_font(test_text)
    
    print(f"✅ FontIdentifier working! Font info: {font_info}")
        
except Exception as e:
    print(f"❌ FontIdentifier failed: {e}")

# Test ToneAnalyzer
print("\n🔍 Testing ToneAnalyzer...")
try:
    sys.path.insert(0, copywriting_path)
    from brandguard.core.tone_analyzer import ToneAnalyzer
    
    # Initialize analyzer
    analyzer = ToneAnalyzer()
    
    # Test with sample text
    test_text = "This is a professional business communication."
    tone_result = analyzer.analyze_tone(test_text)
    
    print(f"✅ ToneAnalyzer working! Tone result: {tone_result}")
        
except Exception as e:
    print(f"❌ ToneAnalyzer failed: {e}")

print("\n🎯 Real Models Test Complete!")
