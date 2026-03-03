"""
Brand Guidelines Extraction Service
Extracts colors, fonts, and settings from PDF/XML/image files
"""
import os
import re
import json
import logging
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)

class BrandExtractor:
    """Extract brand guidelines from PDF/XML/image files"""
    
    def __init__(self, qwen_api_url: str = "http://localhost:8000/v1/chat/completions"):
        self.qwen_api_url = qwen_api_url
        self.timeout = 120
        self._qwen_available = None  # Cache availability check
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            text_content = []
            images = []
            
            # Get number of pages before processing (need this before closing)
            num_pages = len(doc)
            
            # Extract from first 10 pages (usually enough for brand guidelines)
            max_pages = min(10, num_pages)
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_content.append(text)
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        images.append(img_pil)
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            
            return {
                'text': '\n\n'.join(text_content),
                'images': images,
                'num_pages': num_pages
            }
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def extract_colors_from_images(self, images: List[Image.Image]) -> List[str]:
        """Extract dominant colors from images using K-means"""
        try:
            from sklearn.cluster import KMeans
            import warnings
            from sklearn.exceptions import ConvergenceWarning
        except ImportError:
            logger.warning("scikit-learn not available, skipping color extraction from images")
            return []
        
        all_colors = []
        
        for img in images:
            try:
                # Convert PIL to numpy array
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    # Reshape for K-means
                    pixels = img_array.reshape(-1, 3)
                    
                    if len(pixels) == 0:
                        continue
                    
                    # Sample pixels for speed (max 10000)
                    sample_size = min(10000, len(pixels))
                    sampled_pixels = pixels[:sample_size]
                    
                    # Find unique colors to determine appropriate number of clusters
                    unique_pixels = np.unique(sampled_pixels, axis=0)
                    num_unique = len(unique_pixels)
                    
                    # Extract 3-5 dominant colors, but not more than unique colors available
                    n_colors = min(5, num_unique, len(sampled_pixels))
                    
                    if n_colors > 0:
                        # Suppress ConvergenceWarning for cases where we have fewer distinct colors than clusters
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                            kmeans.fit(sampled_pixels)
                        
                        # Get cluster centers (colors)
                        colors = kmeans.cluster_centers_.astype(int)
                        
                        # Convert to hex
                        for color in colors:
                            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                            all_colors.append(hex_color.upper())
            except Exception as e:
                logger.warning(f"Color extraction from image failed: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_colors = []
        for color in all_colors:
            if color not in seen:
                seen.add(color)
                unique_colors.append(color)
        
        return unique_colors[:10]  # Return top 10 unique colors
    
    def _extract_colors_from_pdf_pages(self, pdf_path: str, max_pages: int = 5) -> List[str]:
        """Extract colors by rendering PDF pages as images and analyzing them"""
        try:
            doc = fitz.open(pdf_path)
            all_colors = []
            
            # Process first few pages (usually color palette is on early pages)
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                
                # Render page as image (higher resolution for better color detection)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Extract colors from this page
                page_colors = self.extract_colors_from_images([img])
                all_colors.extend(page_colors)
            
            doc.close()
            
            # Remove duplicates
            seen = set()
            unique_colors = []
            for color in all_colors:
                if color not in seen:
                    seen.add(color)
                    unique_colors.append(color)
            
            return unique_colors
            
        except Exception as e:
            logger.warning(f"Failed to extract colors from PDF pages: {e}")
            return []
    
    def _check_qwen_available(self) -> bool:
        """Check if Qwen API server is available"""
        if self._qwen_available is not None:
            return self._qwen_available
        
        try:
            # Try to connect to the health endpoint or models endpoint
            base_url = self.qwen_api_url.replace('/v1/chat/completions', '')
            health_url = f"{base_url}/v1/models"
            response = requests.get(health_url, timeout=2)
            self._qwen_available = response.status_code == 200
            return self._qwen_available
        except Exception:
            self._qwen_available = False
            return False
    
    def _rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB values to hex code"""
        return f"#{r:02x}{g:02x}{b:02x}".upper()
    
    def _cmyk_to_rgb(self, c: float, m: float, y: float, k: float) -> tuple:
        """Convert CMYK to RGB (approximate conversion)
        CMYK values should be 0-100 (percentages)
        """
        # Normalize CMYK to 0-1 range
        c = c / 100.0
        m = m / 100.0
        y = y / 100.0
        k = k / 100.0
        
        # Convert CMYK to RGB
        r = int(255 * (1 - c) * (1 - k))
        g = int(255 * (1 - m) * (1 - k))
        b = int(255 * (1 - y) * (1 - k))
        
        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return (r, g, b)
    
    def _color_name_to_hex(self, color_name: str) -> Optional[str]:
        """Convert common color names to hex codes"""
        color_map = {
            # Basic colors
            'white': '#FFFFFF', 'black': '#000000', 'red': '#FF0000',
            'green': '#008000', 'blue': '#0000FF', 'yellow': '#FFFF00',
            'cyan': '#00FFFF', 'magenta': '#FF00FF',
            
            # Common brand colors
            'orange red': '#FF4500', 'orange': '#FFA500', 'dark orange': '#FF8C00',
            'light blue': '#ADD8E6', 'dark blue': '#00008B', 'navy': '#000080',
            'light gray': '#D3D3D3', 'gray': '#808080', 'dark gray': '#A9A9A9',
            'light green': '#90EE90', 'dark green': '#006400',
            'pink': '#FFC0CB', 'purple': '#800080', 'brown': '#A52A2A',
            
            # Extended palette
            'crimson': '#DC143C', 'maroon': '#800000', 'olive': '#808000',
            'teal': '#008080', 'silver': '#C0C0C0', 'gold': '#FFD700',
        }
        
        # Try exact match first
        normalized = color_name.lower().strip()
        if normalized in color_map:
            return color_map[normalized]
        
        # Try partial match (e.g., "Orange Red" contains "orange")
        for key, hex_val in color_map.items():
            if key in normalized or normalized in key:
                return hex_val
        
        return None
    
    def _is_neutral_color(self, hex_color: str) -> bool:
        """Check if a color is neutral (black, white, or gray)"""
        # Convert hex to RGB
        try:
            hex_color = hex_color.lstrip('#').upper()
            if len(hex_color) != 6:
                return False
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Check if it's black or white
            if r == g == b:
                # Gray scale - consider neutral
                return True
            # Very dark colors (near black)
            if r < 30 and g < 30 and b < 30:
                return True
            # Very light colors (near white)
            if r > 225 and g > 225 and b > 225:
                return True
            
            # Check if it's a gray (all channels similar, low saturation)
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            if max_val - min_val < 30:  # Low saturation = gray
                return True
            
            return False
        except (ValueError, IndexError):
            return False
    
    def extract_with_qwen(self, text: str, images: Optional[List[Image.Image]] = None) -> Dict[str, Any]:
        """Use Qwen2.5-VL to extract structured brand data - REFACTORED with sensor-only prompt"""
        # Check if Qwen server is available before attempting connection
        if not self._check_qwen_available():
            logger.debug("Qwen API server not available, skipping Qwen extraction and using defaults")
            return self._get_default_structure()
        
        try:
            # Try to import sensor-only prompt
            try:
                from brandguard.core.prompts import get_brand_evidence_prompt
                prompt = get_brand_evidence_prompt(text[:4000])
            except ImportError:
                # Fallback: use inline sensor-only prompt
                prompt = f"""Extract explicit brand evidence from this document.

Rules:
- Do NOT decide primary/secondary colors.
- Do NOT infer importance.
- Extract only what is explicitly mentioned or shown.
- Include evidence text.

Return STRICT JSON ONLY:
{{
  "version": "analysis_v1",
  "detected": {{
    "colors": [
      {{
        "hex": "#HEX",
        "name": "string",
        "source": "logo|text|example",
        "evidence": "exact reference"
      }}
    ],
    "fonts": ["Font name"]
  }},
  "observations": {{}},
  "flags": [],
  "raw_metrics": {{}},
  "confidence": 0.0
}}

Content:
{text[:4000]}"""
            
            # Call Qwen API
            payload = {
                "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.qwen_api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group())
                        
                        # Normalize to canonical format if needed
                        try:
                            from brandguard.core.normalization import ProviderNormalizer
                            normalizer = ProviderNormalizer()
                            normalized = normalizer.normalize(extracted_data, provider="qwen")
                            extracted_data = normalized
                        except ImportError:
                            pass  # Continue with raw data if normalizer not available
                        
                        # Convert canonical format to legacy format for backward compatibility
                        detected = extracted_data.get("detected", {})
                        colors = detected.get("colors", [])
                        
                        # Determine primary color using CODE rules (not AI)
                        primary_color = self._determine_primary_color(colors)
                        
                        # Build legacy format
                        legacy_format = {
                            "primaryColor": {
                                "role": "primary",
                                "color_name": primary_color.get("name", ""),
                                "hex": primary_color.get("hex", ""),
                                "confidence": 0.8,
                                "evidence": primary_color.get("evidence", []),
                                "governance_details": f"Selected from {len(colors)} detected colors using deterministic rules"
                            },
                            "secondaryColors": [c.get("hex", "") for c in colors[:3] if c.get("hex") != primary_color.get("hex")],
                            "accentColors": [c.get("hex", "") for c in colors[3:6]],
                            "fonts": detected.get("fonts", []),
                            "formalityScore": 60,
                            "warmthScore": 50,
                            "energyScore": 50,
                            "confidenceLevel": "balanced"
                        }
                        
                        return legacy_format
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}")
                        return self._get_default_structure()
                else:
                    logger.warning("No JSON found in Qwen response")
                    return self._get_default_structure()
            else:
                logger.warning(f"Qwen API call failed: {response.status_code}")
                return self._get_default_structure()
                
        except requests.exceptions.ConnectionError as e:
            # Connection refused or server not available - this is expected if server isn't running
            logger.warning(f"Qwen API server not available at {self.qwen_api_url}. Extraction will continue with image-based color detection only.")
            self._qwen_available = False  # Cache that server is unavailable
            return self._get_default_structure()
        except Exception as e:
            # Other errors (timeout, parsing, etc.) - log as warning since we have fallback
            logger.warning(f"Qwen extraction failed: {e}. Using fallback extraction method.")
            return self._get_default_structure()
    
    def _determine_primary_color(self, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine primary color using CODE rules (not AI)
        
        Rules:
        1. Logo colors have priority
        2. Neutral colors (white, black, gray) are NEVER primary unless explicitly labeled
        3. First non-neutral color from logo source
        4. If no logo colors, first non-neutral color
        5. If all neutral, return empty
        
        Args:
            colors: List of color dictionaries with hex, name, source, evidence
            
        Returns:
            Primary color dictionary
        """
        if not colors:
            return {"hex": "", "name": "", "evidence": []}
        
        # Priority 1: Logo colors (non-neutral)
        logo_colors = [c for c in colors if c.get("source") == "logo" and not self._is_neutral_color(c.get("hex", ""))]
        if logo_colors:
            return logo_colors[0]
        
        # Priority 2: Any non-neutral color
        non_neutral = [c for c in colors if not self._is_neutral_color(c.get("hex", ""))]
        if non_neutral:
            return non_neutral[0]
        
        # Priority 3: First color (even if neutral)
        return colors[0] if colors else {"hex": "", "name": "", "evidence": []}
    
    def _get_default_structure(self) -> Dict[str, Any]:
        """Return default structure if extraction fails"""
        return {
            "primaryColor": {
                "role": "primary",
                "color_name": "",
                "hex": "",
                "confidence": 0.0,
                "evidence": [],
                "governance_details": ""
            },
            "secondaryColors": [],
            "accentColors": [],
            "fonts": [],
            "formalityScore": 60,
            "warmthScore": 50,
            "energyScore": 50,
            "confidenceLevel": "balanced"
        }
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract brand guidelines directly from text (fallback when Qwen unavailable)"""
        result = {
            "primaryColors": [],
            "secondaryColors": [],
            "accentColors": [],
            "fonts": [],
            "formalityScore": 60,
            "warmthScore": 50,
            "energyScore": 50,
            "confidenceLevel": "balanced"
        }
        
        if not text:
            return result
        
        primary_colors = []
        secondary_colors = []
        accent_colors = []
        uncategorized_colors = []
        seen_colors = set()
        
        # 1. Extract colors with context (Primary, Secondary, Accent)
        # Pattern: "Primary Color: #ff4500" or "Primary: #ff4500" or "Primary Color #ff4500"
        context_patterns = [
            (r'Primary\s+Color[:\s]+#([0-9A-Fa-f]{6})\b', primary_colors),
            (r'Primary[:\s]+#([0-9A-Fa-f]{6})\b', primary_colors),
            (r'Secondary\s+Color[:\s]+#([0-9A-Fa-f]{6})\b', secondary_colors),
            (r'Secondary[:\s]+#([0-9A-Fa-f]{6})\b', secondary_colors),
            (r'Accent\s+Color[:\s]+#([0-9A-Fa-f]{6})\b', accent_colors),
            (r'Accent[:\s]+#([0-9A-Fa-f]{6})\b', accent_colors),
        ]
        
        for pattern, color_list in context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                hex_color = f"#{match.upper()}"
                if hex_color not in seen_colors:
                    seen_colors.add(hex_color)
                    color_list.append(hex_color)
        
        # 2. Extract RGB values with context
        rgb_context_patterns = [
            (r'Primary\s+Color[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', primary_colors),
            (r'Primary[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', primary_colors),
            (r'Secondary\s+Color[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', secondary_colors),
            (r'Secondary[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', secondary_colors),
            (r'Accent\s+Color[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', accent_colors),
            (r'Accent[:\s]+(?:RGB[:\s]+)?(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})', accent_colors),
        ]
        
        for pattern, color_list in rgb_context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for r, g, b in matches:
                try:
                    r_val, g_val, b_val = int(r), int(g), int(b)
                    if 0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255:
                        hex_color = self._rgb_to_hex(r_val, g_val, b_val)
                        if hex_color not in seen_colors:
                            seen_colors.add(hex_color)
                            color_list.append(hex_color)
                except ValueError:
                    continue
        
        # 3. Extract hex color codes without context (for uncategorized colors)
        hex_pattern = r'#([0-9A-Fa-f]{6})\b'
        hex_matches = re.findall(hex_pattern, text)
        for match in hex_matches:
            hex_color = f"#{match.upper()}"
            if hex_color not in seen_colors:
                seen_colors.add(hex_color)
                uncategorized_colors.append(hex_color)
        
        # 4. Extract RGB values without context and convert to hex
        # Patterns: "RGB: 255, 69, 0" or "RGB 255 69 0"
        # Note: Colors already captured by context patterns will be skipped via seen_colors
        rgb_patterns = [
            r'RGB[:\s]*(\d+)[,\s]+(\d+)[,\s]+(\d+)',  # "RGB: 255, 69, 0"
            r'(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})(?=\s*(?:RGB|CMYK|Hex|$))',  # "255, 69, 0" before RGB/CMYK/Hex
        ]
        
        for pattern in rgb_patterns:
            rgb_matches = re.findall(pattern, text, re.IGNORECASE)
            for r, g, b in rgb_matches:
                try:
                    r_val, g_val, b_val = int(r), int(g), int(b)
                    if 0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255:
                        hex_color = self._rgb_to_hex(r_val, g_val, b_val)
                        if hex_color not in seen_colors:
                            seen_colors.add(hex_color)
                            uncategorized_colors.append(hex_color)
                except ValueError:
                    continue
        
        # 5. Extract CMYK values and convert to hex
        # Pattern: "CMYK: 0%, 73%, 100%, 0%" or "CMYK 0 73 100 0"
        cmyk_patterns = [
            r'CMYK[:\s]*(\d+(?:\.\d+)?)%?[,\s]+(\d+(?:\.\d+)?)%?[,\s]+(\d+(?:\.\d+)?)%?[,\s]+(\d+(?:\.\d+)?)%?',
        ]
        
        for pattern in cmyk_patterns:
            cmyk_matches = re.findall(pattern, text, re.IGNORECASE)
            for c, m, y, k in cmyk_matches:
                try:
                    c_val = float(c)
                    m_val = float(m)
                    y_val = float(y)
                    k_val = float(k)
                    
                    # Convert CMYK to RGB
                    rgb = self._cmyk_to_rgb(c_val, m_val, y_val, k_val)
                    hex_color = self._rgb_to_hex(*rgb)
                    
                    if hex_color not in seen_colors:
                        seen_colors.add(hex_color)
                        uncategorized_colors.append(hex_color)
                except (ValueError, TypeError):
                    continue
        
        # 6. Extract color names and convert to hex (only if not already found)
        # Look for color names near color specifications
        color_name_contexts = [
            r'(?:Color|Palette)[:\s]+([A-Za-z\s]+?)(?:\n|RGB|CMYK|Hex|#|$)',
            r'([A-Za-z\s]+?)\s*(?:RGB|CMYK|Hex)',  # Color name before RGB/CMYK/Hex
        ]
        
        for pattern in color_name_contexts:
            name_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in name_matches:
                color_name = match.strip()
                if len(color_name) > 2 and len(color_name) < 30:  # Reasonable color name length
                    hex_color = self._color_name_to_hex(color_name)
                    if hex_color and hex_color not in seen_colors:
                        seen_colors.add(hex_color)
                        uncategorized_colors.append(hex_color)
        
        # Set the categorized colors first, then distribute uncategorized colors
        result['primaryColors'] = primary_colors[:10]  # Limit to 10
        result['secondaryColors'] = secondary_colors[:10]
        result['accentColors'] = accent_colors[:10]
        
        # If we have categorized colors, use them. Otherwise, distribute uncategorized colors
        if not primary_colors and not secondary_colors and not accent_colors and uncategorized_colors:
            num_colors = len(uncategorized_colors)
            result['primaryColors'] = uncategorized_colors[:max(1, num_colors // 2)]
            result['secondaryColors'] = uncategorized_colors[max(1, num_colors // 2):num_colors - 1] if num_colors > 1 else []
            result['accentColors'] = uncategorized_colors[-1:] if num_colors > 2 else []
        elif uncategorized_colors:
            # If we have some categorized colors, add uncategorized to primary as fallback
            # (or distribute evenly, but primary is most common)
            if not result['primaryColors']:
                result['primaryColors'] = uncategorized_colors[:min(3, len(uncategorized_colors))]
        
        # 7. Extract fonts
        font_patterns = [
            r'Font[:\s]+([A-Za-z\s]+?)(?:\n|RGB|CMYK|Hex|$)',  # "Font: Helvetica"
            r'Font Family[:\s]+([A-Za-z\s]+?)(?:\n|RGB|CMYK|Hex|$)',  # "Font Family: Times New Roman"
            r'Typography[:\s]+([A-Za-z\s]+?)(?:\n|RGB|CMYK|Hex|$)',  # "Typography: Arial"
        ]
        
        fonts = []
        for pattern in font_patterns:
            font_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in font_matches:
                font_name = match.strip()
                if font_name and len(font_name) > 1 and font_name not in fonts:
                    fonts.append(font_name)
        
        # Common font names to look for
        common_fonts = [
            'Helvetica', 'Arial', 'Times New Roman', 'Courier', 'Georgia',
            'Verdana', 'Comic Sans', 'Impact', 'Trebuchet', 'Palatino',
            'Garamond', 'Baskerville', 'Futura', 'Gill Sans', 'Optima'
        ]
        for font in common_fonts:
            if font.lower() in text.lower() and font not in fonts:
                fonts.append(font)
        
        result['fonts'] = fonts[:10]  # Limit to 10 fonts
        
        return result
    
    def extract_from_xml(self, xml_path: str) -> Dict[str, Any]:
        """Extract data from XML file (basic implementation)"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Basic XML parsing - adjust based on your XML structure
            text_content = ET.tostring(root, encoding='unicode', method='text')
            
            return {
                'text': text_content,
                'images': []
            }
        except Exception as e:
            logger.error(f"XML extraction failed: {e}")
            return {'text': '', 'images': []}
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract data from image file"""
        try:
            img = Image.open(image_path)
            return {
                'text': '',  # Would need OCR for text
                'images': [img]
            }
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return {'text': '', 'images': []}
    
    def extract_brand_guidelines(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Main extraction method"""
        try:
            # Extract content based on file type
            if file_type == 'pdf':
                extracted = self.extract_from_pdf(file_path)
            elif file_type == 'xml':
                extracted = self.extract_from_xml(file_path)
            elif file_type in ['png', 'jpg', 'jpeg']:
                extracted = self.extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Method 1: Extract colors from embedded images in PDF
            image_colors = []
            if extracted.get('images'):
                image_colors = self.extract_colors_from_images(extracted['images'])
            
            # Method 2: Extract colors by rendering PDF pages as images (for PDFs)
            rendered_colors = []
            if file_type == 'pdf':
                rendered_colors = self._extract_colors_from_pdf_pages(file_path, max_pages=5)
            
            # Method 3: Try Qwen extraction first
            structured_data = self.extract_with_qwen(extracted.get('text', ''), extracted.get('images'))
            
            # Normalize primaryColor object to primaryColors array for backward compatibility
            primary_color_hex = ""
            if isinstance(structured_data.get('primaryColor'), dict):
                primary_color_hex = structured_data['primaryColor'].get('hex', '')
            elif isinstance(structured_data.get('primaryColor'), str):
                primary_color_hex = structured_data['primaryColor']
            
            # Extract colors for fallback check
            text_colors = []
            if primary_color_hex:
                text_colors.append(primary_color_hex)
            text_colors.extend(structured_data.get('secondaryColors', []))
            text_colors.extend(structured_data.get('accentColors', []))
            
            # Method 4: If Qwen didn't find colors, try text-based extraction as fallback
            if not text_colors and extracted.get('text'):
                logger.info("Qwen extraction found no colors, trying text-based extraction")
                text_based_data = self._extract_from_text(extracted.get('text', ''))
                # Merge results, preferring text-based colors
                if text_based_data.get('primaryColors') or text_based_data.get('secondaryColors') or text_based_data.get('accentColors'):
                    structured_data = text_based_data
                # Merge fonts if Qwen didn't find any
                if not structured_data.get('fonts') and text_based_data.get('fonts'):
                    structured_data['fonts'] = text_based_data['fonts']
            
            # Normalize format: convert primaryColor object to primaryColors array for frontend compatibility
            if isinstance(structured_data.get('primaryColor'), dict):
                primary_hex = structured_data['primaryColor'].get('hex', '')
                if primary_hex:
                    structured_data['primaryColors'] = [primary_hex]
                else:
                    structured_data['primaryColors'] = []
            elif isinstance(structured_data.get('primaryColor'), str):
                structured_data['primaryColors'] = [structured_data['primaryColor']] if structured_data['primaryColor'] else []
            elif structured_data.get('primaryColors') is None:
                # Fallback: use primary color from extraction if available
                structured_data['primaryColors'] = [primary_color_hex] if primary_color_hex else []
            
            # Combine ALL color sources: Qwen/text extraction + embedded images + rendered pages
            all_colors = list(set(
                structured_data.get('primaryColors', []) + 
                structured_data.get('secondaryColors', []) + 
                structured_data.get('accentColors', []) + 
                image_colors +
                rendered_colors
            ))
            
            # If we have a primary color from Qwen, prioritize it
            if primary_color_hex and primary_color_hex not in structured_data.get('primaryColors', []):
                structured_data['primaryColors'] = [primary_color_hex] + structured_data.get('primaryColors', [])
            
            # Ensure we have at least one primary color if colors were found
            if all_colors and not structured_data.get('primaryColors'):
                structured_data['primaryColors'] = [all_colors[0]]
                all_colors = all_colors[1:]
            
            # Distribute remaining colors into secondary/accent if we still have them
            if all_colors:
                num_colors = len(all_colors)
                if not structured_data.get('secondaryColors'):
                    structured_data['secondaryColors'] = all_colors[:max(1, num_colors // 2)]
                if not structured_data.get('accentColors'):
                    structured_data['accentColors'] = all_colors[max(1, num_colors // 2):] if num_colors > 1 else []
            
            return {
                'success': True,
                'data': structured_data,
                'metadata': {
                    'text_length': len(extracted.get('text', '')),
                    'num_images': len(extracted.get('images', [])),
                    'colors_found': len(all_colors),
                    'extraction_methods': {
                        'text_based': len(structured_data.get('primaryColors', []) + structured_data.get('secondaryColors', []) + structured_data.get('accentColors', [])),
                        'embedded_images': len(image_colors),
                        'rendered_pages': len(rendered_colors)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Brand extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_default_structure()
            }

