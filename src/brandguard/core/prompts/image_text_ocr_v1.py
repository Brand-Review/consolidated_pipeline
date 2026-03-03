"""
Image Text OCR Prompt - Sensor Only
Extracts visible text from images without analysis
"""

PROMPT_VERSION = "image_text_ocr_v1.0"

def get_image_ocr_prompt() -> str:
    """
    Generate prompt for image OCR
    
    Returns:
        Prompt string
    """
    return """Extract visible text from the image.

Rules:
- Do NOT analyze grammar.
- Do NOT judge quality.
- Extract only what is visible.

Return STRICT JSON ONLY:
{
  "version": "analysis_v1",
  "detected": {
    "texts": [
      {
        "content": "text",
        "bbox": [x, y, w, h]
      }
    ]
  },
  "observations": {},
  "flags": [],
  "raw_metrics": {
    "text_block_count": number
  },
  "confidence": 0.0
}

Remember:
- Grammar analysis is done by text prompt, not image prompt
- Only extract, do not analyze
"""

