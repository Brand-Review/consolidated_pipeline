"""
Typography Detection Prompt - Sensor Only
Detects typography properties without judgment
"""

PROMPT_VERSION = "typography_detection_v1.0"

def get_typography_detection_prompt() -> str:
    """
    Generate prompt for typography detection
    
    Returns:
        Prompt string
    """
    return """Detect typography properties only.

Return STRICT JSON ONLY:
{
  "version": "analysis_v1",
  "detected": {
    "fonts": [
      {
        "family": "string or unknown",
        "weight": "light|regular|bold|unknown",
        "style": "normal|italic|unknown"
      }
    ]
  },
  "observations": {},
  "flags": [],
  "raw_metrics": {},
  "confidence": 0.0
}

Remember:
- Only detect, do not judge compliance
- Use "unknown" if cannot determine
"""

