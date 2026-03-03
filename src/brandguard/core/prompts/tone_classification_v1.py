"""
Tone Classification Prompt - Sensor Only
Classifies tone using only allowed enums, no scoring
"""

PROMPT_VERSION = "tone_classification_v1.0"

def get_tone_classification_prompt(text: str) -> str:
    """
    Generate prompt for tone classification
    
    Args:
        text: Text to classify
        
    Returns:
        Prompt string
    """
    return f"""Classify tone using ONLY the allowed enums.

Text:
"{text}"

Allowed enums:
- tone: ["energetic", "neutral", "calm", "unknown"]
- sentiment: ["positive", "neutral", "negative"]
- confidence_level: ["low", "balanced", "high"]

Return STRICT JSON ONLY:
{{
  "version": "analysis_v1",
  "detected": {{}},
  "observations": {{
    "tone": "enum",
    "sentiment": "enum",
    "confidence_level": "enum"
  }},
  "flags": [],
  "raw_metrics": {{}},
  "confidence": 0.0
}}

Remember:
- No scores
- No explanations
- No "brand fit" judgments
- Only use the allowed enum values
"""

