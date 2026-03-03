"""
Brand Evidence Extraction Prompt - Sensor Only
Extracts explicit brand evidence without governance decisions
"""

PROMPT_VERSION = "brand_evidence_extraction_v1.0"

def get_brand_evidence_prompt(text: str) -> str:
    """
    Generate prompt for brand evidence extraction
    
    Args:
        text: Document text to analyze
        
    Returns:
        Prompt string
    """
    return f"""Extract explicit brand evidence from this document.

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
{text[:4000]}

Remember:
- Code decides primary color using rules, not AI
- Only extract evidence, do not make governance decisions
"""

