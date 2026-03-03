"""
Text Extraction Prompt - Sensor Only
Extracts text observations without scoring or judgment
"""

PROMPT_VERSION = "text_extraction_v1.0"

def get_text_extraction_prompt(text: str) -> str:
    """
    Generate prompt for text extraction and error detection
    
    Args:
        text: Text to analyze
        
    Returns:
        Prompt string
    """
    return f"""You are a text analysis engine.

TASK:
- Extract text observations only.
- Do NOT score quality.
- Do NOT decide compliance.
- Do NOT summarize.

Analyze the following text:

"{text}"

Return STRICT JSON ONLY in this format:
{{
  "version": "analysis_v1",
  "detected": {{
    "text": "{text}",
    "sentence_count": number,
    "word_count": number
  }},
  "observations": {{
    "grammar_errors": ["list exact errors"],
    "spelling_errors": ["list spelling mistakes"],
    "punctuation_issues": ["list punctuation issues"]
  }},
  "flags": [
    "grammar_error_detected" | "spelling_error_detected"
  ],
  "raw_metrics": {{}},
  "confidence": 0.0
}}

Remember:
- Only list errors, do not score them
- Do not include grammar_score, compliance.score, or failure_summary
- Empty arrays are allowed if no errors found
"""

