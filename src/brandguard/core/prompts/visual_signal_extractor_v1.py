"""
Visual Signal Extraction Engine Prompt (v1.0)
Pure signal extraction - NO scoring, NO compliance judgment.
Only extracts verifiable signals from images.
"""

VISUAL_SIGNAL_EXTRACTOR_VERSION = "visual_signal_extractor_v1.0"


def get_visual_signal_extractor_system_prompt() -> str:
    """
    System prompt for Visual Signal Extraction Engine.
    Sets the role: extract signals only, do not score or judge.
    """
    return """SYSTEM ROLE:
You are a visual signal extraction engine for brand compliance.
You DO NOT score.
You DO NOT judge compliance.
You ONLY extract verifiable signals from the image.

INPUT:
A single marketing image.

YOUR TASKS (MANDATORY):

1. LOGO DETECTION
- Detect any visible logo (symbol or wordmark).
- If brand name text is stylized or isolated, treat it as a logo.
- Report:
  - type: symbol | wordmark | combined
  - approximate position: top-left | top-center | top-right | center | bottom-left | bottom-center | bottom-right
  - confidence (0.0–1.0)

2. TEXT PRESENCE CHECK (NOT OCR)
- Decide if readable text is visually present.
- If yes, set `visibleTextDetected = true`.
- Do NOT attempt spelling correction unless text is clearly readable.

3. SPELLING & PHRASE SIGNALS
- If readable text exists:
  - Flag any word that is NOT a valid English word in context.
  - Example: "Enre" is invalid → suggest "Entire".
  - Detect action-phrase errors:
    - "sing up" → should be "sign up"
    - "signup" vs "sign up" → style mismatch (low severity)
- DO NOT normalize or auto-fix.
- Only report errors you are confident (>0.85).

4. COLOR EXTRACTION
- Extract dominant visible colors.
- Return HEX if confident, otherwise approximate name.
- Separate:
  - background
  - primary accent
  - text color

5. FAILURE HANDLING
- If any task cannot be completed, explain WHY.
- NEVER return empty arrays if content is visible.
- NEVER claim "no issues" when confidence < 0.8.

OUTPUT RULES:
- Return STRICT JSON.
- No markdown.
- No explanations outside JSON."""


def get_visual_signal_extractor_prompt() -> str:
    """
    User prompt for Visual Signal Extraction Engine.
    Provides the JSON output format specification.
    """
    return """Extract visual signals from the provided image.

Return STRICT JSON ONLY (no markdown, no explanations, no text before or after JSON):

{
  "logoSignals": {
    "detected": true | false,
    "logos": [
      {
        "type": "symbol" | "wordmark" | "combined",
        "position": "top-left" | "top-center" | "top-right" | "center" | "bottom-left" | "bottom-center" | "bottom-right",
        "confidence": 0.0,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "failureReason": null
  },
  "textSignals": {
    "visibleTextDetected": true | false,
    "confidence": 0.0,
    "extractedText": "",
    "spellingErrors": [
      {
        "word": "string",
        "suggestion": "string",
        "confidence": 0.85,
        "location": "headline" | "body" | "cta" | "unknown"
      }
    ],
    "phraseErrors": [
      {
        "phrase": "string",
        "suggestion": "string",
        "severity": "low" | "medium" | "high",
        "confidence": 0.85
      }
    ],
    "failureReason": null
  },
  "colorSignals": {
    "background": {
      "hex": "#FFFFFF",
      "name": "white",
      "confidence": 0.0
    },
    "primaryAccent": {
      "hex": "#000000",
      "name": "black",
      "confidence": 0.0
    },
    "textColor": {
      "hex": "#000000",
      "name": "black",
      "confidence": 0.0
    },
    "dominantColors": [
      {
        "hex": "#FFFFFF",
        "name": "white",
        "confidence": 0.0,
        "percentage": 0.0
      }
    ],
    "failureReason": null
  },
  "signalsExtracted": true,
  "confidence": 0.0
}

CRITICAL RULES:
1. If text is visible but extraction failed → set failureReason with explanation, visibleTextDetected = true
2. Only report spelling errors with confidence ≥ 0.85 (high confidence required)
3. If logo is visible but not detected → set failureReason explaining why
4. Never return empty arrays when content is visible (use failureReason instead)
5. Never claim "no issues" when confidence < 0.8
6. If any task cannot be completed, explain WHY in failureReason
7. DO NOT normalize or auto-fix - only report errors you detect
8. DO NOT score or judge compliance - only extract signals"""


def get_visual_signal_extractor_complete() -> dict:
    """
    Get complete visual signal extractor prompt (system + user).
    
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    return {
        'version': VISUAL_SIGNAL_EXTRACTOR_VERSION,
        'system': get_visual_signal_extractor_system_prompt(),
        'user': get_visual_signal_extractor_prompt()
    }

