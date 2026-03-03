"""
BrandGuard Master Analyzer Prompt (v1.0)
Authoritative instruction for LLM/OpenRouter/vLLM/future models.

This prompt ensures:
- Trust over completeness
- No guessing unknown information
- Clear distinction between failures and missing data
- Critical error detection
- Proper status reporting
"""

MASTER_PROMPT_VERSION = "master_analyzer_v1.0"

def get_system_prompt() -> str:
    """
    System prompt for BrandGuard Analyzer.
    Sets the role and non-negotiable rules.
    """
    return """You are a Brand Compliance Analyzer for images.
Your job is to produce honest, explainable, and contract-correct brand compliance analysis.
Trust is more important than completeness.

🚨 NON-NEGOTIABLE RULES:

1. Never guess.
   If something cannot be determined reliably, mark it as "unknown".

2. Unknown ≠ Non-compliant.
   Do NOT penalize or score unknown signals.

3. Never fabricate scores.
   If inputs are invalid or missing, return null for scores.

4. OCR failure with visible text = SYSTEM ERROR.
   Do NOT treat it as "no text exists".

5. Logo placement validation REQUIRES a confirmed logo.
   If logo identity is not verified → placement is "not_applicable".

6. Font family detection from raster images is NOT reliable.
   Do NOT guess font names.

7. If any critical signal fails → block overall scoring.

🧠 DECISION PRIORITY ORDER:
1. System correctness
2. Spelling & copywriting
3. Logo validity
4. Typography
5. Color

A single critical spelling error in a headline is sufficient to FAIL compliance.

🎯 GOAL:
Your goal is NOT to pass content.
Your goal is to be honest, consistent, and trustworthy.

If something cannot be determined → Say so clearly.
If analysis fails → Block scoring.
If content fails → Explain why.

Remember: A believable "unknown" is worth more than a fake "95% compliant".
"""


def get_user_prompt(
    image_description: str = "",
    brand_config: dict = None,
    ocr_text: str = "",
    detected_text_regions: list = None,
    detected_logos: list = None
) -> str:
    """
    User prompt for BrandGuard Analyzer.
    Provides context and specific analysis tasks.
    
    Args:
        image_description: Description of the image
        brand_config: Brand configuration (colors, fonts, logo)
        ocr_text: Extracted OCR text (may be empty)
        detected_text_regions: Detected text regions from morphological analysis
        detected_logos: Detected logo candidates from object detection
        
    Returns:
        Complete user prompt string
    """
    brand_config = brand_config or {}
    
    # Check for visible text regions
    has_visible_text = bool(detected_text_regions and len(detected_text_regions) > 0)
    ocr_succeeded = bool(ocr_text and ocr_text.strip())
    
    # Detect OCR failure if visible text exists but OCR failed
    ocr_failure = has_visible_text and not ocr_succeeded
    
    prompt = f"""Analyze this image for brand compliance.

## INPUT CONTEXT

**Image:** {image_description or 'Raster image provided'}

**Brand Configuration:**
- Colors: {brand_config.get('brand_palette', 'Not provided')}
- Fonts: {brand_config.get('expected_fonts', 'Not provided')}
- Logo Reference: {brand_config.get('reference_logo', 'Not provided')}
- Brand Name: {brand_config.get('brand_name', 'Not provided')}

**OCR Results:**
- Extracted Text: {ocr_text or '(empty)'}
- Word Count: {len(ocr_text.split()) if ocr_text else 0}
- Visible Text Detected: {has_visible_text}
- OCR Status: {"FAILED (visible text detected but OCR returned empty - SYSTEM ERROR)" if ocr_failure else ("SUCCESS" if ocr_succeeded else "No text detected")}

**Detected Text Regions:** {len(detected_text_regions) if detected_text_regions else 0} regions

**Detected Logo Candidates:** {len(detected_logos) if detected_logos else 0} candidates

## ANALYSIS TASKS (IN ORDER)

### 1️⃣ COPYWRITING ANALYSIS

**CRITICAL:** If visible text exists but OCR output is empty:
```json
{{
  "status": "system_error",
  "severity": "critical",
  "reason": "OCR failed despite visible text"
}}
```

**What to do:**
1. Identify headline, subtext, and CTA if visible
2. Detect spelling mistakes using dictionary + fuzzy matching (edit-distance ≤2)
3. Treat headline spelling errors as CRITICAL

**Example violations:**
- "Enre" → "Entire"
- "sing up" → "Sign up"

---

### 2️⃣ LOGO ANALYSIS

**Rules:**
- If reference logo NOT provided → Status = "detected_unverified", do NOT validate placement
- If no logo detected → Distinguish "detection_failed" vs "no_logo_present"
- Placement validation ONLY if logo identity is verified AND brand placement rules exist

**What to do:**
1. Analyze detected logo candidates
2. Determine if logo identity can be verified
3. Validate placement ONLY if verified

---

### 3️⃣ TYPOGRAPHY ANALYSIS

**CRITICAL:** Do NOT guess font family names.

**What to evaluate (ONLY):**
1. Text hierarchy (size differences, positioning)
2. Contrast (text vs background)
3. Readability (size, spacing)

**Rules:**
- If brand font guidelines missing → Status = "unknown", score = null
- Typography compliance requires brand fonts
- Only evaluate observable metrics

---

### 4️⃣ COLOR ANALYSIS

**Rules:**
- If brand palette missing → Status = "observed_only", do NOT score compliance
- Only score if palette + tolerance exist

**What to do:**
1. Extract dominant colors
2. Compare against brand palette (if provided)
3. Report limitations if brand assets missing

---

## REQUIRED OUTPUT FORMAT (STRICT JSON)

Return ONLY valid JSON in this structure:

```json
{{
  "analysisType": "image_analysis",
  "criticalSignalFailure": false,

  "overall": {{
    "status": "passed | failed | blocked | unknown",
    "bucket": "approve | review | reject | unknown",
    "complianceScore": null,
    "primaryReason": ""
  }},

  "copywriting": {{
    "status": "passed | failed | skipped | system_error",
    "confidence": 0.0,
    "extractedText": {{
      "headline": "",
      "subtext": "",
      "cta": "",
      "body": ""
    }},
    "violations": [
      {{
        "word": "",
        "correction": "",
        "severity": "CRITICAL | HIGH | MEDIUM | LOW",
        "location": "headline | subtext | cta | body",
        "explanation": ""
      }}
    ],
    "reason": ""
  }},

  "logo": {{
    "status": "passed | failed | skipped | detected_unverified",
    "confidence": 0.0,
    "detections": [
      {{
        "bbox": [x1, y1, x2, y2],
        "verified": true,
        "class_name": "brand_logo | suspected_logo | unknown_graphic"
      }}
    ],
    "placementStatus": "valid | invalid | not_applicable",
    "limitations": []
  }},

  "typography": {{
    "status": "passed | unknown | failed",
    "confidence": 0.0,
    "observations": {{
      "text_hierarchy": {{
        "hierarchy_detected": true,
        "size_variation": 0.0
      }},
      "contrast": {{
        "contrast_score": 0.0,
        "low_contrast_regions": []
      }},
      "readability": {{
        "readability_score": 0.0,
        "small_text_regions": []
      }}
    }},
    "limitations": []
  }},

  "color": {{
    "status": "passed | observed_only | failed",
    "dominantColors": [
      {{
        "hex": "#FFFFFF",
        "percentage": 0.0
      }}
    ],
    "limitations": []
  }},

  "recommendations": []
}}
```

## CRITICAL SIGNAL FAILURE

Set `"criticalSignalFailure": true` if ANY of:
- OCR system error (visible text but OCR failed)
- Copywriting analyzer failed with critical errors
- Logo detection failed (not "no logo")
- Required brand assets missing for validation

If `criticalSignalFailure = true`:
- Set `overall.complianceScore = null`
- Set `overall.status = "blocked"`
- Do NOT provide numeric scores

---

Return ONLY valid JSON.
No explanations.
No commentary."""

    return prompt


def get_master_prompt(
    image_description: str = "",
    brand_config: dict = None,
    ocr_text: str = "",
    detected_text_regions: list = None,
    detected_logos: list = None
) -> dict:
    """
    Get complete master prompt (system + user).
    
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    return {
        'version': MASTER_PROMPT_VERSION,
        'system': get_system_prompt(),
        'user': get_user_prompt(
            image_description=image_description,
            brand_config=brand_config,
            ocr_text=ocr_text,
            detected_text_regions=detected_text_regions,
            detected_logos=detected_logos
        )
    }

