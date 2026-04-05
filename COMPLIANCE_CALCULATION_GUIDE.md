# BrandGuard Compliance Calculation Guide

How scores are computed at every level — from individual color comparisons to the final `overall_compliance_score` returned by the API.

---

## Table of Contents

- [Overall Score](#overall-score)
- [Color Analysis](#1-color-analysis)
- [Typography Analysis](#2-typography-analysis)
- [Copywriting Analysis](#3-copywriting-analysis)
- [Logo Analysis](#4-logo-analysis)
- [Score Thresholds](#score-thresholds)
- [Custom Weights](#custom-weights)
- [Configuration Reference](#configuration-reference)

---

## Overall Score

### Formula

```
overall_compliance_score = Σ ( analyzer_score × weight )
```

Default weights (equal distribution):

| Analyzer | Weight |
|---|---|
| Color analysis | 0.25 |
| Typography analysis | 0.25 |
| Copywriting analysis | 0.25 |
| Logo analysis | 0.25 |

Only analyzers that **ran successfully** contribute to the weighted sum. If an analyzer is disabled or fails, its weight is excluded and the remaining weights retain their relative proportions.

### Implementation

```python
# src/brandguard/core/base_orchestrator.py
def _calculate_overall_compliance(self):
    scores = []
    weights = {
        'color_analysis':      0.25,
        'typography_analysis': 0.25,
        'copywriting_analysis':0.25,
        'logo_analysis':       0.25,
    }
    for model_name, weight in weights.items():
        if model_name in self.analysis_results['model_results']:
            model_result = self.analysis_results['model_results'][model_name]
            if 'compliance_score' in model_result:
                scores.append(model_result['compliance_score'] * weight)

    if scores:
        self.analysis_results['overall_compliance_score'] = round(sum(scores), 3)
```

### Example

```
Color Analysis       = 0.91  →  0.91 × 0.25 = 0.2275
Typography Analysis  = 0.75  →  0.75 × 0.25 = 0.1875
Copywriting Analysis = 0.88  →  0.88 × 0.25 = 0.2200
Logo Analysis        = 0.74  →  0.74 × 0.25 = 0.1850

Overall = 0.2275 + 0.1875 + 0.2200 + 0.1850 = 0.820
```

---

## 1. Color Analysis

**Source:** `src/brandguard/core/color_analyzer.py`

### Step 1 — Color Extraction

K-Means clustering is applied to the image pixels to identify dominant colors. Default: 8 clusters.

```python
# n_colors / n_clusters configurable via request form field
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_  # RGB values
```

### Step 2 — Color Matching (CIEDE2000)

Each extracted color is compared against every color in the brand palette using CIEDE2000 (Delta E 2000), which models human perceptual color difference more accurately than simple RGB Euclidean distance.

```python
delta_e = colour.delta_E(lab_extracted, lab_brand, method='CIE 2000')
similarity = max(0.0, 1.0 - (delta_e / 100.0))
```

The threshold on the API (`primary_threshold`, etc.) is a 0–100 scale representing minimum required similarity × 100.

### Step 3 — Compliance Score

```python
compliance_score = len(compliant_colors) / len(total_extracted_colors)
```

A color is compliant if its CIEDE2000 similarity to the closest brand palette color meets or exceeds the configured threshold.

### WCAG 2.1 Contrast Check

When `enable_contrast_check=true`, luminance-based contrast ratios are computed for foreground/background pairs extracted from the image:

```python
contrast_ratio = (L1 + 0.05) / (L2 + 0.05)   # L1 > L2
# WCAG AA:  normal text ≥ 4.5,  large text ≥ 3.0
# WCAG AAA: normal text ≥ 7.0,  large text ≥ 4.5
```

Contrast violations appear as warnings in the response but do not directly reduce `compliance_score`.

### Default Behavior

When no brand palette colors are provided:
```python
return { 'compliance_score': 1.0,
         'warning': 'No brand colors defined — cannot validate compliance' }
```

---

## 2. Typography Analysis

**Source:** `src/brandguard/core/typography_analyzer.py`
**Config:** `configs/typography_rules.yaml`, `src/brandguard/config/settings.py`

### Step 1 — Text Detection

PaddleOCR (PP-OCRv5) detects text regions and extracts strings. Regions closer than `distance_threshold` pixels are optionally merged (`merge_regions=true`).

### Step 2 — Font Identification

A CNN model classifies each detected text region into one of 49 font categories. Results below `confidence_threshold` (default `0.7`) are discarded.

### Step 3 — Compliance Scoring

```python
def _calculate_typography_score(self, compliance_results):
    base_score = compliance_results.get('compliance_score', 0.0)

    approved_count     = len(compliance_results.get('approved_fonts', []))
    non_compliant_count= len(compliance_results.get('non_compliant_fonts', []))

    if approved_count > 0:
        base_score += 0.1                      # bonus: approved font found

    if non_compliant_count > 0:
        base_score -= 0.1 * non_compliant_count  # penalty per violation

    return max(0.0, min(1.0, base_score))
```

Additional rule checks that affect `base_score`:
- Font size outside `[min_font_size, max_font_size]`
- Line-height ratio outside configured bounds
- Letter-spacing out of range

### Default Behavior

When no font rules are configured:
- `compliance_score: 0.5` (neutral starting point)
- All detected fonts marked as "unknown compliance"
- No bonuses or penalties applied

Default approved fonts (from `settings.py`): `Arial`, `Helvetica`, `Times New Roman`, `Georgia`

---

## 3. Copywriting Analysis

**Source:** `src/brandguard/core/copywriting_analyzer.py`
**Config:** `configs/brand_voice.yaml`, `src/brandguard/config/settings.py`

### Score Formula

```python
def _calculate_copywriting_score(self, tone_results, voice_compliance):
    base_score     = voice_compliance.get('score', 0.5)

    tone_confidence = tone_results.get('formality', {}).get('formality_score', 0.5)
    tone_factor     = tone_confidence * 0.3         # 30% weight

    sentiment       = tone_results.get('sentiment', {}).get('overall_sentiment', 'neutral')
    sentiment_score = {'positive': 0.8, 'neutral': 0.5, 'negative': 0.2}[sentiment]
    sentiment_factor= sentiment_score * 0.2          # 20% weight

    # 50% brand voice compliance + 30% tone match + 20% sentiment
    return max(0.0, min(1.0, base_score * 0.5 + tone_factor + sentiment_factor))
```

### Brand Voice Compliance (`base_score`)

`voice_compliance['score']` is derived by comparing measured attributes to the configured brand voice targets:

| Attribute | Pass condition |
|---|---|
| Formality | Measured value within ±20 of `formality_score` target |
| Confidence level | Matches `confidence_level` setting |
| Sentiment | Appropriate for brand context |
| Readability | Flesch–Kincaid grade matches `readability_level` |
| Prohibited content | No financial guarantees, medical claims, or competitor references found |
| Emoji / slang | Absent when `allow_emojis=false` / `allow_slang=false` |

### Configurable Brand Voice Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `formality_score` | int 0–100 | 60 | Target formality level |
| `confidence_level` | string | `balanced` | `conservative` \| `balanced` \| `aggressive` |
| `warmth_score` | int 0–100 | 50 | Target warmth/friendliness |
| `energy_score` | int 0–100 | 50 | Target energy/urgency |
| `readability_level` | string | `grade8` | `grade6` \| `grade8` \| `grade10` \| `grade12` |
| `persona_type` | string | `general` | `general` \| `professional` \| `casual` \| `friendly` |
| `allow_emojis` | bool | `false` | |
| `allow_slang` | bool | `false` | |
| `no_financial_guarantees` | bool | `true` | Flag guarantee language |
| `no_medical_claims` | bool | `true` | Flag medical claim language |
| `no_competitor_bashing` | bool | `true` | Flag competitor references |

### LLM Path

When `HybridToneAnalyzer` is active (default), the full text + image is sent to Qwen2.5-VL-3B-Instruct (via vLLM or OpenRouter). The model returns structured JSON with individual attribute scores that map directly into the formula above.

### Default Behavior

When no brand voice config is supplied:
```python
BrandVoiceSettings(
    formality_score=60,
    confidence_level="balanced",
    warmth_score=50,
    energy_score=50,
    readability_level="grade8",
    persona_type="general",
    allow_emojis=False,
    allow_slang=False,
)
```
Analysis evaluates against general professional writing best practices.

---

## 4. Logo Analysis

**Source:** `src/brandguard/core/logo_analyzer.py`
**Config:** `configs/logo_detection.yaml`

### Score Formula

```python
def _calculate_logo_compliance_score(self, detections, placement_validation, brand_compliance):
    placement_score = placement_validation.get('compliance_score', 0)
    brand_score     = brand_compliance.get('compliance_score', 0)

    return round((placement_score * 0.5) + (brand_score * 0.5), 2)
```

Equal weight between placement correctness and brand identity compliance.

### Placement Validation Score

Each detected logo is evaluated against placement rules. `placement_score` is the fraction of logos that pass all checks:

| Check | Default rule |
|---|---|
| Placement zone | Must be in one of `allowed_zones` (default: all four corners) |
| Size (min) | Logo area ≥ `min_logo_size` × image area (default: 1%) |
| Size (max) | Logo area ≤ `max_logo_size` × image area (default: 25%) |
| Edge clearance | All edges ≥ `min_edge_distance` × image dimension (default: 5%) |
| Aspect ratio | Within ±`aspect_ratio_tolerance` of reference (default: ±20%) |

### Brand Compliance Score

- If the logo matches a known brand-approved logo: pass.
- If multiple conflicting brand logos are detected: warning (partial credit).
- If no logo is detected (and logo detection was expected): 0.

### Detection Method Impact

| Detection path | Notes |
|---|---|
| YOLOv8 nano | Bounding boxes from COCO-trained model; may detect objects that aren't logos (general object detection) |
| Qwen2.5-VL-3B (vLLM) | Activated only when YOLOv8 finds nothing; provides richer brand identity context |

### Default Configuration

```python
# settings.py
LogoDetectionSettings(
    confidence_threshold=0.5,
    max_detections=100,
    allowed_zones=["top-left", "top-right", "bottom-left", "bottom-right"],
    min_logo_size=0.01,     # 1% of image area
    max_logo_size=0.25,     # 25% of image area
    min_edge_distance=0.05, # 5% margin
    aspect_ratio_tolerance=0.2,
)
```

---

## Score Thresholds

Applied to both the overall score and each individual analyzer score:

| Range | Status | Pipeline action |
|---|---|---|
| 0.90 – 1.00 | Passed | `passed_checks += 1` |
| 0.70 – 0.89 | Warning | `warnings += 1` |
| 0.50 – 0.69 | Moderate | `critical_issues += 1` |
| 0.00 – 0.49 | Critical | `critical_issues += 1` |

The default `pass_threshold` for the overall score is `0.70`. Pass it as a form field to override:

```bash
-F "pass_threshold=0.80"
```

---

## Custom Weights

Pass `scoring_weights` as a JSON string to shift emphasis per content type:

```bash
# Text-heavy ad: prioritize copywriting
-F 'scoring_weights={"color_analysis":0.10,"typography_analysis":0.20,"copywriting_analysis":0.60,"logo_analysis":0.10}'

# Logo-centric asset: prioritize logo + color
-F 'scoring_weights={"color_analysis":0.35,"typography_analysis":0.15,"copywriting_analysis":0.10,"logo_analysis":0.40}'
```

Weights must sum to 1.0. If they do not, the orchestrator normalizes them.

---

## Configuration Reference

### File locations

```
consolidated_pipeline/
├── configs/
│   ├── color_palette.yaml      # Brand palette + tolerances
│   ├── typography_rules.yaml   # Font lists, size bounds
│   ├── brand_voice.yaml        # Voice settings + prohibited content
│   └── logo_detection.yaml     # YOLOv8 + vLLM settings, placement zones
└── src/brandguard/config/
    └── settings.py             # Typed dataclasses + defaults
```

### Relevant source locations

| Calculation | File | Approx. line |
|---|---|---|
| Overall score | `base_orchestrator.py` | `_calculate_overall_compliance` |
| Score thresholds | `base_orchestrator.py` | `_build_summary` |
| Color extraction | `color_analyzer.py` | `_extract_colors` |
| Color compliance | `color_analyzer.py` | `_calculate_brand_compliance` |
| Typography score | `typography_analyzer.py` | `_calculate_typography_score` |
| Copywriting score | `copywriting_analyzer.py` | `_calculate_copywriting_score` |
| Logo compliance | `logo_analyzer.py` | `_calculate_logo_compliance_score` |
| Default settings | `config/settings.py` | `BrandVoiceSettings`, `LogoDetectionSettings` |

---

## Summary: With vs. Without User Settings

| Analyzer | With settings | Without settings |
|---|---|---|
| **Color** | CIEDE2000 match against provided palette | Auto-pass (`1.0`) — no palette to validate |
| **Typography** | Exact font whitelist/blacklist + size rules | Neutral (`0.5`) — general best practices |
| **Copywriting** | Scored against brand voice targets | General professional writing quality |
| **Logo** | Zone + size rules from config | Standard placement zones, 1%–25% size range |

Settings are always optional. The system returns meaningful scores with or without them, but brand-specific settings produce stricter, more actionable results.
