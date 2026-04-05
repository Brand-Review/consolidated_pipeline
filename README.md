# BrandGuard Consolidated Pipeline

The Python ML backend for BrandReview. Orchestrates four independent brand-compliance analyzers — Color, Typography, Copywriting, and Logo Detection — behind a single Flask API.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Brand Profiles](#brand-profiles)
- [Analyzer Details](#analyzer-details)
- [System Requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
consolidated_pipeline/
├── app.py                          # Flask entry point (port 5001)
├── app_fastapi.py                  # FastAPI alternative entry point
├── configs/
│   ├── color_palette.yaml
│   ├── typography_rules.yaml
│   ├── brand_voice.yaml
│   ├── logo_detection.yaml
│   └── production.yaml
└── src/brandguard/
    ├── config/settings.py          # Typed settings dataclasses
    ├── core/
    │   ├── base_orchestrator.py    # Core orchestration logic
    │   ├── pipeline_orchestrator_new.py  # Active orchestrator (thin subclass)
    │   ├── color_analyzer.py
    │   ├── typography_analyzer.py
    │   ├── copywriting_analyzer.py
    │   ├── logo_analyzer.py
    │   ├── brand_compliance_judge.py  # OpenRouter LLM judge
    │   └── model_imports.py        # Dynamic model loader
    └── brand_profile/
        ├── brand_store.py          # MongoDB brand registry
        ├── pdf_extractor.py        # PDF guideline ingestion
        ├── text_rag.py             # Qdrant text retrieval
        └── asset_rag.py            # Qdrant image retrieval
```

### Request Lifecycle

```
POST /api/analyze
       │
       ▼
  Flask (app.py) ──► background thread
                           │
                           ▼
               PipelineOrchestrator.analyze_content()
                           │
               ┌───────────┼───────────────┬──────────────┐
               ▼           ▼               ▼              ▼
         ColorAnalyzer  LogoAnalyzer  TypographyAnalyzer  CopywritingAnalyzer
               │           │               │              │
               └───────────┴───────────────┴──────────────┘
                                    │
                           BrandComplianceJudge (optional)
                                    │
                                    ▼
                           POST callback_url  ←─ br-be receives results
```

The main `/api/analyze` endpoint is **async**: it returns `{job_id, status: "processing"}` immediately (HTTP 202) and dispatches analysis to a background thread. Results are delivered via a `callback_url` POST once complete. All per-analyzer endpoints (`/api/analyze/color`, etc.) are **synchronous**.

---

## Quick Start

### Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.8+ | Runtime |
| MongoDB | Brand profile storage |
| Qdrant | RAG vector store for brand profiles |
| vLLM server (optional) | Logo detection fallback + copywriting |
| OpenRouter API key (optional) | Copywriting LLM via `HybridToneAnalyzer` |

### Installation

```bash
cd consolidated_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY, MONGODB_URI, QDRANT_URL
```

### Start the server

```bash
# Development
python app.py

# With vLLM for logo detection / copywriting (optional)
cd ../LogoDetector && python setup_vllm.py &
cd ../consolidated_pipeline && python app.py
```

### Verify

```bash
curl http://localhost:5001/api/health
```

```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "settings_loaded": true,
  "version": "1.0.0"
}
```

---

## API Reference

Interactive docs: `http://localhost:5001/api/docs` (Swagger UI)
OpenAPI spec: `GET /api/openapi.json`

### `POST /api/analyze` — Full analysis (async)

Accepts `multipart/form-data`. Returns **HTTP 202** immediately; results are POSTed to `callback_url` when ready.

**Core fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | — | Image or PDF to analyze |
| `input_type` | string | `file` | `file` \| `text` \| `url` |
| `text_content` | string | — | Required when `input_type=text` |
| `url` | string | — | Required when `input_type=url` |
| `callback_url` | string | — | br-be endpoint to receive results |
| `brand_id` | string | — | Registered brand profile ID |
| `pass_threshold` | float | `0.70` | Minimum passing compliance score |
| `scoring_weights` | JSON string | `{}` | Override per-analyzer weights |

**Analyzer toggles**

| Field | Default |
|---|---|
| `enable_color` | `true` |
| `enable_typography` | `true` |
| `enable_copywriting` | `true` |
| `enable_logo` | `true` |

**Color fields**

| Field | Default | Description |
|---|---|---|
| `primary_colors` | `""` | Comma-separated hex values, e.g. `#1E40AF,#FFFFFF` |
| `secondary_colors` | `""` | Comma-separated hex values |
| `accent_colors` | `""` | Comma-separated hex values |
| `primary_threshold` | `75` | CIEDE2000 match threshold (0–100) |
| `secondary_threshold` | `75` | |
| `accent_threshold` | `75` | |
| `color_tolerance` | `2.3` | Delta-E tolerance for raw extraction |
| `enable_contrast_check` | `true` | WCAG 2.1 contrast validation |

**Logo fields**

| Field | Default | Description |
|---|---|---|
| `logo_confidence_threshold` | `0.5` | Minimum detection confidence |
| `max_logo_detections` | `100` | Cap on detections returned |
| `enable_placement_validation` | `true` | Zone + size checks |
| `generate_annotations` | `true` | Base64 annotated image in response |
| `allowed_zones` | `top-left,...` | Comma-separated allowed placement zones |
| `min_logo_size` | `0.01` | Fraction of image area |
| `max_logo_size` | `0.25` | Fraction of image area |

**Brand voice fields**

| Field | Default | Description |
|---|---|---|
| `formality_score` | `60` | 0–100 |
| `confidence_level` | `balanced` | `conservative` \| `balanced` \| `aggressive` |
| `warmth_score` | `50` | 0–100 |
| `energy_score` | `50` | 0–100 |

**Response (202)**

```json
{ "job_id": "uuid", "status": "processing" }
```

**Callback payload (delivered to `callback_url`)**

```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": {
    "overall_compliance_score": 0.82,
    "model_results": {
      "color_analysis":      { "compliance_score": 0.91, ... },
      "typography_analysis": { "compliance_score": 0.75, ... },
      "copywriting_analysis":{ "compliance_score": 0.88, ... },
      "logo_analysis":       { "compliance_score": 0.74, ... }
    },
    "summary": {
      "passed_checks": 2,
      "warnings": 2,
      "critical_issues": 0
    },
    "recommendations": [...]
  }
}
```

---

### `POST /api/analyze/color` — Synchronous color-only

```bash
curl -X POST http://localhost:5001/api/analyze/color \
  -F "file=@ad.jpg" \
  -F "primary_colors=#1E40AF,#FFFFFF" \
  -F "primary_threshold=75"
```

---

### `POST /api/analyze/typography` — Synchronous typography-only

```bash
curl -X POST http://localhost:5001/api/analyze/typography \
  -F "file=@ad.jpg" \
  -F "confidence_threshold=0.7"
```

---

### `POST /api/analyze/copywriting` — Synchronous copywriting-only

```bash
# From image
curl -X POST http://localhost:5001/api/analyze/copywriting \
  -F "file=@ad.jpg" \
  -F "formality_score=70"

# From raw text
curl -X POST http://localhost:5001/api/analyze/copywriting \
  -F "input_type=text" \
  -F "text_content=Your message here" \
  -F "formality_score=70"
```

---

### `POST /api/analyze/logo` — Synchronous logo-only

```bash
curl -X POST http://localhost:5001/api/analyze/logo \
  -F "file=@ad.jpg" \
  -F "enable_placement_validation=true" \
  -F "generate_annotations=true"
```

---

### Brand Profile Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/brand/onboard` | Ingest brand guidelines PDF + example images |
| `GET` | `/api/brand` | List all registered brands |
| `GET` | `/api/brand/<brand_id>` | Get brand profile by ID |
| `DELETE` | `/api/brand/<brand_id>` | Delete brand + Qdrant collections |

**Onboarding a brand**

```bash
curl -X POST http://localhost:5001/api/brand/onboard \
  -F "brand_name=Acme Corp" \
  -F "brand_id=<optional-mongodb-folder-id>" \
  -F "guideline_pdf=@brand_guidelines.pdf" \
  -F "approved_images=@approved1.jpg" \
  -F "rejected_images=@rejected1.jpg" \
  -F 'rejection_reasons=[["off-brand colors","wrong logo placement"]]'
```

Response:
```json
{
  "success": true,
  "brand_id": "...",
  "brand_name": "Acme Corp",
  "extracted_rules": { ... },
  "chunks_indexed": 48,
  "assets_indexed": 3
}
```

---

### Other Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/config` | Active configuration snapshot |
| `POST` | `/api/config` | Update runtime configuration |
| `GET` | `/api/status/<analysis_id>` | Analysis job status |
| `GET` | `/api/openapi.json` | OpenAPI 3.1 spec |
| `GET` | `/api/docs` | Swagger UI |

---

## Configuration

Runtime defaults live in `configs/*.yaml`. Settings are parsed into typed dataclasses in `src/brandguard/config/settings.py`.

```
configs/
├── color_palette.yaml      # Brand palette + CIEDE2000 tolerances
├── typography_rules.yaml   # Approved/forbidden fonts, size bounds
├── brand_voice.yaml        # Formality, tone, prohibited content flags
├── logo_detection.yaml     # YOLOv8 thresholds, vLLM URL, placement zones
└── production.yaml         # Environment overrides
```

**Example `color_palette.yaml`**

```yaml
brand_palette:
  name: "Acme Corp"
  primary_colors:
    - name: "Brand Blue"
      hex: "#1E40AF"
      tolerance: 0.1
  secondary_colors:
    - name: "Slate Gray"
      hex: "#64748B"
      tolerance: 0.15
  accent_colors:
    - name: "Orange"
      hex: "#F97316"
      tolerance: 0.1
```

**Example `brand_voice.yaml`**

```yaml
brand_voice:
  formality_score: 70
  confidence_level: "balanced"
  warmth_score: 60
  energy_score: 50
  readability_level: "grade8"
  persona_type: "professional"
  allow_emojis: false
  allow_slang: false
  no_financial_guarantees: true
  no_medical_claims: true
  no_competitor_bashing: true
```

**Environment variables**

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Recommended | Powers `HybridToneAnalyzer` and `BrandComplianceJudge` |
| `INTERNAL_WEBHOOK_SECRET` | Yes (matches br-be) | Signs callback POSTs |
| `MONGODB_URI` | Yes | Brand profile persistence |
| `QDRANT_URL` | Yes | RAG vector store |

---

## Brand Profiles

When a `brand_id` is supplied to `/api/analyze`, the pipeline enriches analysis with brand-specific context:

1. **PDF extraction** — `PDFRuleExtractor` parses the brand guideline PDF into structured rules + text chunks.
2. **Text RAG** — Chunks are embedded and indexed in Qdrant. At analysis time, relevant rules are retrieved and passed to `BrandComplianceJudge`.
3. **Asset RAG** — Approved and rejected example images are embedded in Qdrant. Visual similarity against known rejects flags potential violations.
4. **LLM judge** — `BrandComplianceJudge` synthesizes retrieved rules + analyzer outputs via OpenRouter (Qwen2.5-VL-32B-Instruct) to produce a final verdict with citations.

Without a `brand_id`, analyzers use the static config defaults in `configs/`.

---

## Analyzer Details

### Color

- **Extraction**: K-means clustering (default 8 clusters) to identify dominant colors.
- **Matching**: CIEDE2000 (Delta E 2000) perceptual color difference against the brand palette.
- **Thresholds**: Per-category (`primary_threshold`, `secondary_threshold`, `accent_threshold`) on a 0–100 scale.
- **Accessibility**: WCAG 2.1 contrast ratio validation when `enable_contrast_check=true`.
- **Default behavior**: If no brand palette is provided, color returns `compliance_score: 1.0` (no palette to validate against).

### Typography

- **OCR**: PaddleOCR (PP-OCRv5) for text region detection and extraction.
- **Font identification**: CNN model trained on 49 font categories.
- **Validation**: Checks identified fonts against `approved_fonts` / `forbidden_fonts` lists, font-size bounds, line-height, and letter-spacing rules from config.
- **Default behavior**: Returns neutral score of 0.5 when no font rules are configured; applies `+0.1` bonus per approved font found, `-0.1` per non-compliant font.

### Copywriting

- **Primary path**: `HybridToneAnalyzer` — calls vLLM (Qwen2.5-VL-3B-Instruct) first, falls back to OpenRouter (Qwen2.5-VL-32B-Instruct).
- **Fallback**: `VLLMToneAnalyzer` → `ToneAnalyzer` / `BrandVoiceValidator` (traditional NLP).
- **Analysis**: Tone (formality, confidence, warmth, energy), sentiment, grammar, readability, prohibited content detection (financial guarantees, medical claims, competitor references).
- **Score formula**: `base_score × 0.5 + tone_factor × 0.3 + sentiment_factor × 0.2`

### Logo Detection

Two-stage hybrid system:

```
YOLOv8 nano (~50ms)
      │
      ▼ objects found?
  YES ─────────────► convert bounding boxes ─► compliance validation
  NO  ─────────────► Qwen2.5-VL-3B via vLLM ─► compliance validation
```

- **YOLOv8 nano**: Fast primary pass. Configured via `logo_detection.yaml`.
- **Qwen2.5-VL-3B-Instruct**: Multimodal LLM fallback served by vLLM on port 8000.
- **Placement validation**: Zone containment, size fraction, edge clearance, aspect ratio.
- **Score formula**: `(placement_score × 0.5) + (brand_score × 0.5)`

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — | 8 GB (for vLLM) |
| Disk | 5 GB | 15 GB |
| OS | Linux / macOS | Linux |

---

## Performance

| Analyzer | Typical latency |
|---|---|
| Color | ~200 ms |
| Typography | ~500 ms |
| Copywriting (OpenRouter) | ~3–8 s |
| Logo (YOLOv8 path) | ~50–200 ms |
| Logo (Qwen/vLLM path) | ~5–15 s |
| Full pipeline (all analyzers) | ~5–20 s |

---

## Troubleshooting

### vLLM connection refused

```bash
# Check if vLLM is running
curl http://localhost:8000/v1/models

# Start it
cd ../LogoDetector && python setup_vllm.py
```

Logo detection will fall back gracefully to YOLOv8 if vLLM is unavailable. Copywriting will fall back to OpenRouter.

### Logo timeouts

Increase the timeout in `configs/logo_detection.yaml`:

```yaml
qwen_timeout: 180
```

Images larger than 512px are auto-resized before being sent to Qwen.

### Color analysis wrong results

Verify hex format and comma separation:

```bash
# Correct
primary_colors=#1E40AF,#FFFFFF

# Wrong (missing #)
primary_colors=1E40AF,FFFFFF
```

Threshold values must be in 0–100 range (not 0–1).

### Out of memory

```bash
nvidia-smi   # check GPU usage
# Pre-resize large images before upload; max file size is 50 MB
```

### Port already in use

```bash
lsof -i :5001   # find the process
kill <PID>
```

### Models fail to load

```bash
python -c "import torch; print(torch.cuda.is_available())"
pip install -r requirements.txt --upgrade
```

### Debug logging

```python
# In app.py or via environment
import logging
logging.basicConfig(level=logging.DEBUG)
```
