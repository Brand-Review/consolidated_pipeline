# BrandGuard Consolidated Pipeline

The Python ML backend for BrandReview. Orchestrates four independent brand-compliance analyzers — Color, Typography, Copywriting, and Logo Detection — behind a single Flask API.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Brand Profiles](#brand-profiles)
- [Brand Knowledge RAG](#brand-knowledge-rag-phases-13)
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
│   ├── rag.yaml                   # Phase 1/2/3 chunking, embedding, retrieval, generation
│   └── production.yaml
├── prompts/
│   ├── grounded_answer_v1.yaml    # Phase 3 grounded-answer prompt
│   ├── citation_verifier_v1.yaml  # Phase 3 LLM-as-judge citation check
│   └── brand_compliance_judge_v1.yaml
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
    │   ├── llm_client.py           # Shared OpenRouter chat client (Phase 3)
    │   └── model_imports.py        # Dynamic model loader
    └── brand_profile/
        ├── brand_store.py          # MongoDB brand registry
        ├── document_store.py       # documents[] persistence (Phase 1)
        ├── pdf_extractor.py        # Structured-rule extraction (LLM)
        ├── s3_client.py            # Raw + processed upload/download (Phase 1)
        ├── embeddings.py           # E5 dense + Qdrant BM25 sparse encoders
        ├── deduper.py              # Within-brand cosine ≥ 0.95 dedup
        ├── loaders/                # Phase 1 multi-format ingestion
        │   ├── pdf_loader.py       # PyMuPDF + heading detection
        │   ├── markdown_loader.py
        │   ├── html_loader.py
        │   └── text_loader.py
        ├── chunkers/               # Phase 1 pluggable chunking strategies
        │   ├── fixed_chunker.py
        │   ├── recursive_chunker.py
        │   └── semantic_chunker.py
        ├── retrieval/              # Phase 2 hybrid retrieval
        │   ├── hybrid_retriever.py # dense + BM25 + RRF + rerank
        │   ├── rerankers/          # cross-encoder rerank
        │   └── types.py
        ├── generation/             # Phase 3 grounded RAG pipeline
        │   ├── grounded_pipeline.py    # Orchestrator behind /ask
        │   ├── grounded_generator.py   # Cited-answer LLM call
        │   ├── citation_parser.py      # Extract [N] claims
        │   ├── citation_verifier.py    # LLM-as-judge support check
        │   ├── completeness_judge.py
        │   ├── confidence_scorer.py    # composite = 0.4 ret + 0.4 cov + 0.2 comp
        │   └── idk_responder.py        # Structured "I don't know" payload
        ├── text_rag.py             # Qdrant write + dense-only legacy read path
        └── asset_rag.py            # Qdrant image retrieval (unchanged)
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
| MongoDB | Brand profile + document metadata storage |
| Qdrant | Hybrid (dense + BM25 sparse) RAG vector store |
| AWS S3 (optional) | Raw + processed brand-document persistence (Phase 1) |
| vLLM server (optional) | Logo detection fallback + copywriting |
| OpenRouter API key | Copywriting + Phase 3 grounded RAG (`/ask`) |

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
| `POST` | `/api/brand/onboard` | Ingest brand guidelines (PDF / Markdown / HTML / TXT) + example images |
| `GET` | `/api/brand` | List all registered brands |
| `GET` | `/api/brand/<brand_id>` | Get brand profile by ID (includes `documents[]`) |
| `DELETE` | `/api/brand/<brand_id>` | Delete brand + Qdrant collections |
| `POST` | `/api/brand/<brand_id>/retrieve` | Phase 2 hybrid retrieval debug endpoint |
| `POST` | `/api/brand/<brand_id>/ask` | Phase 3 grounded, cited Q&A over brand knowledge |

**Onboarding a brand (Phase 1 multi-format ingestion)**

```bash
curl -X POST http://localhost:5001/api/brand/onboard \
  -F "brand_name=Acme Corp" \
  -F "brand_id=<optional-mongodb-folder-id>" \
  -F "documents=@guidelines.pdf" \
  -F "documents=@voice.md" \
  -F "documents=@logo-rules.html" \
  -F "chunking_strategy=recursive" \
  -F "approved_images=@approved1.jpg" \
  -F "rejected_images=@rejected1.jpg" \
  -F 'rejection_reasons=[["off-brand colors","wrong logo placement"]]'
```

`chunking_strategy` is one of `fixed`, `recursive` (default), or `semantic`. The
legacy `guideline_pdf=@…` field is still accepted for backward compatibility.

Response:
```json
{
  "success": true,
  "brand_id": "...",
  "brand_name": "Acme Corp",
  "extracted_rules": { ... },
  "documents": [
    {
      "doc_id": "uuid",
      "source_filename": "voice.md",
      "chunks_indexed": 14,
      "chunks_skipped_dedup": 2,
      "raw_s3_key": "brand-onboarding/.../raw/voice.md"
    }
  ],
  "assets_indexed": 3
}
```

**Asking a grounded question (Phase 3)**

```bash
curl -X POST http://localhost:5001/api/brand/<brand_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the minimum logo clear space?"}'
```

Happy-path response:
```json
{
  "success": true,
  "result": {
    "answer": "The logo must have at least 20 px of clear space on all sides [1].",
    "citations": [{ "idx": 1, "status": "supported", ... }],
    "confidence": { "composite": 0.84, "retrieval": 0.93, "citation_coverage": 1.0, "completeness": 0.7 },
    "unsupported_claims": [],
    "is_idk": false,
    "retrieval_debug": { "latency_ms": 412.7, "used_strategies": ["dense","sparse","rerank"], "chunk_count": 5 }
  }
}
```

When the gate fires (composite < 0.4 or retrieval < 0.3, or the generator
itself refuses), the result switches to a structured IDK payload:

```json
{
  "answer": null,
  "is_idk": true,
  "idk_reason": "low_composite_confidence",
  "found": [{ "idx": 1, "source_filename": "voice.md", "section": "Brand Voice", "page": 1, "preview": "...", "rerank_score": 0.82 }],
  "missing": "The retrieved context did not adequately support a cited answer to the question.",
  "suggested_documents": [{ "source_filename": "voice.md", ... }]
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
| `OPENROUTER_API_KEY` | Yes | Powers `HybridToneAnalyzer`, `BrandComplianceJudge`, and Phase 3 grounded RAG (`/ask`) |
| `OPENROUTER_MODEL` | Optional | Override default chat model (e.g. `openai/gpt-4o-mini`) |
| `INTERNAL_WEBHOOK_SECRET` | Yes (matches br-be) | Signs callback POSTs |
| `MONGODB_URI` | Yes | Brand profile + document metadata persistence |
| `QDRANT_URL` | Yes | Hybrid (dense + BM25) vector store |
| `S3_BUCKET_NAME` | Optional | Raw + processed brand-document persistence (Phase 1) |
| `AWS_REGION` | Optional | Same as br-be S3 region |
| `RAG_DISABLE_HYBRID` | Optional | Set `1` to force dense-only retrieval |

---

## Brand Profiles

When a `brand_id` is supplied to `/api/analyze`, the pipeline enriches analysis with brand-specific context:

1. **Multi-format ingestion (Phase 1)** — PDF / Markdown / HTML / TXT files go through pluggable loaders → chunkers → within-brand dedup, then are written to a Qdrant hybrid collection (dense E5 + BM25 sparse) and a MongoDB `documents[]` ledger. Raw bytes are persisted to S3 so re-indexing never requires re-upload.
2. **Hybrid retrieval (Phase 2)** — `text_rag.retrieve_hybrid` runs dense + sparse queries in parallel, fuses with Reciprocal Rank Fusion (`0.7 dense + 0.3 sparse`), then cross-encoder reranks the candidate pool. Sparse failures degrade gracefully to dense-only with `fallback_reason="sparse_unavailable"`.
3. **Grounded Q&A (Phase 3)** — `POST /api/brand/{brand_id}/ask` produces cited answers (`[N]` chunk references), runs LLM-as-judge citation verification, scores composite confidence, and returns a structured "I don't know" payload when the gate fires.
4. **Structured rules** — `PDFRuleExtractor` still runs on the first PDF to populate the legacy `extracted_rules` block.
5. **Asset RAG** — Approved and rejected example images are embedded in Qdrant. Visual similarity against known rejects flags potential violations.
6. **LLM judge** — `BrandComplianceJudge` synthesizes retrieved rules + analyzer outputs via OpenRouter to produce a final verdict with inline `[N]` citations against the numbered brand rules.

Without a `brand_id`, analyzers use the static config defaults in `configs/`.

---

## Brand Knowledge RAG (Phases 1–3)

The brand-knowledge RAG stack is split into three phases that share one Qdrant
collection (`brand_{brand_id}_guidelines`) and one MongoDB document ledger.
Phase 1 owns the **write path**, Phase 2 + 3 own the **read path**.

### Top-level: two flows share one storage layer

```
           ┌─────────────────────────────────────────────────────────┐
           │                   SHARED STORAGE                        │
           │  ┌──────────┐   ┌──────────────┐   ┌───────────────┐    │
           │  │   S3     │   │   MongoDB    │   │   Qdrant      │    │
           │  │ raw docs │   │ brand_profile│   │ hybrid coll.  │    │
           │  │          │   │ .documents[] │   │ dense + BM25  │    │
           │  └──────────┘   └──────────────┘   └───────────────┘    │
           └─────────▲─────────────▲──────────────────▲──────────────┘
                     │             │                  │
          ┌──────────┴─────┐       │      ┌───────────┴──────────┐
          │ PHASE 1: WRITE │       │      │ PHASE 2 + 3: READ    │
          │  (onboard)     │       │      │ (ask / retrieve)     │
          └────────────────┘       │      └──────────────────────┘
                                   │
                           metadata ↑↓
```

### Phase 1 — Ingestion (`POST /api/brand/onboard`)

```
br-fe setup page
  │  (docs[], strategy, overrides)
  ▼
br-be  ──relays multipart──▶  consolidated_pipeline  POST /api/brand/onboard
                                        │
                                        ▼
                ┌──────────── per-file loop ────────────┐
                │                                       │
                │  1. s3_client.upload_raw  ─────────▶  S3 (raw bytes)
                │                                       │
                │  2. DocumentLoader.load(file)         │
                │     ├── pdf_loader   (PyMuPDF)        │
                │     ├── markdown_loader (md-it)       │
                │     ├── html_loader  (bs4)            │
                │     └── text_loader  (passthrough)    │
                │         → RawDocument{plaintext,      │
                │             sections, pages}          │
                │                                       │
                │  3. s3_client.upload_processed ────▶  S3 (plaintext.json)
                │                                       │
                │  4. Chunker.split(doc, strategy)      │
                │     ├── fixed      (size+overlap)     │
                │     ├── recursive  (headings→seps)    │
                │     └── semantic   (sentence-sim)     │
                │         → [Chunk{text, section,       │
                │             page, chunk_index, …}]    │
                │                                       │
                │  5. Deduper.filter_new_chunks         │
                │     (cosine ≥ 0.95 vs brand index)    │
                │         → kept[], skipped[]           │
                │                                       │
                │  6. EmbeddingService                  │
                │     ├── embed_dense_passages (E5)     │
                │     └── embed_sparse_passages(BM25)   │
                │                                       │
                │  7. QdrantStore.upsert atomic ─────▶  Qdrant
                │     (dense + sparse in one call)      │  named vec "dense" (1024, COSINE)
                │                                       │  sparse vec "bm25"
                │                                       │  payload: text, section, page,
                │                                       │           chunk_index, source_filename,
                │                                       │           chunking_strategy, char_count,
                │                                       │           doc_id, brand_id
                │                                       │
                │  8. brand_store.add_document ──────▶  MongoDB
                │     ($push documents[])               │  doc_id, s3 keys,
                │                                       │  chunk_count_indexed,
                │                                       │  chunk_count_skipped_dedup, …
                └───────────────────────────────────────┘
                                        │
                                        ▼
                    structured-rule extraction (1st PDF only, unchanged)
                                        │
                                        ▼
                 response: {brand_id, documents: [{doc_id, chunks_indexed,
                                                   chunks_skipped_dedup, …}]}
```

### Phase 2 — Hybrid Retrieval (used by Phase 3 and `/retrieve`)

```
text_rag.retrieve_hybrid(brand_id, query, brand_override?)
        │
        ▼
  HybridRetriever.retrieve
        │
        ├── EmbeddingService.embed_dense_query(q)          ┐
        │   → Qdrant query_points(using="dense")           │  parallel
        │                                                  │
        ├── EmbeddingService.embed_sparse_query(q)         │
        │   → Qdrant query_points(using="bm25")            ┘
        │
        ├── Reciprocal Rank Fusion
        │     fused = 0.7 · dense_rrf + 0.3 · sparse_rrf
        │     → candidate pool (top N)
        │
        ├── Cross-encoder rerank (fine top_k over candidates)
        │     → rerank_score per chunk
        │
        └── fallback handling
              sparse missing → "dense-only"
              query failure   → fallback_reason populated
        │
        ▼
RetrievalResult {
  query, chunks: [Reranked{candidate, fused_score, rerank_score, ranks}],
  latency_ms, used_strategies: [dense, sparse, rerank],
  fallback_reason
}
```

### Phase 3 — Grounded Generation (`POST /api/brand/{brand_id}/ask`)

```
Client
  │ { question, brand_override? }
  ▼
app.py  ask_brand()
  ├── 404 if brand not found
  ├── 503 if pipeline not initialized
  ├── 400 if question blank
  ▼
GroundedRAGPipeline.answer(brand_id, question, brand_override)
  │
  ▼
 [1] text_rag.retrieve_hybrid  ────────►  Phase 2
  │
  │  chunks == [] ?  ──yes──► IDK (reason = fallback_reason || no_retrieved_chunks)
  │
  ▼
 [2] GroundedGenerator.generate(q, chunks)
  │     prompt: grounded_answer_v1.yaml  (numbered context [1]…[N])
  │     LLMClient.chat(json_object) → {"answer": "…[N]…"}
  │     CitationParser.parse(text) → claims[], cited_indices
  │     → RawAnswer{text, claims, said_idk, usage}
  │
  │  said_idk ?  ──yes──► IDK (reason = generator_said_idk)
  │
  ▼
 [3] CitationVerifier.verify(claims, chunks)   ← LLM-as-judge
  │     per claim: decide supported | unsupported | partial
  │     → VerifiedAnswer{text, claims, citations[{idx,status,…}]}
  │
  ▼
 [4] CompletenessJudge.judge(q, answer_text)   ← LLM-as-judge
  │     → completeness ∈ [0,1]
  │
  ▼
 [5] ConfidenceScorer.score(retrieval, verified, completeness)
  │     retrieval         = f(top rerank_score)
  │     citation_coverage = supported / total claims
  │     completeness      = judge output
  │     composite = 0.4·retrieval + 0.4·coverage + 0.2·completeness
  │     → ConfidenceBreakdown
  │
  ▼
 [6] IDK gate
       composite < 0.4  OR  retrieval < 0.3 ?
         │
         ├── yes ─► IDKResponder.synthesize
         │            found[]:        top retrieved chunks (preview, section, page, score)
         │            missing:        reason from LLM tail or templated
         │            suggested_documents: unique source filenames
         │          → GroundedResponse{is_idk:true, idk_reason, found, missing, …}
         │
         └── no  ─► GroundedResponse{
                      answer, citations, confidence,
                      unsupported_claims (claims whose cited chunk was judged unsupported),
                      is_idk:false, retrieval_debug
                    }
```

### Request life-cycle at a glance

```
User asks    ─────►  FE    ─────►  BE    ─────►  Pipeline /ask
                                                  │
                                                  ▼
                                            Phase 2 (retrieve)
                                                  │  Qdrant (dense + sparse)
                                                  ▼
                                            Phase 3 (generate → verify → judge → score)
                                                  │  OpenRouter LLM × 3
                                                  ▼
                                            IDK gate → GroundedResponse
                                                  │
User gets  ◄─────  FE  ◄─────  BE  ◄─────────────┘
  { answer + [N] citations + confidence }   OR   { is_idk, found, missing, suggested }
```

### Single-collection invariant

`brand_{brand_id}_guidelines` stores **one Qdrant point per chunk** carrying
**both** a 1024-d dense vector (`"dense"`) and a BM25 sparse vector (`"bm25"`).
Phase 1 writes both atomically; Phase 2 reads both in parallel and fuses.
That is why Phase 2 needed no re-ingest after Phase 1, and why
`fallback_reason="sparse_unavailable"` is a soft degradation rather than a
hard failure.

### Key knobs in `configs/rag.yaml`

| Section | Knob | Effect |
|---|---|---|
| `chunking.default_strategy` | `fixed` \| `recursive` \| `semantic` | Per-onboard default; overridable per request |
| `chunking.recursive.chunk_size` | int | Soft max chars per chunk |
| `embeddings.dense_model` | model id | E5 model used for both passages and queries |
| `embeddings.sparse_model` | model id | Qdrant-served BM25 |
| `dedup.cosine_threshold` | float | Within-brand dedup similarity cutoff (default `0.95`) |
| `retrieval.fusion_weights` | `{dense, sparse}` | RRF weights (default `0.7 / 0.3`) |
| `retrieval.rerank_top_k` | int | Cross-encoder rerank window |
| `confidence.thresholds.idk_composite` | float | IDK gate (default `0.4`) |
| `confidence.thresholds.idk_retrieval` | float | IDK gate (default `0.3`) |
| `verification.enabled` | bool | Toggle LLM-as-judge citation verification |

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
