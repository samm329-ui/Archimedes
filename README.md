# Reverse Motion Compiler — v2.0.0

Deterministic, closed-loop system: **MP4 → motion analysis → JSON → render → compare → validate**

```
video → ingest → metadata → scene detection → adaptive sampling
      → detection → segmentation → feature extraction → tracking + re-ID
      → grouping → layering → motion analysis → curve fitting
      → hypothesis engine → role assignment → schema builder
      → render loop → 6-layer validation → approved JSON
```

---

## Complete File Structure

```
motion_compiler/
├── backend/
│   ├── main.py                        ← FastAPI + 18-stage orchestration
│   ├── schemas/
│   │   ├── __init__.py                ← Canonical Pydantic schemas
│   │   ├── input_schema.py            ← Input contract validation
│   │   ├── element_schema.py          ← Full element schema
│   │   ├── scene_schema.py            ← Scene analysis schema
│   │   ├── motion_schema.py           ← Motion axis + curve fit schemas
│   │   └── validation_schema.py       ← Six-layer validation schema
│   ├── pipeline/
│   │   ├── ingest.py                  ← Video ingestion, frame extraction
│   │   ├── metadata.py                ← Dedicated metadata + ffprobe enrichment
│   │   ├── scene_detection.py         ← Histogram + SSIM + optical flow
│   │   ├── frame_sampling.py          ← Adaptive sampling with motion spike bursts
│   │   ├── detection.py               ← Multi-signal element detection
│   │   ├── segmentation.py            ← GrabCut + watershed mask generation
│   │   ├── feature_extraction.py      ← Color, edge, texture, spatial features
│   │   ├── tracking.py                ← IoU + feature tracking
│   │   ├── reid.py                    ← Re-identification across occlusion
│   │   ├── grouping.py                ← Multi-signal scored grouping (PRD §16)
│   │   ├── layering.py                ← Occlusion-based layer inference (PRD §17)
│   │   ├── motion_analysis.py         ← Trajectory extraction
│   │   ├── curve_fitting.py           ← 8-easing curve fitting (PRD §19)
│   │   ├── hypothesis_engine.py       ← Multi-hypothesis management (PRD §30)
│   │   ├── role_assignment.py         ← Score-based role scoring (PRD §20)
│   │   ├── schema_builder.py          ← TemplateJSON assembly
│   │   ├── render_loop.py             ← Deterministic render + SSIM compare
│   │   ├── validator.py               ← Structural validation
│   │   ├── diagnostics.py             ← Stage timing + confidence profiling
│   │   ├── error_handler.py           ← Centralized error classification
│   │   └── provenance.py              ← Derivation chain tracking
│   └── tests/
│       ├── conftest.py                ← Shared fixtures + synthetic video factory
│       ├── test_ingest.py             ← 21 tests
│       ├── test_metadata.py           ← 13 tests
│       ├── test_detection.py          ← 16 tests
│       ├── test_frame_sampling.py     ← 16 tests
│       ├── test_feature_extraction.py ← 16 tests
│       ├── test_tracking.py           ← 14 tests
│       ├── test_motion.py             ← 23 tests
│       ├── test_grouping_layering.py  ← 20 tests
│       ├── test_curve_fitting_hypothesis.py ← 42 tests
│       ├── test_role_diagnostics_error.py   ← 52 tests
│       ├── test_schema.py             ← 39 tests
│       └── test_integration.py        ← 18 tests (end-to-end)
│
└── frontend/                          ← Next.js 14 inspection console
    ├── src/
    │   ├── types/pipeline.ts          ← TypeScript types (mirrors backend schemas)
    │   ├── lib/api.ts                 ← Backend API client
    │   ├── lib/colors.ts              ← Shared color utilities
    │   ├── app/
    │   │   ├── page.tsx               ← Root → redirect to /upload
    │   │   ├── layout.tsx             ← Root layout
    │   │   ├── globals.css
    │   │   ├── upload/page.tsx        ← Upload + stage progress
    │   │   ├── result/page.tsx        ← Result summary + download
    │   │   └── inspection/page.tsx    ← Full inspection console
    │   └── components/
    │       ├── FrameViewer.tsx        ← Canvas frame preview + detection overlay
    │       ├── Timeline.tsx           ← Timeline scrubber with element tracks
    │       ├── JSONViewer.tsx         ← Syntax-highlighted JSON viewer
    │       ├── ElementInspector.tsx   ← Per-element detail + motion curves
    │       ├── SceneInspector.tsx     ← Scene list + dual-hypothesis display
    │       ├── ValidationPanel.tsx    ← Six-layer validation display
    │       └── ConfidencePanel.tsx    ← Stage confidence + timing profiler
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.ts
    ├── postcss.config.js
    └── tsconfig.json
```

---

## Setup

### Backend

```bash
cd motion_compiler
pip install -r requirements.txt
```

**Python 3.10+ required. NumPy must be `<2` for OpenCV compatibility.**

### Frontend

```bash
cd motion_compiler/frontend
npm install
```

---

## Run

### Backend API

```bash
cd motion_compiler
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd motion_compiler/frontend
npm run dev
# Open http://localhost:3000
```

---

## Tests

```bash
cd motion_compiler
python -m pytest                              # 301 tests
python -m pytest -v --tb=short               # verbose
python -m pytest backend/tests/test_integration.py  # end-to-end only
```

---

## API

```
GET  /health                 → {"status": "ok", "version": "2.0.0"}
POST /analyze                → upload MP4 → PipelineResult
POST /analyze/path?path=...  → local file path (dev)
```

### PipelineResult shape

```json
{
  "status": "approved | needs_refinement | rejected",
  "template": { TemplateJSON },
  "errors": [ StructuredError ],
  "quality": { "element_count": 12, "mean_element_confidence": 0.71 },
  "diagnostics": { DiagnosticsReport },
  "validation_report": { FullValidationReport }
}
```

---

## 18-Stage Pipeline

| # | Stage | Module | PRD |
|---|-------|--------|-----|
| 1 | Ingest + validate | `ingest.py` | §11 |
| 2 | Metadata extraction | `metadata.py` | §11 |
| 3 | Scene detection | `scene_detection.py` | §12 |
| 4 | Adaptive frame sampling | `frame_sampling.py` | §13 |
| 5 | Frame extraction | `ingest.py` | §13 |
| 6 | Multi-signal detection | `detection.py` | §14 |
| 7 | Segmentation | `segmentation.py` | §9, §14 |
| 8 | Feature extraction | `feature_extraction.py` | §14 |
| 9 | Tracking + Re-ID | `tracking.py`, `reid.py` | §15 |
| 10 | Grouping | `grouping.py` | §16 |
| 11 | Layer order inference | `layering.py` | §17 |
| 12 | Motion analysis | `motion_analysis.py` | §18 |
| 13 | Curve fitting | `curve_fitting.py` | §19 |
| 14 | Hypothesis consolidation | `hypothesis_engine.py` | §19, §30 |
| 15 | Role assignment | `role_assignment.py` | §20 |
| 16 | Schema building | `schema_builder.py` | §21 |
| 17 | Render + compare | `render_loop.py` | §22 |
| 18 | Six-layer validation | `validator.py` + `main.py` | §23 |

---

## Non-Negotiable PRD Rules Enforced

- **No raw LLM-only video interpretation** — all numerical/structural
- **No early type finalization** — TypeCandidates always maintained
- **No silent failure** — every exception → typed `StructuredError`
- **No output approval without render validation** — SSIM mandatory
- **No uncertainty collapse** — `alternatives`, `hypothesis_only` retained
- **No hidden heuristics** — all thresholds are named constants
- **No missing provenance** — `ProvenanceRegistry` tracks every derivation
- **No single-pass pipeline** — closed-loop with max 3 refinements
