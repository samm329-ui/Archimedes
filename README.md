# Reverse Motion Compiler

A deterministic, closed-loop system that converts motion-graphics MP4 videos into structured, renderable JSON templates.

```
video → analysis → JSON → render → compare → refine → validate
```

---

## Architecture

```
backend/
  main.py                    ← FastAPI app + full pipeline orchestrator
  schemas/__init__.py        ← All Pydantic schemas (canonical data contracts)
  pipeline/
    ingest.py                ← Video ingestion, metadata extraction, frame extraction
    scene_detection.py       ← Temporal scene segmentation (histogram + SSIM + flow)
    detection.py             ← Multi-signal element detection (contours + text heuristic)
    tracking.py              ← IoU + feature-based element tracking across frames
    motion_analysis.py       ← Trajectory extraction, curve fitting, motion hypotheses
    schema_builder.py        ← Assembles TemplateJSON from all pipeline outputs
    render_loop.py           ← Deterministic render + SSIM comparison + refinement loop
    validator.py             ← Structural validation, field completeness, confidence gates
  tests/
    conftest.py              ← Shared fixtures, synthetic video factory
    test_ingest.py           ← 21 tests
    test_detection.py        ← 16 tests
    test_tracking.py         ← 14 tests
    test_motion.py           ← 23 tests
    test_schema.py           ← 39 tests
    test_integration.py      ← 18 tests (full pipeline end-to-end)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Required Python:** 3.10+  
**NumPy constraint:** Must be `<2` for OpenCV compatibility.

---

## Run Tests

```bash
cd motion_compiler
python -m pytest                        # all 131 tests
python -m pytest backend/tests/test_ingest.py -v
python -m pytest backend/tests/test_integration.py -v
```

---

## Run Server

```bash
cd motion_compiler
python -m uvicorn backend.main:app --reload --port 8000
```

### API Endpoints

```
GET  /health                  → {"status": "ok"}
POST /analyze                 → upload MP4, returns PipelineResult
POST /analyze/path?path=...   → analyze local file path (dev/testing)
```

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@my_video.mp4" | python -m json.tool
```

---

## Pipeline Stages

| Stage | Module | Rule |
|---|---|---|
| 1. Ingest | `ingest.py` | Reject corrupt/missing files with StructuredError |
| 2. Scene detection | `scene_detection.py` | Never collapse dual hypotheses below confidence threshold |
| 3. Frame sampling | `scene_detection.py` | Adaptive: densify at motion spikes, always include boundaries |
| 4. Detection | `detection.py` | Multi-signal; never assign final type early |
| 5. Tracking | `tracking.py` | IoU + feature similarity; log all lost tracks |
| 6. Motion analysis | `motion_analysis.py` | Multiple hypotheses; store raw + fitted curves |
| 7. Schema building | `schema_builder.py` | Every field has confidence + provenance |
| 8. Render + compare | `render_loop.py` | SSIM-gated; max 3 refinement attempts |
| 9. Validation | `validator.py` | Structural gate: approved / needs_refinement / rejected |

---

## Output Schema

```json
{
  "schema_version": "1.0.0",
  "meta": { "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 300 },
  "canvas": { "width": 1920, "height": 1080, "origin": "top-left" },
  "scenes": [{ "scene_id": "scene_000", "start_frame": 0, "end_frame": 300 }],
  "elements": [{
    "id": "trk_abc123",
    "type": "text",
    "confidence": 0.74,
    "role_scores": [{ "role": "title", "score": 0.7 }],
    "layout": { "x_norm": 0.1, "y_norm": 0.08, "w_norm": 0.5, "h_norm": 0.09 },
    "motion": [{
      "primitive": "translateY",
      "easing": "easeOutCubic",
      "from_value": 300.0,
      "to_value": 0.0,
      "duration_frames": 18,
      "raw_curve": [0.0, 0.2, 0.5, 0.8, 1.0],
      "confidence": 0.81
    }],
    "alternatives": [{ "type": "shape", "confidence": 0.18 }],
    "provenance": { "source_module": "schema_builder", "method": "track_assembly" }
  }],
  "validation": { "status": "approved", "ssim_score": 0.912 },
  "errors": []
}
```

---

## Non-Negotiable Principles

- **No raw LLM-only video interpretation** — all analysis is numerical/structural
- **No early type finalization** — multiple TypeCandidates always maintained
- **No silent failure** — every error produces a `StructuredError` with `failure_mode`, `stage`, `recoverable`
- **No output approval without render validation** — SSIM comparison is mandatory
- **No uncertainty collapse** — ambiguous fields retain `alternatives` and `hypotheses`
- **No hidden heuristics** — all thresholds are named constants in each module
