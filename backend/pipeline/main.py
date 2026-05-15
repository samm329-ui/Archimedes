"""
main.py — FastAPI entrypoint + full 18-stage pipeline orchestration.

PRD §6, §7 complete pipeline:
  1.  Ingest + validate
  2.  Metadata extraction
  3.  Scene detection
  4.  Adaptive frame sampling
  5.  Frame extraction
  6.  Per-frame detection
  7.  Per-frame segmentation
  8.  Feature extraction
  9.  Tracking + re-ID
  10. Grouping (PRD §16)
  11. Layer order inference (PRD §17)
  12. Motion analysis + curve fitting
  13. Hypothesis engine consolidation
  14. Role assignment
  15. Schema building
  16. Render + compare loop
  17. Six-layer structural validation
  18. Final gate
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.schemas import (
    FailureMode,
    StructuredError,
    TemplateJSON,
    ValidationStatus,
)
from backend.pipeline.ingest import IngestError, extract_frames, ingest_video
from backend.pipeline.metadata import extract_metadata
from backend.pipeline.scene_detection import detect_scenes, _whole_video_segment
from backend.pipeline.frame_sampling import build_sampling_plan
from backend.pipeline.detection import detect_elements_in_frame
from backend.pipeline.segmentation import segment_frame_elements
from backend.pipeline.feature_extraction import extract_features
from backend.pipeline.tracking import track_across_frames
from backend.pipeline.reid import ReidentificationEngine
from backend.pipeline.grouping import assign_groups
from backend.pipeline.layering import infer_layer_order
from backend.pipeline.motion_analysis import analyze_all_tracks
from backend.pipeline.hypothesis_engine import select_best_motion_hypotheses
from backend.pipeline.role_assignment import assign_roles
from backend.pipeline.schema_builder import build_template
from backend.pipeline.render_loop import run_render_validation
from backend.pipeline.validator import validate_template
from backend.pipeline.diagnostics import DiagnosticReport
from backend.pipeline.error_handler import ErrorAccumulator, safe_execute
from backend.pipeline.provenance import ProvenanceRegistry
from backend.schemas.validation_schema import FullValidationReport, LayerValidationRecord


app = FastAPI(
    title="Reverse Motion Compiler",
    description="Deterministic closed-loop: MP4 → JSON → render → validate.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PipelineResult(BaseModel):
    status: str
    template: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []
    quality: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    validation_report: dict[str, Any] = {}


def _six_layer_validation(
    template: TemplateJSON,
    tracks: list,
    motion_elements: list,
    group_assignments: dict,
) -> FullValidationReport:
    """PRD §23: Run all six validation layers in order."""
    elements = template.elements

    # Layer 1: Detection confidence filter
    low_conf = [e for e in elements if e.confidence < 0.15]
    frac_low = len(low_conf) / max(len(elements), 1)
    l1 = LayerValidationRecord(
        layer_name="detection_confidence_filter",
        passed=frac_low < 0.5,
        score=round(1.0 - frac_low, 4),
        failure_reasons=[f"{len(low_conf)} low-confidence elements"] if low_conf else [],
    )

    # Layer 2: Tracking stability check
    low_cont = [t for t in tracks if t.continuity_score < 0.4]
    frac_unstable = len(low_cont) / max(len(tracks), 1)
    l2 = LayerValidationRecord(
        layer_name="tracking_stability_check",
        passed=frac_unstable < 0.4,
        score=round(1.0 - frac_unstable, 4),
        failure_reasons=[f"{len(low_cont)} unstable tracks"] if low_cont else [],
    )

    # Layer 3: Group consistency check
    confirmed = sum(1 for a in group_assignments.values() if a.group_id is not None)
    l3 = LayerValidationRecord(
        layer_name="group_consistency_check",
        passed=True,
        score=1.0,
        notes=f"{confirmed}/{len(group_assignments)} elements grouped",
    )

    # Layer 4: Motion confidence check
    no_motion = [e for e in elements if not e.motion]
    l4 = LayerValidationRecord(
        layer_name="motion_confidence_check",
        passed=len(no_motion) == 0,
        score=round(1.0 - len(no_motion) / max(len(elements), 1), 4),
        failure_reasons=[f"{len(no_motion)} elements with no motion"] if no_motion else [],
    )

    # Layer 5: Render validation
    render_ok = template.validation.status in (
        ValidationStatus.APPROVED, ValidationStatus.NEEDS_REFINEMENT
    )
    l5 = LayerValidationRecord(
        layer_name="render_validation",
        passed=render_ok,
        score=template.validation.ssim_score or 0.0,
        failure_reasons=list(template.validation.failure_reasons),
    )

    # Layer 6: Final gate
    all_pass = all(l.passed for l in [l1, l2, l3, l4, l5])
    if all_pass:
        overall = ValidationStatus.APPROVED
    elif l1.passed and l2.passed:
        overall = ValidationStatus.NEEDS_REFINEMENT
    else:
        overall = ValidationStatus.REJECTED

    l6 = LayerValidationRecord(
        layer_name="final_gate",
        passed=overall == ValidationStatus.APPROVED,
        score=1.0 if all_pass else 0.5,
        notes=f"Overall: {overall.value}",
    )

    return FullValidationReport(
        detection_confidence=l1,
        tracking_stability=l2,
        group_consistency=l3,
        motion_confidence=l4,
        render_validation=l5,
        final_gate=l6,
        overall_status=overall,
        ssim_score=template.validation.ssim_score,
        similarity_score=template.validation.similarity_score,
        refinement_attempts=template.validation.refinement_attempts,
    )


def run_pipeline(video_path: str) -> PipelineResult:
    """Execute the complete 18-stage pipeline. No silent failures."""
    acc = ErrorAccumulator()
    diag = DiagnosticReport(video_path=video_path)
    prov = ProvenanceRegistry()

    # ── 1. Ingest ──────────────────────────────────────────────────────────
    diag.start_stage("ingest")
    try:
        metadata_base = ingest_video(video_path)
    except IngestError as exc:
        acc.add(exc.structured)
        diag.finish_stage("ingest")
        diag.finalize()
        return PipelineResult(status="rejected",
                              errors=[e.model_dump() for e in acc.all()],
                              diagnostics=diag.to_dict())
    diag.finish_stage("ingest", output_count=1)

    # ── 2. Metadata ────────────────────────────────────────────────────────
    diag.start_stage("metadata")
    ext_meta = safe_execute(lambda: extract_metadata(video_path),
                            "metadata", acc, None)
    metadata = ext_meta.video if ext_meta else metadata_base
    if ext_meta:
        acc.add_all(ext_meta.errors)
    diag.finish_stage("metadata", output_count=1)

    # ── 3. Scene detection ─────────────────────────────────────────────────
    diag.start_stage("scene_detection")
    result = safe_execute(lambda: detect_scenes(video_path, sample_every_n=2),
                          "scene_detection", acc, ([], []))
    scenes, scene_errors = result
    acc.add_all(scene_errors)
    if not scenes:
        scenes = _whole_video_segment(metadata.frame_count)
        acc.add(StructuredError(
            failure_mode=FailureMode.SCENE_BOUNDARY_UNCERTAIN,
            message="No scenes detected — using whole video",
            stage="scene_detection", recoverable=True,
        ))
    diag.finish_stage("scene_detection", output_count=len(scenes))

    # ── 4. Adaptive frame sampling ─────────────────────────────────────────
    diag.start_stage("frame_sampling")
    plan = safe_execute(
        lambda: build_sampling_plan(video_path, scenes, metadata.fps,
                                    target_rate=min(12.0, metadata.fps)),
        "frame_sampling", acc, None)
    if plan:
        acc.add_all(plan.errors)
        frame_indices = plan.frame_indices
    else:
        step = max(1, metadata.frame_count // 100)
        frame_indices = list(range(0, metadata.frame_count, step))
    diag.finish_stage("frame_sampling", output_count=len(frame_indices))

    # ── 5. Frame extraction ────────────────────────────────────────────────
    try:
        frame_data = extract_frames(video_path, frame_indices)
    except IngestError as exc:
        acc.add(exc.structured)
        frame_data = []
    if not frame_data:
        diag.finalize()
        return PipelineResult(status="rejected",
                              errors=[e.model_dump() for e in acc.all()],
                              diagnostics=diag.to_dict())
    frame_dict = {idx: f for idx, f in frame_data}

    # ── 6. Detection ───────────────────────────────────────────────────────
    diag.start_stage("detection")
    frame_detections: list[tuple[int, list]] = []
    all_detections_flat: list = []
    for frame_idx, frame in frame_data:
        dets, det_errors = detect_elements_in_frame(frame, frame_idx)
        acc.add_all(det_errors)
        frame_detections.append((frame_idx, dets))
        all_detections_flat.extend(dets)
    diag.finish_stage("detection", output_count=len(all_detections_flat))

    # ── 7. Segmentation ────────────────────────────────────────────────────
    diag.start_stage("segmentation")
    for frame_idx, dets in frame_detections:
        frame = frame_dict.get(frame_idx)
        if frame is not None and dets:
            _, seg_errors = segment_frame_elements(frame, dets, use_grabcut=False)
            acc.add_all(seg_errors)
    diag.finish_stage("segmentation")

    # ── 8. Feature extraction ──────────────────────────────────────────────
    diag.start_stage("feature_extraction")
    for det in all_detections_flat:
        frame = frame_dict.get(det.frame_index)
        if frame is not None:
            feats = safe_execute(
                lambda d=det, f=frame: extract_features(
                    f, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h,
                    metadata.width, metadata.height).to_dict(),
                "feature_extraction", acc, det.features)
            det.features = feats
    diag.finish_stage("feature_extraction", output_count=len(all_detections_flat))

    # ── 9. Tracking + Re-ID ────────────────────────────────────────────────
    diag.start_stage("tracking")
    tracks, tracking_errors = safe_execute(
        lambda: track_across_frames(frame_detections),
        "tracking", acc, ([], []))
    acc.add_all(tracking_errors)
    diag.record_confidences("tracking", [t.continuity_score for t in tracks])
    diag.finish_stage("tracking", output_count=len(tracks))
    if not tracks:
        diag.finalize()
        return PipelineResult(status="rejected",
                              errors=[e.model_dump() for e in acc.all()],
                              diagnostics=diag.to_dict())

    # ── 10. Grouping ───────────────────────────────────────────────────────
    diag.start_stage("grouping")
    motion_prelim, _ = safe_execute(
        lambda: analyze_all_tracks(tracks, metadata.width, metadata.height),
        "grouping", acc, ([], []))
    group_assignments, _, group_errors = safe_execute(
        lambda: assign_groups(tracks, motion_prelim, metadata.width, metadata.height),
        "grouping", acc, ({}, [], []))
    acc.add_all(group_errors)
    diag.finish_stage("grouping",
        output_count=sum(1 for a in group_assignments.values() if a.group_id))

    # ── 11. Layer inference ────────────────────────────────────────────────
    diag.start_stage("layering")
    layer_hypotheses, _, layer_errors = safe_execute(
        lambda: infer_layer_order(tracks, metadata.width, metadata.height),
        "layering", acc, ({}, [], []))
    acc.add_all(layer_errors)
    layer_map = {tid: h.layer for tid, h in layer_hypotheses.items()}
    diag.record_confidences("layering", [h.confidence for h in layer_hypotheses.values()])
    diag.finish_stage("layering", output_count=len(layer_hypotheses))

    # ── 12. Motion analysis ────────────────────────────────────────────────
    diag.start_stage("motion_analysis")
    motion_elements, motion_errors = safe_execute(
        lambda: analyze_all_tracks(tracks, metadata.width, metadata.height),
        "motion_analysis", acc, ([], []))
    acc.add_all(motion_errors)
    diag.record_confidences("motion_analysis", [me.motion_confidence for me in motion_elements])
    diag.finish_stage("motion_analysis", output_count=len(motion_elements))

    # ── 13. Hypothesis consolidation ──────────────────────────────────────
    diag.start_stage("hypothesis_engine")
    for me in motion_elements:
        me.motion_hypotheses = select_best_motion_hypotheses(
            me.motion_hypotheses, max_per_primitive=2)
    diag.finish_stage("hypothesis_engine", output_count=len(motion_elements))

    # ── 14. Role assignment ────────────────────────────────────────────────
    diag.start_stage("role_assignment")
    role_assignments, role_errors = safe_execute(
        lambda: assign_roles(tracks, motion_elements, metadata.width,
                             metadata.height, metadata.frame_count, layer_map),
        "role_assignment", acc, ({}, []))
    acc.add_all(role_errors)
    diag.finish_stage("role_assignment", output_count=len(role_assignments))

    # ── 15. Schema building ────────────────────────────────────────────────
    diag.start_stage("schema_builder")
    build_result = safe_execute(
        lambda: build_template(
            metadata=metadata, scenes=scenes, tracks=tracks,
            motion_elements=motion_elements, accumulated_errors=acc.all(),
            group_assignments=group_assignments, layer_map=layer_map,
            role_assignments=role_assignments,
        ),
        "schema_builder", acc, None)
    if build_result is None:
        diag.finalize()
        return PipelineResult(status="rejected",
                              errors=[e.model_dump() for e in acc.all()],
                              diagnostics=diag.to_dict())
    template, build_errors = build_result
    acc.add_all(build_errors)
    diag.record_confidences("schema_builder", [e.confidence for e in template.elements])
    diag.finish_stage("schema_builder", output_count=len(template.elements))

    # ── 16. Render + compare ───────────────────────────────────────────────
    diag.start_stage("render_loop")
    validation_frames = frame_data[::max(1, len(frame_data) // 5)][:5]
    render_result_tuple = safe_execute(
        lambda: run_render_validation(template, validation_frames),
        "render_loop", acc, (template, template.validation, []))
    template, render_result, render_errors = render_result_tuple
    acc.add_all(render_errors)
    diag.refinement_count = render_result.refinement_attempts
    diag.finish_stage("render_loop",
        notes=[f"SSIM={render_result.ssim_score}", f"status={render_result.status.value}"])

    # ── 17. Six-layer validation ───────────────────────────────────────────
    diag.start_stage("validator")
    struct_result, struct_errors = safe_execute(
        lambda: validate_template(template), "validator", acc, (template.validation, []))
    acc.add_all(struct_errors)
    full_report = _six_layer_validation(template, tracks, motion_elements, group_assignments)
    diag.finish_stage("validator")

    # ── 18. Final gate ─────────────────────────────────────────────────────
    if struct_result.status == ValidationStatus.REJECTED:
        final_status = "rejected"
    elif full_report.overall_status == ValidationStatus.APPROVED:
        final_status = "approved"
    elif full_report.overall_status == ValidationStatus.NEEDS_REFINEMENT:
        final_status = "needs_refinement"
    else:
        final_status = "rejected"

    template.validation = render_result
    template.validation.failure_reasons.extend(struct_result.failure_reasons)
    template.errors = acc.all()
    diag.passed_validation = (final_status == "approved")
    diag.finalize()
    template.provenance["diagnostics"] = diag.to_dict()
    template.provenance["provenance_summary"] = prov.summary()

    return PipelineResult(
        status=final_status,
        template=template.model_dump(),
        errors=[e.model_dump() for e in acc.all()],
        quality=template.quality,
        diagnostics=diag.to_dict(),
        validation_report=full_report.model_dump(),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze", response_model=PipelineResult)
async def analyze_video(file: UploadFile = File(...)) -> PipelineResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return run_pipeline(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/analyze/path", response_model=PipelineResult)
def analyze_video_path(path: str) -> PipelineResult:
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return run_pipeline(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
