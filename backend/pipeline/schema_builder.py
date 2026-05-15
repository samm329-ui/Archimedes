"""
schema_builder.py — Assembles the final TemplateJSON from all pipeline outputs.

Now consumes:
  - group_assignments (from grouping.py)
  - layer_map (from layering.py)
  - role_assignments (from role_assignment.py)

Rules (unchanged from PRD §21):
  - Every element must have explicit type, confidence, and provenance
  - Ambiguous fields retain alternatives
  - Never invent structure without evidence
  - Uncertainty stored, not collapsed
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.schemas import (
    ElementType,
    FailureMode,
    LayoutInfo,
    MotionElement,
    ProvenanceRecord,
    RoleScore,
    SceneSegment,
    StructuredError,
    StyleInfo,
    TemplateElement,
    TemplateJSON,
    TimingInfo,
    TrackedElement,
    TypeCandidate,
    ValidationResult,
    ValidationStatus,
    VideoMetadata,
)


def _dominant_type(candidates: list[TypeCandidate]) -> tuple[ElementType, float]:
    if not candidates:
        return ElementType.UNKNOWN, 0.0
    best = max(candidates, key=lambda c: c.confidence)
    return best.type, best.confidence


def _build_layout(
    track: TrackedElement,
    canvas_width: int,
    canvas_height: int,
    layer: int = 0,
    layer_confidence: float = 0.5,
) -> LayoutInfo:
    if not track.bbox_sequence:
        return LayoutInfo(
            x_norm=0,
            y_norm=0,
            w_norm=0,
            h_norm=0,
            layer=layer,
            layer_confidence=layer_confidence,
        )
    bboxes = [b for _, b in track.bbox_sequence]
    x = float(np.median([b.x for b in bboxes]))
    y = float(np.median([b.y for b in bboxes]))
    w = float(np.median([b.w for b in bboxes]))
    h = float(np.median([b.h for b in bboxes]))
    return LayoutInfo(
        x_norm=x / canvas_width,
        y_norm=y / canvas_height,
        w_norm=w / canvas_width,
        h_norm=h / canvas_height,
        layer=layer,
        layer_confidence=layer_confidence,
    )


def _build_style(track: TrackedElement) -> StyleInfo:
    return StyleInfo(
        dominant_color=None,
        has_glow=False,
        has_shadow=False,
        opacity=1.0,
        blur_radius=0.0,
    )


def _build_timing(track: TrackedElement, fps: float) -> TimingInfo:
    if not track.bbox_sequence:
        return TimingInfo(enter_frame=0, exit_frame=0, duration_frames=0, fps=fps)
    frames = [f for f, _ in track.bbox_sequence]
    enter = min(frames)
    exit_ = max(frames)
    return TimingInfo(
        enter_frame=enter, exit_frame=exit_, duration_frames=exit_ - enter, fps=fps
    )


def build_template(
    metadata: VideoMetadata,
    scenes: list[SceneSegment],
    tracks: list[TrackedElement],
    motion_elements: list[MotionElement],
    accumulated_errors: list[StructuredError],
    group_assignments: Optional[dict] = None,
    layer_map: Optional[dict[str, int]] = None,
    role_assignments: Optional[dict[str, list[RoleScore]]] = None,
) -> tuple[TemplateJSON, list[StructuredError]]:
    """
    Assemble all pipeline outputs into a TemplateJSON.
    Accepts optional group_assignments, layer_map, role_assignments from
    the dedicated modules (grouping.py, layering.py, role_assignment.py).
    """
    errors: list[StructuredError] = []
    motion_by_track = {me.track_id: me for me in motion_elements}
    canvas = {
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.fps,
        "origin": "top-left",
        "coordinate_system": "pixel",
    }

    template_elements: list[TemplateElement] = []

    for track in tracks:
        if not track.bbox_sequence:
            continue

        elem_type, elem_conf = _dominant_type(track.type_candidates)

        # Layer from layering module or default
        layer = (layer_map or {}).get(track.track_id, 0)
        layer_conf = 0.5
        if layer_map and track.track_id in layer_map:
            layer_conf = 0.7  # has explicit assignment

        layout = _build_layout(
            track, metadata.width, metadata.height, layer, layer_conf
        )
        style = _build_style(track)
        timing = _build_timing(track, metadata.fps)

        # Roles from role_assignment module or fallback
        roles: list[RoleScore] = []
        if role_assignments and track.track_id in role_assignments:
            roles = role_assignments[track.track_id]
        if not roles:
            roles = [RoleScore(role="accent", score=0.3)]

        # Motion
        motion_hyps = []
        motion_conf = 0.0
        if track.track_id in motion_by_track:
            me = motion_by_track[track.track_id]
            motion_hyps = me.motion_hypotheses
            motion_conf = me.motion_confidence

        # Group info
        group_id = None
        if group_assignments and track.track_id in group_assignments:
            ga = group_assignments[track.track_id]
            group_id = ga.group_id

        # Failure modes
        failure_modes = []
        if elem_conf < 0.5:
            failure_modes.append(FailureMode.UNSUPPORTED_CLASS)
        if motion_conf < 0.3 and motion_hyps:
            failure_modes.append(FailureMode.MOTION_AMBIGUOUS)

        validation_notes = []
        if track.continuity_score < 0.5:
            validation_notes.append(f"Low continuity: {track.continuity_score:.2f}")
        if elem_conf < 0.4:
            validation_notes.append(f"Ambiguous type @ {elem_conf:.2f}")

        alternatives = [
            tc
            for tc in track.type_candidates
            if tc.type != elem_type and tc.confidence > 0.1
        ]

        content = None
        if elem_type == ElementType.TEXT and track.bbox_sequence:
            content = track._feature_cache.get("text_content")

        template_elements.append(
            TemplateElement(
                id=track.track_id,
                type=elem_type,
                subtype=None,
                confidence=round(elem_conf, 4),
                role_scores=roles,
                content=content,
                layout=layout,
                style=style,
                motion=motion_hyps,
                timing=timing,
                group_id=group_id,
                layer=layer,
                alternatives=alternatives,
                provenance=ProvenanceRecord(
                    source_module="schema_builder",
                    source_frame_range=(timing.enter_frame, timing.exit_frame),
                    method="track_assembly",
                    confidence=round(elem_conf, 4),
                ),
                failure_modes=failure_modes,
                validation_notes=validation_notes,
            )
        )

    if not template_elements:
        errors.append(
            StructuredError(
                failure_mode=FailureMode.SCHEMA_BUILD_FAILED,
                message="No elements to assemble into template",
                stage="schema_builder",
                recoverable=True,
            )
        )

    quality = {
        "element_count": len(template_elements),
        "mean_element_confidence": (
            float(np.mean([e.confidence for e in template_elements]))
            if template_elements
            else 0.0
        ),
        "scene_count": len(scenes),
        "total_errors": len(accumulated_errors) + len(errors),
        "grouped_elements": sum(1 for e in template_elements if e.group_id),
    }

    provenance = {
        "pipeline_stages": [
            "ingest",
            "metadata",
            "scene_detection",
            "frame_sampling",
            "detection",
            "segmentation",
            "feature_extraction",
            "tracking",
            "reid",
            "grouping",
            "layering",
            "motion_analysis",
            "curve_fitting",
            "hypothesis_engine",
            "role_assignment",
            "schema_builder",
        ],
        "schema_version": "2.0.0",
    }

    template = TemplateJSON(
        schema_version="2.0.0",
        meta=metadata,
        quality=quality,
        canvas=canvas,
        scenes=scenes,
        elements=template_elements,
        camera={"motion": "static", "confidence": 0.5},
        validation=ValidationResult(status=ValidationStatus.PENDING),
        errors=accumulated_errors + errors,
        provenance=provenance,
    )

    return template, errors
