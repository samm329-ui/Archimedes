"""
role_assignment.py — Score-based element role assignment.

PRD §20 requirements:
  - Role assignment MUST remain scoring-based, not hardcoded
  - Roles stored as scores — multiple roles per element allowed
  - Roles may vary across time segments
  - Never force a single role when evidence is ambiguous

Roles supported (from PRD):
  main_subject, background, accent, title, subtitle, callout,
  icon, decorative, overlay, transition_element

Signals used:
  - Spatial position (top/center/bottom of canvas)
  - Element area (fraction of canvas)
  - Layer order
  - Element type (text vs shape vs overlay)
  - Motion characteristics (entering/exiting = transition)
  - Temporal presence (short = accent, long = background/main)
  - Proximity to canvas edges
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.schemas import (
    BoundingBox,
    ElementType,
    FailureMode,
    MotionElement,
    MotionPrimitive,
    RoleScore,
    StructuredError,
    TrackedElement,
    TypeCandidate,
)


# ── Role definitions ──────────────────────────────────────────────────────────

ALL_ROLES = [
    "main_subject",
    "background",
    "accent",
    "title",
    "subtitle",
    "callout",
    "icon",
    "decorative",
    "overlay",
    "transition_element",
]

# Thresholds (all explicit, none hidden)
BACKGROUND_AREA_MIN = 0.50
TITLE_AREA_MIN = 0.005
TITLE_AREA_MAX = 0.25
TITLE_Y_MAX_NORM = 0.40      # top 40% of canvas
SUBTITLE_Y_MIN_NORM = 0.35
SUBTITLE_Y_MAX_NORM = 0.80
ICON_AREA_MAX = 0.012
ACCENT_AREA_MAX = 0.03
MAIN_SUBJECT_AREA_MIN = 0.03
MAIN_SUBJECT_AREA_MAX = 0.60
MIN_SCORE_TO_REPORT = 0.10


@dataclass
class RoleEvidence:
    """All evidence used to score roles for one element."""
    area_norm: float
    cx_norm: float
    cy_norm: float
    layer: int
    dominant_type: ElementType
    has_enter_motion: bool
    has_exit_motion: bool
    temporal_fraction: float     # fraction of total video this element is present
    is_text: bool
    is_shape: bool
    is_overlay: bool
    aspect_ratio: float


def _build_evidence(
    track: TrackedElement,
    motion: Optional[MotionElement],
    canvas_width: int,
    canvas_height: int,
    total_frames: int,
    layer: int = 0,
) -> RoleEvidence:
    if not track.bbox_sequence:
        return RoleEvidence(
            area_norm=0.0, cx_norm=0.5, cy_norm=0.5,
            layer=layer, dominant_type=ElementType.UNKNOWN,
            has_enter_motion=False, has_exit_motion=False,
            temporal_fraction=0.0, is_text=False,
            is_shape=False, is_overlay=False, aspect_ratio=1.0,
        )

    bboxes = [b for _, b in track.bbox_sequence]
    areas = [b.w * b.h / max(canvas_width * canvas_height, 1) for b in bboxes]
    cx_vals = [(b.x + b.w / 2) / canvas_width for b in bboxes]
    cy_vals = [(b.y + b.h / 2) / canvas_height for b in bboxes]
    ar_vals = [b.w / max(b.h, 1.0) for b in bboxes]

    mean_area = float(np.median(areas))
    mean_cx = float(np.median(cx_vals))
    mean_cy = float(np.median(cy_vals))
    mean_ar = float(np.median(ar_vals))

    dom_type = ElementType.UNKNOWN
    if track.type_candidates:
        best = max(track.type_candidates, key=lambda tc: tc.confidence)
        dom_type = best.type

    # Motion entry/exit
    has_enter = False
    has_exit = False
    if motion and motion.motion_hypotheses:
        prims = {h.primitive for h in motion.motion_hypotheses if h.confidence > 0.2}
        has_enter = any(p in (
            MotionPrimitive.TRANSLATE_X, MotionPrimitive.TRANSLATE_Y,
            MotionPrimitive.SCALE, MotionPrimitive.OPACITY
        ) for p in prims)
        has_exit = has_enter  # Symmetrical assumption without full exit analysis

    # Temporal presence
    frames_present = len(set(f for f, _ in track.bbox_sequence))
    temporal_fraction = frames_present / max(total_frames, 1)

    return RoleEvidence(
        area_norm=mean_area,
        cx_norm=mean_cx,
        cy_norm=mean_cy,
        layer=layer,
        dominant_type=dom_type,
        has_enter_motion=has_enter,
        has_exit_motion=has_exit,
        temporal_fraction=temporal_fraction,
        is_text=(dom_type == ElementType.TEXT),
        is_shape=(dom_type == ElementType.SHAPE),
        is_overlay=(dom_type == ElementType.OVERLAY),
        aspect_ratio=mean_ar,
    )


def score_roles(ev: RoleEvidence) -> list[RoleScore]:
    """
    Score all roles for one element based on evidence.
    Returns list of RoleScores sorted descending.
    Never returns empty — at least one role always scored.
    """
    scores: dict[str, float] = {r: 0.0 for r in ALL_ROLES}

    # ── background ────────────────────────────────────────────────────────
    if ev.area_norm >= BACKGROUND_AREA_MIN:
        scores["background"] += 0.70
    if ev.layer == 0:
        scores["background"] += 0.20
    if ev.temporal_fraction > 0.8:
        scores["background"] += 0.10

    # ── title ─────────────────────────────────────────────────────────────
    if (ev.is_text and
        TITLE_AREA_MIN <= ev.area_norm <= TITLE_AREA_MAX and
        ev.cy_norm <= TITLE_Y_MAX_NORM):
        scores["title"] += 0.65
        if ev.has_enter_motion:
            scores["title"] += 0.15
        if ev.aspect_ratio > 2.5:   # wide text block
            scores["title"] += 0.10

    # ── subtitle ──────────────────────────────────────────────────────────
    if (ev.is_text and
        SUBTITLE_Y_MIN_NORM <= ev.cy_norm <= SUBTITLE_Y_MAX_NORM and
        ev.area_norm < TITLE_AREA_MAX):
        scores["subtitle"] += 0.55
        if ev.area_norm < 0.05:
            scores["subtitle"] += 0.10

    # ── main_subject ──────────────────────────────────────────────────────
    if (MAIN_SUBJECT_AREA_MIN <= ev.area_norm <= MAIN_SUBJECT_AREA_MAX and
        0.15 <= ev.cx_norm <= 0.85 and
        0.10 <= ev.cy_norm <= 0.90):
        scores["main_subject"] += 0.50
        if ev.layer >= 3:
            scores["main_subject"] += 0.15
        if not ev.is_text:
            scores["main_subject"] += 0.10

    # ── callout ───────────────────────────────────────────────────────────
    if (ev.is_text and
        ev.area_norm < 0.06 and
        ev.has_enter_motion and
        0.2 <= ev.cx_norm <= 0.8):
        scores["callout"] += 0.55

    # ── icon ──────────────────────────────────────────────────────────────
    if (ev.area_norm <= ICON_AREA_MAX and
        0.85 <= ev.aspect_ratio <= 1.15):   # roughly square
        scores["icon"] += 0.55
        if ev.layer >= 4:
            scores["icon"] += 0.10

    # ── accent ────────────────────────────────────────────────────────────
    if ev.area_norm <= ACCENT_AREA_MAX and not ev.is_text:
        scores["accent"] += 0.45
        if ev.temporal_fraction < 0.5:
            scores["accent"] += 0.15

    # ── decorative ────────────────────────────────────────────────────────
    if (ev.is_shape and
        ev.area_norm < 0.05 and
        not ev.has_enter_motion and
        ev.temporal_fraction > 0.5):
        scores["decorative"] += 0.50

    # ── overlay ───────────────────────────────────────────────────────────
    if ev.is_overlay or (ev.area_norm > 0.2 and ev.layer >= 5):
        scores["overlay"] += 0.55

    # ── transition_element ────────────────────────────────────────────────
    if (ev.has_enter_motion and
        ev.has_exit_motion and
        ev.temporal_fraction < 0.3):
        scores["transition_element"] += 0.60

    # Normalize so no score exceeds 1.0
    for role in scores:
        scores[role] = min(1.0, scores[role])

    # Build result list
    result = [
        RoleScore(role=role, score=round(score, 4))
        for role, score in scores.items()
        if score >= MIN_SCORE_TO_REPORT
    ]
    result.sort(key=lambda rs: -rs.score)

    # Guarantee at least one role
    if not result:
        result = [RoleScore(role="accent", score=0.20)]

    return result


def assign_roles(
    tracks: list[TrackedElement],
    motion_elements: list[MotionElement],
    canvas_width: int,
    canvas_height: int,
    total_frames: int,
    layer_map: Optional[dict[str, int]] = None,
) -> tuple[dict[str, list[RoleScore]], list[StructuredError]]:
    """
    Assign role scores to all tracks.

    Returns:
        - role_assignments: {track_id: list[RoleScore]}
        - errors
    """
    errors: list[StructuredError] = []
    motion_by_track = {me.track_id: me for me in motion_elements}
    role_assignments: dict[str, list[RoleScore]] = {}

    for track in tracks:
        layer = (layer_map or {}).get(track.track_id, 0)
        motion = motion_by_track.get(track.track_id)

        try:
            evidence = _build_evidence(
                track, motion,
                canvas_width, canvas_height,
                total_frames, layer,
            )
            roles = score_roles(evidence)
        except Exception as exc:
            errors.append(StructuredError(
                failure_mode=FailureMode.SCHEMA_BUILD_FAILED,
                message=f"Role assignment failed for {track.track_id}: {exc}",
                stage="role_assignment",
                recoverable=True,
                details={"track_id": track.track_id},
            ))
            roles = [RoleScore(role="accent", score=0.20)]

        role_assignments[track.track_id] = roles

    return role_assignments, errors
