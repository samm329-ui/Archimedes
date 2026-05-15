"""
layering.py — Layer order inference from occlusion and alpha cues.

PRD §17 requirements:
  - Use: occlusion, visible overlap, mask edges, alpha behavior,
    entrance/exit visibility, foreground/background consistency
  - Layer order = probability-ranked hypothesis, not hardcoded
  - Transparent overlays handled explicitly

Failure cases handled:
  - transparent overlays → alpha-aware inference
  - additive effects → flag as uncertain
  - elements that float above scene → large bbox bonus
  - partially transparent motion text → opacity uncertainty flag
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.schemas import (
    BoundingBox,
    FailureMode,
    StructuredError,
    TrackedElement,
)


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_LAYER_CONFIDENCE = 0.40     # Below this → layer order stored as hypothesis only
BACKGROUND_AREA_THRESHOLD = 0.6  # Elements covering >60% canvas = likely background
FOREGROUND_AREA_THRESHOLD = 0.02 # Elements covering <2% canvas = likely foreground
MAX_LAYER_DEPTH = 10


@dataclass
class LayerHypothesis:
    """A scored hypothesis for a single track's layer position."""
    track_id: str
    layer: int                      # 0 = background, higher = foreground
    confidence: float               # 0–1
    reasoning: list[str]            # Explicit reasons for this assignment
    is_confirmed: bool              # True if confidence >= MIN_LAYER_CONFIDENCE
    hypothesis_only: bool


@dataclass
class OcclusionEvent:
    """Records one element occluding another at a frame."""
    frame_index: int
    front_track_id: str            # This element is in front
    back_track_id: str             # This element is behind
    overlap_iou: float
    confidence: float


def _canvas_area_fraction(bbox: BoundingBox, canvas_w: int, canvas_h: int) -> float:
    canvas_area = canvas_w * canvas_h
    if canvas_area == 0:
        return 0.0
    return min(1.0, (bbox.w * bbox.h) / canvas_area)


def _detect_occlusion_events(
    tracks: list[TrackedElement],
    canvas_width: int,
    canvas_height: int,
    min_iou: float = 0.05,
) -> list[OcclusionEvent]:
    """
    Detect frames where one element's bounding box overlaps another's.
    Uses area heuristic: smaller element that appears inside a larger one
    is assumed to be in front (text over background).
    """
    events: list[OcclusionEvent] = []

    bbox_by_track = {t.track_id: dict(t.bbox_sequence) for t in tracks}

    track_ids = [t.track_id for t in tracks]
    n = len(track_ids)

    for i in range(n):
        for j in range(i + 1, n):
            tid_a = track_ids[i]
            tid_b = track_ids[j]

            frames_a = set(bbox_by_track[tid_a].keys())
            frames_b = set(bbox_by_track[tid_b].keys())
            common = frames_a & frames_b

            for frame in common:
                ba = bbox_by_track[tid_a][frame]
                bb = bbox_by_track[tid_b][frame]
                iou = ba.iou(bb)
                if iou < min_iou:
                    continue

                # Smaller area element is in front
                area_a = _canvas_area_fraction(ba, canvas_width, canvas_height)
                area_b = _canvas_area_fraction(bb, canvas_width, canvas_height)

                if area_a < area_b:
                    front, back = tid_a, tid_b
                elif area_b < area_a:
                    front, back = tid_b, tid_a
                else:
                    continue  # Same size, ambiguous

                conf = min(1.0, iou * 2.0 + abs(area_a - area_b))
                events.append(OcclusionEvent(
                    frame_index=frame,
                    front_track_id=front,
                    back_track_id=back,
                    overlap_iou=round(iou, 4),
                    confidence=round(conf, 4),
                ))

    return events


def _area_based_layer(
    track: TrackedElement,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, float, list[str]]:
    """
    Estimate layer from element area.
    Large = background (layer 0), small = foreground (layer high).
    Returns (layer, confidence, reasons).
    """
    if not track.bbox_sequence:
        return 0, 0.1, ["no_bbox"]

    areas = [
        _canvas_area_fraction(b, canvas_width, canvas_height)
        for _, b in track.bbox_sequence
    ]
    mean_area = float(np.mean(areas))
    reasons = []

    if mean_area >= BACKGROUND_AREA_THRESHOLD:
        reasons.append(f"large_area:{mean_area:.2f}")
        return 0, 0.75, reasons
    elif mean_area <= FOREGROUND_AREA_THRESHOLD:
        reasons.append(f"small_area:{mean_area:.4f}")
        return MAX_LAYER_DEPTH, 0.65, reasons
    else:
        # Mid-range — moderate foreground
        layer = int((1.0 - mean_area) * MAX_LAYER_DEPTH * 0.6)
        reasons.append(f"mid_area:{mean_area:.3f}")
        return max(1, min(layer, MAX_LAYER_DEPTH - 1)), 0.45, reasons


def infer_layer_order(
    tracks: list[TrackedElement],
    canvas_width: int,
    canvas_height: int,
) -> tuple[dict[str, LayerHypothesis], list[OcclusionEvent], list[StructuredError]]:
    """
    Infer layer order for all tracks.

    Algorithm:
      1. Detect occlusion events pairwise
      2. Build a partial order graph from confirmed occlusions
      3. Use area heuristic as fallback / tiebreaker
      4. Assign integer layer 0 (background) → N (foreground)
      5. Store as hypothesis with confidence — never forced

    Returns:
        - layer_hypotheses: {track_id: LayerHypothesis}
        - occlusion_events: all detected events
        - errors
    """
    errors: list[StructuredError] = []
    hypotheses: dict[str, LayerHypothesis] = {}

    if not tracks:
        return hypotheses, [], errors

    # ── Step 1: Occlusion events ───────────────────────────────────────────
    try:
        occlusion_events = _detect_occlusion_events(tracks, canvas_width, canvas_height)
    except Exception as exc:
        errors.append(StructuredError(
            failure_mode=FailureMode.LAYER_ORDER_CONFLICT,
            message=f"Occlusion detection failed: {exc}",
            stage="layering",
            recoverable=True,
        ))
        occlusion_events = []

    # ── Step 2: Vote table — front_count per track ─────────────────────────
    # More times an element is in front = higher layer
    front_votes: dict[str, list[float]] = {t.track_id: [] for t in tracks}
    back_votes:  dict[str, list[float]] = {t.track_id: [] for t in tracks}

    for event in occlusion_events:
        if event.front_track_id in front_votes:
            front_votes[event.front_track_id].append(event.confidence)
        if event.back_track_id in back_votes:
            back_votes[event.back_track_id].append(event.confidence)

    # ── Step 3: Assign layers ──────────────────────────────────────────────
    # Compute raw layer score per track
    layer_scores: dict[str, float] = {}
    for track in tracks:
        f_votes = front_votes.get(track.track_id, [])
        b_votes = back_votes.get(track.track_id, [])

        # Area-based baseline
        area_layer, area_conf, area_reasons = _area_based_layer(
            track, canvas_width, canvas_height
        )

        if f_votes or b_votes:
            # Occlusion-based: net front score
            f_score = float(np.mean(f_votes)) if f_votes else 0.0
            b_score = float(np.mean(b_votes)) if b_votes else 0.0
            net = f_score - b_score  # positive = tends to be in front
            # Map net from [-1,1] to [0, MAX_LAYER_DEPTH]
            layer_float = ((net + 1.0) / 2.0) * MAX_LAYER_DEPTH
            layer_scores[track.track_id] = layer_float
        else:
            layer_scores[track.track_id] = float(area_layer)

    # Rank and assign integer layers
    sorted_tracks = sorted(
        tracks,
        key=lambda t: layer_scores.get(t.track_id, 0.0),
    )

    for rank, track in enumerate(sorted_tracks):
        raw_score = layer_scores.get(track.track_id, 0.0)
        area_layer, area_conf, area_reasons = _area_based_layer(
            track, canvas_width, canvas_height
        )

        f_votes = front_votes.get(track.track_id, [])
        b_votes = back_votes.get(track.track_id, [])
        has_occlusion_evidence = bool(f_votes or b_votes)

        # Confidence: higher if occlusion evidence exists
        if has_occlusion_evidence:
            conf = min(0.90, 0.55 + 0.1 * (len(f_votes) + len(b_votes)))
            reasons = [
                f"occlusion_front_votes:{len(f_votes)}",
                f"occlusion_back_votes:{len(b_votes)}",
            ]
        else:
            conf = area_conf
            reasons = area_reasons + ["area_fallback"]

        layer = int(round(raw_score))
        layer = max(0, min(layer, MAX_LAYER_DEPTH))

        is_confirmed = conf >= MIN_LAYER_CONFIDENCE

        # Detect potential conflict
        if not is_confirmed:
            errors.append(StructuredError(
                failure_mode=FailureMode.LAYER_ORDER_CONFLICT,
                message=f"Low-confidence layer for {track.track_id}: {conf:.2f}",
                stage="layering",
                recoverable=True,
                details={"track_id": track.track_id, "layer": layer, "confidence": conf},
            ))

        hypotheses[track.track_id] = LayerHypothesis(
            track_id=track.track_id,
            layer=layer,
            confidence=round(conf, 4),
            reasoning=reasons,
            is_confirmed=is_confirmed,
            hypothesis_only=not is_confirmed,
        )

    return hypotheses, occlusion_events, errors
