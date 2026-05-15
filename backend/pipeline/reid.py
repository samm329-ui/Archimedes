"""
reid.py — Element re-identification across occlusion and disappearance.

PRD §15: Re-ID must use context, not just appearance.
The system must reconnect partial occlusion, handle temporary disappearance,
and score re-identification confidence explicitly.

Re-ID uses:
  - Appearance embedding (color histogram similarity)
  - Spatial proximity (distance from predicted position)
  - Size consistency (area ratio)
  - Temporal gap penalty (longer gap → lower confidence)
  - Type candidate overlap (same type family → bonus)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.pipeline.feature_extraction import feature_distance, histogram_similarity
from backend.schemas import BoundingBox, FailureMode, StructuredError, TypeCandidate


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_REID_FRAME_GAP = 30          # Max frames to attempt re-ID
MIN_REID_CONFIDENCE = 0.35       # Below this — do not re-identify
SPATIAL_PROXIMITY_WEIGHT = 0.30
APPEARANCE_WEIGHT = 0.45
SIZE_WEIGHT = 0.15
TYPE_BONUS = 0.10


@dataclass
class ReidCandidate:
    """A hypothesis that a lost track matches a new detection."""
    lost_track_id: str
    new_detection_id: str
    reid_confidence: float       # 0–1
    spatial_score: float
    appearance_score: float
    size_score: float
    type_bonus: float
    frame_gap: int


@dataclass
class LostTrackRecord:
    """Remembers a track that disappeared, for possible re-ID."""
    track_id: str
    last_frame: int
    last_bbox: BoundingBox
    last_features: dict
    last_velocity: tuple[float, float]   # (vx, vy) pixels/frame
    type_candidates: list[TypeCandidate]


def _predict_position(record: LostTrackRecord, current_frame: int) -> BoundingBox:
    """
    Predict where a lost track would be using last known velocity.
    """
    gap = current_frame - record.last_frame
    vx, vy = record.last_velocity
    pred_x = record.last_bbox.x + vx * gap
    pred_y = record.last_bbox.y + vy * gap
    return BoundingBox(
        x=pred_x,
        y=pred_y,
        w=record.last_bbox.w,
        h=record.last_bbox.h,
    )


def _spatial_score(
    predicted: BoundingBox,
    candidate: BoundingBox,
    canvas_diagonal: float,
) -> float:
    """
    Spatial proximity score. 1 = overlap, 0 = very far away.
    Based on distance between centers relative to canvas diagonal.
    """
    px = predicted.x + predicted.w / 2
    py = predicted.y + predicted.h / 2
    cx = candidate.x + candidate.w / 2
    cy = candidate.y + candidate.h / 2
    dist = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    # Normalize: within one bbox width = 1.0, at canvas diagonal = 0.0
    norm_dist = dist / max(canvas_diagonal, 1.0)
    return float(max(0.0, 1.0 - norm_dist * 3.0))


def _size_score(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Score based on area ratio. 1 = same size, 0 = 10x different."""
    a1 = bbox1.area()
    a2 = bbox2.area()
    if a1 == 0 or a2 == 0:
        return 0.0
    ratio = min(a1, a2) / max(a1, a2)
    return float(ratio)


def _type_bonus(
    types1: list[TypeCandidate],
    types2: list[TypeCandidate],
) -> float:
    """Bonus if dominant types overlap."""
    t1 = {tc.type for tc in types1 if tc.confidence > 0.3}
    t2 = {tc.type for tc in types2 if tc.confidence > 0.3}
    return TYPE_BONUS if bool(t1 & t2) else 0.0


def _temporal_penalty(frame_gap: int) -> float:
    """Confidence penalty that increases with frame gap. 0=no gap, 1=max gap."""
    return min(1.0, frame_gap / MAX_REID_FRAME_GAP)


class ReidentificationEngine:
    """
    Maintains a registry of lost tracks and attempts re-identification
    when new detections appear.

    Usage:
        reid = ReidentificationEngine(canvas_w, canvas_h)
        reid.register_lost(track_id, frame, bbox, features, velocity, type_candidates)
        candidates = reid.attempt_reid(frame_index, new_detections)
    """

    def __init__(self, canvas_width: int = 1920, canvas_height: int = 1080):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self._canvas_diagonal = float(
            np.sqrt(canvas_width ** 2 + canvas_height ** 2)
        )
        self._lost_tracks: dict[str, LostTrackRecord] = {}

    def register_lost(
        self,
        track_id: str,
        last_frame: int,
        last_bbox: BoundingBox,
        last_features: dict,
        velocity: tuple[float, float],
        type_candidates: list[TypeCandidate],
    ) -> None:
        self._lost_tracks[track_id] = LostTrackRecord(
            track_id=track_id,
            last_frame=last_frame,
            last_bbox=last_bbox,
            last_features=last_features,
            last_velocity=velocity,
            type_candidates=type_candidates,
        )

    def attempt_reid(
        self,
        current_frame: int,
        detections: list,          # list[DetectedElement]
    ) -> tuple[list[ReidCandidate], list[StructuredError]]:
        """
        Try to match lost tracks to new detections.

        Returns:
          - list of ReidCandidates sorted by confidence descending
          - list of StructuredErrors
        """
        errors: list[StructuredError] = []
        candidates: list[ReidCandidate] = []

        # Expire old lost tracks
        expired = [
            tid for tid, rec in self._lost_tracks.items()
            if (current_frame - rec.last_frame) > MAX_REID_FRAME_GAP
        ]
        for tid in expired:
            del self._lost_tracks[tid]

        if not self._lost_tracks or not detections:
            return candidates, errors

        for lost_id, record in self._lost_tracks.items():
            frame_gap = current_frame - record.last_frame
            if frame_gap <= 0:
                continue

            predicted = _predict_position(record, current_frame)

            for det in detections:
                # Spatial
                sp_score = _spatial_score(
                    predicted, det.bbox, self._canvas_diagonal
                )

                # Appearance
                app_score = 1.0 - feature_distance(
                    record.last_features, det.features
                )
                app_score = max(0.0, app_score)

                # Size
                sz_score = _size_score(record.last_bbox, det.bbox)

                # Type bonus
                tb = _type_bonus(record.type_candidates, det.type_candidates)

                # Temporal penalty
                temp_penalty = _temporal_penalty(frame_gap)

                # Composite confidence
                raw_conf = (
                    SPATIAL_PROXIMITY_WEIGHT * sp_score
                    + APPEARANCE_WEIGHT * app_score
                    + SIZE_WEIGHT * sz_score
                    + tb
                )
                conf = raw_conf * (1.0 - temp_penalty * 0.5)
                conf = max(0.0, min(1.0, conf))

                if conf >= MIN_REID_CONFIDENCE:
                    candidates.append(ReidCandidate(
                        lost_track_id=lost_id,
                        new_detection_id=det.id,
                        reid_confidence=round(conf, 4),
                        spatial_score=round(sp_score, 4),
                        appearance_score=round(app_score, 4),
                        size_score=round(sz_score, 4),
                        type_bonus=tb,
                        frame_gap=frame_gap,
                    ))

        # Sort by confidence descending
        candidates.sort(key=lambda c: -c.reid_confidence)
        return candidates, errors

    def confirm_reid(self, lost_track_id: str) -> None:
        """Remove a track from the lost registry once re-identified."""
        self._lost_tracks.pop(lost_track_id, None)

    def get_lost_count(self) -> int:
        return len(self._lost_tracks)
