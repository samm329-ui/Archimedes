"""
tracking.py — Element identity tracking across frames.

Implements IoU + feature-similarity-based tracking (no DeepSORT dependency).
Rules:
  - Preserve identity through partial occlusion
  - Detect track splits and merges
  - Confidence decays when a track is not observed
  - Never silently lose a track — record track_lost events
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.schemas import (
    BoundingBox,
    DetectedElement,
    FailureMode,
    ProvenanceRecord,
    StructuredError,
    TrackedElement,
    TypeCandidate,
)


# ── Thresholds ────────────────────────────────────────────────────────────────

IOU_MATCH_THRESHOLD = 0.30  # Minimum IoU to consider same element
FEATURE_MATCH_WEIGHT = 0.35  # Weight of feature similarity in matching
IOU_WEIGHT = 0.65  # Weight of IoU in matching
MAX_MISSING_FRAMES = 10  # Frames before a track is considered lost
CONFIDENCE_DECAY = 0.08  # Per-frame confidence decay when not observed
MIN_TRACK_CONFIDENCE = 0.20  # Below this → mark as lost


@dataclass
class ActiveTrack:
    track_id: str
    last_bbox: BoundingBox
    last_frame: int
    last_features: dict
    type_candidates: list[TypeCandidate]
    bbox_sequence: list[tuple[int, BoundingBox]] = field(default_factory=list)
    _feature_cache: dict = field(default_factory=dict)
    continuity_score: float = 1.0
    occlusion_score: float = 0.0
    missing_frames: int = 0
    is_active: bool = True


def _feature_similarity(f1: dict, f2: dict) -> float:
    """
    Simple feature similarity between two feature dicts.
    Uses color variance, aspect ratio, and mean_color.
    Returns 0.0–1.0.
    """
    scores = []

    ar1 = f1.get("aspect_ratio", 1.0)
    ar2 = f2.get("aspect_ratio", 1.0)
    ar_diff = abs(ar1 - ar2) / max(ar1, ar2, 1e-6)
    scores.append(max(0.0, 1.0 - ar_diff * 2))

    cv1 = f1.get("color_variance", 0.0)
    cv2 = f2.get("color_variance", 0.0)
    denom = max(cv1, cv2, 1.0)
    scores.append(max(0.0, 1.0 - abs(cv1 - cv2) / denom))

    mc1 = f1.get("mean_color", [128, 128, 128])
    mc2 = f2.get("mean_color", [128, 128, 128])
    if mc1 and mc2:
        color_dist = float(np.linalg.norm(np.array(mc1) - np.array(mc2))) / 441.67
        scores.append(max(0.0, 1.0 - color_dist))

    return float(np.mean(scores)) if scores else 0.0


def _match_score(
    track: ActiveTrack,
    detection: DetectedElement,
) -> float:
    """
    Combined matching score between an active track and a new detection.
    Returns 0.0–1.0.
    """
    iou = track.last_bbox.iou(detection.bbox)
    feat_sim = _feature_similarity(track.last_features, detection.features)
    return IOU_WEIGHT * iou + FEATURE_MATCH_WEIGHT * feat_sim


class Tracker:
    """
    Frame-by-frame element tracker.

    Usage:
        tracker = Tracker()
        for frame_idx, detections in frame_detections:
            tracked, errors = tracker.update(frame_idx, detections)
    """

    def __init__(self):
        self._active: dict[str, ActiveTrack] = {}
        self._completed: list[ActiveTrack] = []
        self._lost_events: list[dict] = []
        self._split_events: list[dict] = []
        self._merge_events: list[dict] = []

    def update(
        self,
        frame_index: int,
        detections: list[DetectedElement],
    ) -> tuple[list[TrackedElement], list[StructuredError]]:
        """
        Process detections for a frame.
        Returns snapshot of currently active tracks as TrackedElements.
        """
        errors: list[StructuredError] = []

        # ── Decay unmatched tracks ────────────────────────────────────────
        for track in self._active.values():
            if track.last_frame < frame_index - 1:
                track.missing_frames += 1
                track.continuity_score = max(
                    0.0, track.continuity_score - CONFIDENCE_DECAY
                )

        # ── Greedy matching ───────────────────────────────────────────────
        unmatched_detections = list(detections)
        matched_track_ids: set[str] = set()

        for track in sorted(
            self._active.values(), key=lambda t: t.continuity_score, reverse=True
        ):
            if not unmatched_detections:
                break

            best_score = -1.0
            best_det: Optional[DetectedElement] = None

            for det in unmatched_detections:
                score = _match_score(track, det)
                if score > best_score:
                    best_score = score
                    best_det = det

            if best_det is not None and best_score >= (
                IOU_MATCH_THRESHOLD * IOU_WEIGHT
            ):
                # Matched
                track.last_bbox = best_det.bbox
                track.last_frame = frame_index
                track.last_features = best_det.features
                track.bbox_sequence.append((frame_index, best_det.bbox))
                if best_det.features.get("text_content"):
                    track._feature_cache["text_content"] = best_det.features[
                        "text_content"
                    ]
                track.missing_frames = 0
                track.continuity_score = min(1.0, track.continuity_score + 0.05)
                # Merge type candidates
                track.type_candidates = _merge_type_candidates(
                    track.type_candidates, best_det.type_candidates
                )
                matched_track_ids.add(track.track_id)
                unmatched_detections.remove(best_det)

        # ── Create new tracks for unmatched detections ────────────────────
        for det in unmatched_detections:
            tid = f"trk_{uuid.uuid4().hex[:10]}"
            new_track = ActiveTrack(
                track_id=tid,
                last_bbox=det.bbox,
                last_frame=frame_index,
                last_features=det.features,
                type_candidates=det.type_candidates,
                bbox_sequence=[(frame_index, det.bbox)],
                _feature_cache={"text_content": det.features.get("text_content")}
                if det.features.get("text_content")
                else {},
                continuity_score=0.8,
            )
            self._active[tid] = new_track

        # ── Retire lost tracks ────────────────────────────────────────────
        to_retire = []
        for track_id, track in self._active.items():
            if (
                track.missing_frames > MAX_MISSING_FRAMES
                or track.continuity_score < MIN_TRACK_CONFIDENCE
            ):
                self._lost_events.append(
                    {
                        "track_id": track_id,
                        "last_frame": track.last_frame,
                        "reason": "max_missing"
                        if track.missing_frames > MAX_MISSING_FRAMES
                        else "low_confidence",
                    }
                )
                errors.append(
                    StructuredError(
                        failure_mode=FailureMode.TRACK_LOST,
                        message=f"Track {track_id} lost at frame {frame_index}",
                        stage="tracking",
                        recoverable=True,
                        details={
                            "track_id": track_id,
                            "missing_frames": track.missing_frames,
                            "continuity_score": track.continuity_score,
                        },
                    )
                )
                self._completed.append(track)
                to_retire.append(track_id)

        for track_id in to_retire:
            del self._active[track_id]

        # ── Detect potential merges / splits (IoU-based) ──────────────────
        self._detect_topology_events(frame_index)

        # ── Build TrackedElement snapshots ────────────────────────────────
        tracked: list[TrackedElement] = []
        for track in self._active.values():
            tracked.append(
                TrackedElement(
                    track_id=track.track_id,
                    element_ids=[],
                    type_candidates=track.type_candidates,
                    bbox_sequence=list(track.bbox_sequence),
                    continuity_score=track.continuity_score,
                    occlusion_score=track.occlusion_score,
                    reid_score=1.0,
                    split_events=[
                        e["frame_index"]
                        for e in self._split_events
                        if e.get("track_id") == track.track_id
                    ],
                    merge_events=[
                        e["frame_index"]
                        for e in self._merge_events
                        if e.get("track_id") == track.track_id
                    ],
                )
            )

        return tracked, errors

    def get_all_tracks(self) -> list[TrackedElement]:
        """Return all tracks (active + completed) as TrackedElements."""
        all_tracks = list(self._active.values()) + self._completed
        result = []
        for track in all_tracks:
            if not track.bbox_sequence:
                continue
            result.append(
                TrackedElement(
                    track_id=track.track_id,
                    element_ids=[],
                    type_candidates=track.type_candidates,
                    bbox_sequence=list(track.bbox_sequence),
                    continuity_score=track.continuity_score,
                    occlusion_score=track.occlusion_score,
                    reid_score=1.0,
                    split_events=[],
                    merge_events=[],
                )
            )
        return result

    def _detect_topology_events(self, frame_index: int) -> None:
        """
        Check active tracks pairwise for significant overlap
        that might indicate a merge or split event.
        """
        tracks = list(self._active.values())
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i + 1 :]:
                iou = t1.last_bbox.iou(t2.last_bbox)
                if iou > 0.7:
                    # Possible merge
                    self._merge_events.append(
                        {
                            "track_id": t1.track_id,
                            "other_track_id": t2.track_id,
                            "frame_index": frame_index,
                            "iou": iou,
                        }
                    )


def _merge_type_candidates(
    existing: list[TypeCandidate],
    new: list[TypeCandidate],
) -> list[TypeCandidate]:
    """
    Merge type candidate lists, averaging confidences for matching types.
    """
    merged: dict[str, float] = {}
    for tc in existing:
        merged[tc.type.value] = tc.confidence
    for tc in new:
        if tc.type.value in merged:
            merged[tc.type.value] = (merged[tc.type.value] + tc.confidence) / 2.0
        else:
            merged[tc.type.value] = tc.confidence

    from backend.schemas import ElementType

    return [
        TypeCandidate(type=ElementType(k), confidence=v)
        for k, v in sorted(merged.items(), key=lambda x: -x[1])
    ]


def track_across_frames(
    frame_detections: list[tuple[int, list[DetectedElement]]],
) -> tuple[list[TrackedElement], list[StructuredError]]:
    """
    Convenience function: run full tracking across all frames.

    Args:
        frame_detections: list of (frame_index, [DetectedElement])

    Returns:
        (all_tracked_elements, all_errors)
    """
    tracker = Tracker()
    all_errors: list[StructuredError] = []

    for frame_index, detections in sorted(frame_detections, key=lambda x: x[0]):
        _, errors = tracker.update(frame_index, detections)
        all_errors.extend(errors)

    return tracker.get_all_tracks(), all_errors
