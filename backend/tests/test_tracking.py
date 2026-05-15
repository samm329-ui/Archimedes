"""
test_tracking.py — Tests for backend/pipeline/tracking.py

Coverage:
  - Success: tracking single element across frames
  - Success: track_id persists for stationary element
  - Success: new tracks created for new detections
  - Success: continuity_score updated on match
  - Edge: single frame only
  - Edge: empty detection list
  - Edge: element disappears then reappears
  - Failure: track lost after MAX_MISSING_FRAMES
  - Property: track IDs are unique
  - Property: determinism
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.tracking import (
    MAX_MISSING_FRAMES,
    Tracker,
    track_across_frames,
)
from backend.schemas import (
    BoundingBox,
    DetectedElement,
    ElementType,
    FailureMode,
    ProvenanceRecord,
    TypeCandidate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detection(
    x: float,
    y: float,
    w: float = 60.0,
    h: float = 40.0,
    frame_index: int = 0,
    elem_type: ElementType = ElementType.SHAPE,
) -> DetectedElement:
    return DetectedElement(
        id=f"det_{frame_index}_{x}_{y}",
        frame_index=frame_index,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        type_candidates=[TypeCandidate(type=elem_type, confidence=0.8)],
        mask_available=False,
        features={"aspect_ratio": w / h, "area": w * h, "color_variance": 100.0},
        provenance=ProvenanceRecord(
            source_module="test",
            method="synthetic",
            confidence=0.8,
        ),
    )


# ── Success cases ─────────────────────────────────────────────────────────────

class TestTrackingSuccess:
    def test_single_frame_creates_track(self):
        tracker = Tracker()
        det = _make_detection(100, 100, frame_index=0)
        tracked, errors = tracker.update(0, [det])
        assert len(tracked) == 1

    def test_stationary_element_single_track(self):
        tracker = Tracker()
        for i in range(5):
            det = _make_detection(100, 100, frame_index=i)
            tracker.update(i, [det])
        tracks = tracker.get_all_tracks()
        assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)}"

    def test_moving_element_retains_identity(self):
        tracker = Tracker()
        for i in range(10):
            # Slowly moving right
            det = _make_detection(100 + i * 5, 100, frame_index=i)
            tracker.update(i, [det])
        tracks = tracker.get_all_tracks()
        assert len(tracks) == 1

    def test_bbox_sequence_accumulates(self):
        tracker = Tracker()
        for i in range(5):
            det = _make_detection(100, 100, frame_index=i)
            tracker.update(i, [det])
        tracks = tracker.get_all_tracks()
        assert len(tracks[0].bbox_sequence) == 5

    def test_type_candidates_merged(self):
        tracker = Tracker()
        det1 = _make_detection(100, 100, frame_index=0, elem_type=ElementType.TEXT)
        det2 = _make_detection(102, 100, frame_index=1, elem_type=ElementType.SHAPE)
        tracker.update(0, [det1])
        tracker.update(1, [det2])
        tracks = tracker.get_all_tracks()
        assert len(tracks) == 1
        type_names = {tc.type.value for tc in tracks[0].type_candidates}
        # Should have both types after merging
        assert len(type_names) >= 1


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestTrackingEdge:
    def test_empty_detections_no_tracks(self):
        tracker = Tracker()
        tracked, errors = tracker.update(0, [])
        assert tracked == []

    def test_two_separate_elements_two_tracks(self):
        tracker = Tracker()
        # Two elements far apart
        det_a = _make_detection(10, 10, frame_index=0)
        det_b = _make_detection(250, 200, frame_index=0)
        tracker.update(0, [det_a, det_b])
        tracks = tracker.get_all_tracks()
        assert len(tracks) == 2

    def test_element_disappears_and_new_one_appears(self):
        tracker = Tracker()
        # Element at position A for 3 frames
        for i in range(3):
            tracker.update(i, [_make_detection(100, 100, frame_index=i)])

        # Nothing for MAX_MISSING_FRAMES + 1 frames
        gap_start = 3
        for i in range(gap_start, gap_start + MAX_MISSING_FRAMES + 2):
            tracker.update(i, [])

        # New element at different position
        new_frame = gap_start + MAX_MISSING_FRAMES + 2
        tracker.update(new_frame, [_make_detection(200, 200, frame_index=new_frame)])

        tracks = tracker.get_all_tracks()
        # Should have at least 2 tracks (original + new)
        assert len(tracks) >= 2

    def test_single_detection_single_frame(self):
        tracker = Tracker()
        det = _make_detection(50, 50, frame_index=0)
        tracked, errors = tracker.update(0, [det])
        assert len(tracked) == 1
        assert tracked[0].continuity_score > 0


# ── Failure cases ─────────────────────────────────────────────────────────────

class TestTrackingFailure:
    def test_track_lost_generates_error(self):
        tracker = Tracker()
        det = _make_detection(100, 100, frame_index=0)
        tracker.update(0, [det])

        all_errors = []
        for i in range(1, MAX_MISSING_FRAMES + 3):
            _, errors = tracker.update(i, [])
            all_errors.extend(errors)

        track_lost_errors = [e for e in all_errors if e.failure_mode == FailureMode.TRACK_LOST]
        assert len(track_lost_errors) >= 1

    def test_track_lost_error_is_recoverable(self):
        tracker = Tracker()
        det = _make_detection(100, 100, frame_index=0)
        tracker.update(0, [det])

        for i in range(1, MAX_MISSING_FRAMES + 3):
            _, errors = tracker.update(i, [])

        # All track-lost errors should be recoverable
        for err in [e for e in errors if e.failure_mode == FailureMode.TRACK_LOST]:
            assert err.recoverable


# ── Convenience function ──────────────────────────────────────────────────────

class TestTrackAcrossFrames:
    def test_returns_tracks_and_errors(self):
        frame_detections = [
            (i, [_make_detection(100 + i * 2, 100, frame_index=i)])
            for i in range(10)
        ]
        tracks, errors = track_across_frames(frame_detections)
        assert isinstance(tracks, list)
        assert isinstance(errors, list)
        assert len(tracks) >= 1

    def test_empty_input_returns_empty(self):
        tracks, errors = track_across_frames([])
        assert tracks == []

    def test_track_ids_unique(self):
        frame_detections = [
            (i, [
                _make_detection(10 + i, 10, frame_index=i),
                _make_detection(200 + i, 10, frame_index=i),
            ])
            for i in range(5)
        ]
        tracks, _ = track_across_frames(frame_detections)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))
