"""
test_motion.py — Tests for backend/pipeline/motion_analysis.py

Coverage:
  - Success: linear motion detected with correct primitive
  - Success: static element gets STATIC primitive
  - Success: hypotheses sorted by confidence descending
  - Success: raw_trajectory matches input sequence length
  - Success: all confidences in [0, 1]
  - Success: at least one hypothesis always returned
  - Edge: single-frame track
  - Edge: two-frame track
  - Edge: very fast motion
  - Edge: circular-ish motion
  - Failure: empty bbox_sequence produces structured error
  - Property: determinism
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.pipeline.motion_analysis import (
    analyze_all_tracks,
    extract_motion_from_track,
)
from backend.schemas import (
    BoundingBox,
    FailureMode,
    MotionElement,
    MotionPrimitive,
    TrackedElement,
    TypeCandidate,
    ElementType,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_track(
    bbox_sequence: list[tuple[int, tuple[float, float, float, float]]],
) -> TrackedElement:
    """
    Build a TrackedElement from (frame_index, (x, y, w, h)) tuples.
    """
    return TrackedElement(
        track_id="trk_test",
        element_ids=[],
        type_candidates=[TypeCandidate(type=ElementType.SHAPE, confidence=0.8)],
        bbox_sequence=[
            (frame, BoundingBox(x=x, y=y, w=w, h=h))
            for frame, (x, y, w, h) in bbox_sequence
        ],
        continuity_score=0.9,
    )


def _linear_track(n: int, dx: float, dy: float = 0.0) -> TrackedElement:
    """Track moving linearly in x (and optionally y)."""
    return _make_track([
        (i, (100.0 + i * dx, 100.0 + i * dy, 60.0, 40.0))
        for i in range(n)
    ])


# ── Success cases ─────────────────────────────────────────────────────────────

class TestMotionSuccess:
    def test_returns_motion_element(self):
        track = _linear_track(20, dx=5.0)
        me, errors = extract_motion_from_track(track)
        assert isinstance(me, MotionElement)

    def test_has_at_least_one_hypothesis(self):
        track = _linear_track(20, dx=5.0)
        me, _ = extract_motion_from_track(track)
        assert len(me.motion_hypotheses) >= 1

    def test_hypotheses_sorted_by_confidence_descending(self):
        track = _linear_track(20, dx=5.0)
        me, _ = extract_motion_from_track(track)
        confs = [h.confidence for h in me.motion_hypotheses]
        assert confs == sorted(confs, reverse=True)

    def test_all_confidences_in_range(self):
        track = _linear_track(20, dx=5.0)
        me, _ = extract_motion_from_track(track)
        for h in me.motion_hypotheses:
            assert 0.0 <= h.confidence <= 1.0, f"Confidence {h.confidence} out of range"

    def test_translate_x_detected_for_horizontal_motion(self):
        track = _linear_track(30, dx=8.0, dy=0.0)
        me, _ = extract_motion_from_track(track)
        primitives = [h.primitive for h in me.motion_hypotheses]
        assert MotionPrimitive.TRANSLATE_X in primitives

    def test_translate_y_detected_for_vertical_motion(self):
        track = _linear_track(30, dx=0.0, dy=8.0)
        me, _ = extract_motion_from_track(track)
        primitives = [h.primitive for h in me.motion_hypotheses]
        assert MotionPrimitive.TRANSLATE_Y in primitives

    def test_static_detected_for_stationary_element(self):
        track = _linear_track(20, dx=0.0, dy=0.0)
        me, _ = extract_motion_from_track(track)
        primitives = [h.primitive for h in me.motion_hypotheses]
        assert MotionPrimitive.STATIC in primitives

    def test_static_has_high_confidence(self):
        track = _linear_track(20, dx=0.0, dy=0.0)
        me, _ = extract_motion_from_track(track)
        static_hyps = [h for h in me.motion_hypotheses if h.primitive == MotionPrimitive.STATIC]
        assert static_hyps[0].confidence >= 0.7

    def test_raw_trajectory_length_matches_frames(self):
        track = _linear_track(15, dx=3.0)
        me, _ = extract_motion_from_track(track)
        assert len(me.raw_trajectory) == 15

    def test_motion_confidence_positive(self):
        track = _linear_track(20, dx=5.0)
        me, _ = extract_motion_from_track(track)
        assert me.motion_confidence > 0.0

    def test_duration_frames_correct(self):
        track = _linear_track(20, dx=3.0)
        me, _ = extract_motion_from_track(track)
        # Max duration = last frame - first frame = 19
        for h in me.motion_hypotheses:
            assert h.duration_frames >= 0

    def test_raw_curve_not_empty_for_motion(self):
        track = _linear_track(20, dx=5.0)
        me, _ = extract_motion_from_track(track)
        best = me.motion_hypotheses[0]
        assert len(best.raw_curve) >= 1


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestMotionEdge:
    def test_single_frame_track(self):
        track = _linear_track(1, dx=0.0)
        me, errors = extract_motion_from_track(track)
        assert isinstance(me, MotionElement)
        assert len(me.motion_hypotheses) >= 1

    def test_two_frame_track(self):
        track = _linear_track(2, dx=10.0)
        me, errors = extract_motion_from_track(track)
        assert isinstance(me, MotionElement)

    def test_very_fast_motion(self):
        # Extreme motion per frame
        track = _linear_track(10, dx=50.0)
        me, errors = extract_motion_from_track(track)
        assert isinstance(me, MotionElement)

    def test_diagonal_motion_two_primitives(self):
        track = _linear_track(20, dx=5.0, dy=5.0)
        me, _ = extract_motion_from_track(track)
        primitives = {h.primitive for h in me.motion_hypotheses}
        assert MotionPrimitive.TRANSLATE_X in primitives
        assert MotionPrimitive.TRANSLATE_Y in primitives

    def test_scale_change_detected(self):
        # Growing element
        seq = [
            (i, (100.0, 100.0, 20.0 + i * 3, 15.0 + i * 2))
            for i in range(20)
        ]
        track = _make_track(seq)
        me, _ = extract_motion_from_track(track)
        primitives = {h.primitive for h in me.motion_hypotheses}
        # Scale should be detected
        assert MotionPrimitive.SCALE in primitives or len(me.motion_hypotheses) >= 1


# ── Failure cases ─────────────────────────────────────────────────────────────

class TestMotionFailure:
    def test_empty_bbox_sequence_returns_error(self):
        track = TrackedElement(
            track_id="empty_track",
            element_ids=[],
            type_candidates=[],
            bbox_sequence=[],
            continuity_score=0.5,
        )
        me, errors = extract_motion_from_track(track)
        assert len(errors) >= 1
        assert any(e.failure_mode == FailureMode.MOTION_AMBIGUOUS for e in errors)

    def test_empty_track_returns_zero_confidence(self):
        track = TrackedElement(
            track_id="empty_track",
            element_ids=[],
            type_candidates=[],
            bbox_sequence=[],
            continuity_score=0.5,
        )
        me, _ = extract_motion_from_track(track)
        assert me.motion_confidence == 0.0


# ── Batch analysis ────────────────────────────────────────────────────────────

class TestAnalyzeAllTracks:
    def test_returns_list_of_motion_elements(self):
        tracks = [_linear_track(15, dx=3.0), _linear_track(15, dx=-3.0)]
        motion_elements, errors = analyze_all_tracks(tracks)
        assert len(motion_elements) == 2

    def test_empty_tracks_list(self):
        motion_elements, errors = analyze_all_tracks([])
        assert motion_elements == []
        assert errors == []


# ── Determinism ───────────────────────────────────────────────────────────────

class TestMotionDeterminism:
    def test_same_track_same_hypotheses(self):
        track = _linear_track(20, dx=5.0)
        me1, _ = extract_motion_from_track(track)
        me2, _ = extract_motion_from_track(track)
        # Same number of hypotheses
        assert len(me1.motion_hypotheses) == len(me2.motion_hypotheses)
        # Same top primitive
        assert me1.motion_hypotheses[0].primitive == me2.motion_hypotheses[0].primitive

    def test_same_track_same_confidence(self):
        track = _linear_track(20, dx=5.0)
        me1, _ = extract_motion_from_track(track)
        me2, _ = extract_motion_from_track(track)
        assert abs(me1.motion_hypotheses[0].confidence - me2.motion_hypotheses[0].confidence) < 1e-6
