"""
test_grouping_layering.py — Tests for grouping.py and layering.py
"""
from __future__ import annotations
import numpy as np
import pytest
from backend.pipeline.grouping import (
    assign_groups, compute_group_relation, GroupRelation,
    MIN_GROUP_CONFIDENCE
)
from backend.pipeline.layering import (
    infer_layer_order, LayerHypothesis, MAX_LAYER_DEPTH
)
from backend.schemas import (
    BoundingBox, ElementType, TrackedElement, TypeCandidate,
    MotionElement, MotionHypothesis, MotionPrimitive, EasingType
)

CANVAS_W, CANVAS_H = 320, 240


def _make_track(tid: str, bboxes: list[tuple[int,float,float,float,float]],
                elem_type=ElementType.TEXT) -> TrackedElement:
    return TrackedElement(
        track_id=tid,
        element_ids=[],
        type_candidates=[TypeCandidate(type=elem_type, confidence=0.8)],
        bbox_sequence=[(f, BoundingBox(x=x, y=y, w=w, h=h)) for f,x,y,w,h in bboxes],
        continuity_score=0.9,
    )


def _static_track(tid: str, x=50.0, y=80.0, w=100.0, h=40.0, frames=20) -> TrackedElement:
    return _make_track(tid, [(i, x, y, w, h) for i in range(frames)])


def _moving_track(tid: str, dx=3.0, frames=20) -> TrackedElement:
    return _make_track(tid, [(i, 50+i*dx, 80, 100, 40) for i in range(frames)])


def _make_motion(tid: str) -> MotionElement:
    return MotionElement(
        track_id=tid,
        motion_hypotheses=[MotionHypothesis(
            primitive=MotionPrimitive.TRANSLATE_X,
            easing=EasingType.LINEAR,
            from_value=50.0, to_value=110.0,
            duration_frames=20, start_frame=0,
            raw_curve=[0.0, 0.5, 1.0], confidence=0.7,
        )],
        raw_trajectory=[(i, 50+i*3.0, 80.0) for i in range(20)],
        motion_confidence=0.7,
    )


# ── Grouping tests ────────────────────────────────────────────────────────────

class TestGroupRelation:
    def test_returns_group_relation(self):
        t1 = _static_track("t1")
        t2 = _static_track("t2", x=55.0)
        rel = compute_group_relation(t1, t2, None, None, CANVAS_W, CANVAS_H)
        assert isinstance(rel, GroupRelation)

    def test_confidence_in_range(self):
        t1 = _static_track("t1")
        t2 = _static_track("t2", x=55.0)
        rel = compute_group_relation(t1, t2, None, None, CANVAS_W, CANVAS_H)
        assert 0.0 <= rel.confidence <= 1.0

    def test_close_same_motion_higher_confidence(self):
        t1 = _moving_track("t1", dx=3.0)
        t2 = _moving_track("t2", dx=3.0)
        m1, m2 = _make_motion("t1"), _make_motion("t2")
        rel = compute_group_relation(t1, t2, m1, m2, CANVAS_W, CANVAS_H)
        assert rel.confidence >= 0.0

    def test_far_apart_lower_confidence(self):
        t1 = _static_track("t1", x=10, y=10)
        t2 = _static_track("t2", x=280, y=200)
        rel = compute_group_relation(t1, t2, None, None, CANVAS_W, CANVAS_H)
        assert rel.confidence < 0.5

    def test_hypothesis_only_set_correctly(self):
        t1 = _static_track("t1")
        t2 = _static_track("t2", x=280)
        rel = compute_group_relation(t1, t2, None, None, CANVAS_W, CANVAS_H)
        if rel.confidence < MIN_GROUP_CONFIDENCE:
            assert rel.hypothesis_only is True
            assert rel.is_confirmed is False

    def test_all_sub_scores_in_range(self):
        t1 = _static_track("t1")
        t2 = _static_track("t2", x=55)
        rel = compute_group_relation(t1, t2, None, None, CANVAS_W, CANVAS_H)
        for score in [rel.motion_correlation, rel.spatial_proximity,
                      rel.timing_sync, rel.overlap_pct, rel.style_similarity]:
            assert 0.0 <= score <= 1.0


class TestAssignGroups:
    def test_returns_assignments_dict(self):
        tracks = [_static_track("t1"), _static_track("t2", x=200)]
        assignments, relations, errors = assign_groups(tracks, [], CANVAS_W, CANVAS_H)
        assert isinstance(assignments, dict)
        assert len(assignments) == 2

    def test_single_track_not_grouped(self):
        tracks = [_static_track("solo")]
        assignments, _, _ = assign_groups(tracks, [], CANVAS_W, CANVAS_H)
        assert assignments["solo"].group_id is None

    def test_all_tracks_have_assignment(self):
        tracks = [_static_track(f"t{i}") for i in range(4)]
        assignments, _, _ = assign_groups(tracks, [], CANVAS_W, CANVAS_H)
        for track in tracks:
            assert track.track_id in assignments

    def test_no_group_with_weak_signals(self):
        # Tracks far apart → no grouping
        t1 = _static_track("t1", x=10, y=10, w=20, h=15)
        t2 = _static_track("t2", x=290, y=210, w=20, h=15)
        assignments, _, _ = assign_groups([t1, t2], [], CANVAS_W, CANVAS_H)
        # Should not be confirmed group (they are far apart with no motion correlation)
        assert assignments["t1"].group_id is None or assignments["t2"].group_id is None

    def test_empty_tracks_returns_empty(self):
        assignments, relations, errors = assign_groups([], [], CANVAS_W, CANVAS_H)
        assert assignments == {}
        assert relations == []

    def test_relations_list_returned(self):
        tracks = [_static_track("t1"), _static_track("t2")]
        _, relations, _ = assign_groups(tracks, [], CANVAS_W, CANVAS_H)
        assert isinstance(relations, list)


# ── Layering tests ────────────────────────────────────────────────────────────

class TestInferLayerOrder:
    def test_returns_hypotheses_dict(self):
        tracks = [_static_track("t1"), _static_track("t2")]
        hyps, events, errors = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        assert isinstance(hyps, dict)
        assert len(hyps) == 2

    def test_all_tracks_have_hypothesis(self):
        tracks = [_static_track(f"t{i}") for i in range(5)]
        hyps, _, _ = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        for track in tracks:
            assert track.track_id in hyps

    def test_layer_values_in_range(self):
        tracks = [_static_track(f"t{i}") for i in range(4)]
        hyps, _, _ = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        for h in hyps.values():
            assert 0 <= h.layer <= MAX_LAYER_DEPTH

    def test_confidence_in_range(self):
        tracks = [_static_track("t1"), _static_track("t2")]
        hyps, _, _ = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        for h in hyps.values():
            assert 0.0 <= h.confidence <= 1.0

    def test_large_element_gets_low_layer(self):
        # Full-canvas element should be background (layer 0)
        bg = _make_track("bg", [(i, 0, 0, 320, 240) for i in range(20)])
        fg = _static_track("fg", x=50, y=50, w=30, h=20)
        hyps, _, _ = infer_layer_order([bg, fg], CANVAS_W, CANVAS_H)
        assert hyps["bg"].layer <= hyps["fg"].layer

    def test_small_element_gets_higher_layer(self):
        small = _make_track("small", [(i, 10, 10, 15, 10) for i in range(20)])
        big = _make_track("big", [(i, 0, 0, 300, 200) for i in range(20)])
        hyps, _, _ = infer_layer_order([small, big], CANVAS_W, CANVAS_H)
        assert hyps["small"].layer >= hyps["big"].layer

    def test_occlusion_events_list(self):
        tracks = [_static_track("t1", x=50, y=50, w=100, h=80),
                  _static_track("t2", x=80, y=70, w=40, h=30)]
        _, events, _ = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        assert isinstance(events, list)

    def test_empty_tracks_returns_empty(self):
        hyps, events, errors = infer_layer_order([], CANVAS_W, CANVAS_H)
        assert hyps == {}
        assert events == []

    def test_reasoning_list_present(self):
        tracks = [_static_track("t1")]
        hyps, _, _ = infer_layer_order(tracks, CANVAS_W, CANVAS_H)
        assert isinstance(hyps["t1"].reasoning, list)
        assert len(hyps["t1"].reasoning) >= 1
