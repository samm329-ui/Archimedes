"""
test_curve_fitting_hypothesis.py — Tests for curve_fitting.py and hypothesis_engine.py
"""
from __future__ import annotations
import numpy as np
import pytest
from backend.pipeline.curve_fitting import (
    fit_all_easings, build_motion_hypotheses, smooth_trajectory,
    detect_rotation, detect_opacity_fade, FitResult, MIN_R2_TO_USE_FIT
)
from backend.pipeline.hypothesis_engine import (
    HypothesisSet, ElementHypothesisManager,
    merge_type_hypotheses, select_best_motion_hypotheses,
    validate_hypothesis_coverage, HypothesisState
)
from backend.schemas import (
    EasingType, ElementType, MotionHypothesis, MotionPrimitive, TypeCandidate
)


def _linear_values(n=30, slope=5.0, noise=0.0):
    frames = np.arange(n, dtype=float)
    vals = 100.0 + frames * slope
    if noise > 0:
        vals += np.random.normal(0, noise, n)
    return frames, vals


def _ease_out_values(n=30):
    frames = np.arange(n, dtype=float)
    t = frames / (n - 1)
    vals = 100 + 200 * (1 - (1 - t)**3)
    return frames, vals


# ── Curve fitting tests ───────────────────────────────────────────────────────

class TestFitAllEasings:
    def test_returns_list_of_fitresults(self):
        frames, vals = _linear_values()
        results = fit_all_easings(frames, vals)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_are_fitresult_type(self):
        frames, vals = _linear_values()
        for r in fit_all_easings(frames, vals):
            assert isinstance(r, FitResult)

    def test_sorted_by_r2_descending(self):
        frames, vals = _ease_out_values()
        results = fit_all_easings(frames, vals)
        r2s = [r.r2 for r in results]
        assert r2s == sorted(r2s, reverse=True)

    def test_linear_motion_best_fit_is_linear_or_similar(self):
        frames, vals = _linear_values(n=40, slope=4.0)
        results = fit_all_easings(frames, vals)
        best = results[0]
        assert best.r2 > 0.9

    def test_ease_out_motion_detected(self):
        frames, vals = _ease_out_values(n=40)
        results = fit_all_easings(frames, vals)
        best = results[0]
        assert best.r2 > 0.7
        assert best.easing == EasingType.EASE_OUT_CUBIC

    def test_raw_curve_length_reasonable(self):
        frames, vals = _linear_values()
        for r in fit_all_easings(frames, vals):
            assert 1 <= len(r.raw_curve) <= 40

    def test_sparse_data_returns_result(self):
        frames = np.array([0.0, 5.0, 10.0])
        vals = np.array([0.0, 50.0, 100.0])
        results = fit_all_easings(frames, vals)
        assert len(results) >= 1

    def test_single_point_returns_result(self):
        frames = np.array([0.0])
        vals = np.array([100.0])
        results = fit_all_easings(frames, vals)
        assert len(results) >= 1
        assert results[0].easing == EasingType.UNKNOWN

    def test_confidence_in_range(self):
        frames, vals = _linear_values()
        for r in fit_all_easings(frames, vals):
            assert 0.0 <= r.confidence <= 1.0


class TestBuildMotionHypotheses:
    def test_returns_list(self):
        frames, vals = _linear_values()
        hyps = build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_X, 0)
        assert isinstance(hyps, list)

    def test_at_least_one_hypothesis(self):
        frames, vals = _linear_values()
        hyps = build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_X, 0)
        assert len(hyps) >= 1

    def test_confidence_in_range(self):
        frames, vals = _linear_values()
        for h in build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_X, 0):
            assert 0.0 <= h.confidence <= 1.0

    def test_primitive_set_correctly(self):
        frames, vals = _linear_values()
        for h in build_motion_hypotheses(frames, vals, MotionPrimitive.SCALE, 0):
            assert h.primitive == MotionPrimitive.SCALE

    def test_start_frame_set(self):
        frames = np.arange(5, 25, dtype=float)
        vals = np.linspace(100, 200, 20)
        hyps = build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_Y, 5)
        assert hyps[0].start_frame == 5

    def test_raw_curve_not_empty(self):
        frames, vals = _linear_values()
        hyps = build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_X, 0)
        assert len(hyps[0].raw_curve) >= 1

    def test_tiny_motion_low_confidence(self):
        frames = np.arange(20, dtype=float)
        vals = np.full(20, 100.0)  # no motion
        hyps = build_motion_hypotheses(frames, vals, MotionPrimitive.TRANSLATE_X, 0,
                                       magnitude_scale=50.0)
        if hyps:
            assert hyps[0].confidence < 0.6


class TestSmoothTrajectory:
    def test_same_length(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        smoothed = smooth_trajectory(vals)
        assert len(smoothed) == len(vals)

    def test_short_array_returned_unchanged(self):
        vals = np.array([1.0, 2.0])
        result = smooth_trajectory(vals)
        assert len(result) == 2

    def test_reduces_noise(self):
        np.random.seed(42)
        clean = np.linspace(0, 100, 30)
        noisy = clean + np.random.normal(0, 5, 30)
        smoothed = smooth_trajectory(noisy)
        noise_before = float(np.std(np.diff(noisy)))
        noise_after = float(np.std(np.diff(smoothed)))
        assert noise_after <= noise_before


# ── Hypothesis engine tests ───────────────────────────────────────────────────

class TestHypothesisSet:
    def test_add_and_retrieve_best(self):
        hs = HypothesisSet[str]("test")
        hs.add("A", 0.8)
        hs.add("B", 0.5)
        assert hs.best_value() == "A"
        assert hs.best_confidence() == 0.8

    def test_sorted_descending(self):
        hs = HypothesisSet[str]("test")
        hs.add("C", 0.3)
        hs.add("A", 0.9)
        hs.add("B", 0.6)
        assert hs.hypotheses[0].value == "A"

    def test_below_min_confidence_pruned(self):
        hs = HypothesisSet[str]("test")
        hs.add("weak", 0.01)  # Below MIN_RETAIN_CONFIDENCE
        assert hs.best_value() is None

    def test_cap_enforced(self):
        from backend.pipeline.hypothesis_engine import MAX_HYPOTHESES_PER_SLOT
        hs = HypothesisSet[str]("test")
        for i in range(MAX_HYPOTHESES_PER_SLOT + 5):
            hs.add(f"item_{i}", 0.5 - i * 0.02)
        assert len(hs.hypotheses) <= MAX_HYPOTHESES_PER_SLOT

    def test_reinforce_existing(self):
        hs = HypothesisSet[str]("test")
        hs.add("X", 0.5)
        hs.reinforce("X", 0.2, source="test")
        assert hs.best_confidence() > 0.5

    def test_penalize_existing(self):
        hs = HypothesisSet[str]("test")
        hs.add("X", 0.8)
        hs.penalize("X", 0.3)
        assert hs.best_confidence() < 0.8

    def test_is_decided_true_when_clear_winner(self):
        hs = HypothesisSet[str]("test")
        hs.add("winner", 0.95)
        hs.add("loser", 0.20)
        assert hs.is_decided() is True

    def test_is_decided_false_when_close(self):
        hs = HypothesisSet[str]("test")
        hs.add("A", 0.80)
        hs.add("B", 0.75)
        assert hs.is_decided() is False

    def test_to_dict_format(self):
        hs = HypothesisSet[str]("test")
        hs.add("A", 0.7)
        result = hs.to_dict()
        assert isinstance(result, list)
        assert result[0]["value"] == "A"


class TestElementHypothesisManager:
    def test_add_type_evidence(self):
        mgr = ElementHypothesisManager("elem_1")
        mgr.add_type_evidence(ElementType.TEXT, 0.8, "detector")
        t, c = mgr.get_best_type()
        assert t == ElementType.TEXT
        assert c == 0.8

    def test_add_multiple_types(self):
        mgr = ElementHypothesisManager("elem_1")
        mgr.add_type_evidence(ElementType.TEXT, 0.7, "source1")
        mgr.add_type_evidence(ElementType.SHAPE, 0.4, "source2")
        candidates = mgr.get_type_candidates()
        types = [c.type for c in candidates]
        assert ElementType.TEXT in types
        assert ElementType.SHAPE in types

    def test_summary_returns_dict(self):
        mgr = ElementHypothesisManager("elem_1")
        mgr.add_type_evidence(ElementType.TEXT, 0.7, "test")
        summary = mgr.summary()
        assert isinstance(summary, dict)
        assert "element_id" in summary

    def test_layer_evidence(self):
        mgr = ElementHypothesisManager("e1")
        mgr.add_layer_evidence(3, 0.7, "layering")
        layer, conf = mgr.get_best_layer()
        assert layer == 3
        assert conf == 0.7


class TestMergeTypeHypotheses:
    def test_merges_matching_types(self):
        s1 = [TypeCandidate(type=ElementType.TEXT, confidence=0.7)]
        s2 = [TypeCandidate(type=ElementType.TEXT, confidence=0.5)]
        merged = merge_type_hypotheses([s1, s2])
        text_cands = [c for c in merged if c.type == ElementType.TEXT]
        assert len(text_cands) == 1
        assert 0.5 <= text_cands[0].confidence <= 0.7

    def test_empty_input_returns_empty(self):
        assert merge_type_hypotheses([]) == []

    def test_weighted_merge(self):
        s1 = [TypeCandidate(type=ElementType.TEXT, confidence=0.8)]
        s2 = [TypeCandidate(type=ElementType.SHAPE, confidence=0.6)]
        merged = merge_type_hypotheses([s1, s2], weights=[2.0, 1.0])
        types = [c.type for c in merged]
        assert ElementType.TEXT in types


class TestSelectBestMotionHypotheses:
    def test_returns_list(self):
        hyps = [
            MotionHypothesis(primitive=MotionPrimitive.TRANSLATE_X,
                             easing=EasingType.LINEAR, from_value=0, to_value=100,
                             duration_frames=20, start_frame=0,
                             raw_curve=[0.0, 1.0], confidence=0.8),
        ]
        result = select_best_motion_hypotheses(hyps)
        assert isinstance(result, list)

    def test_max_per_primitive_respected(self):
        hyps = []
        for i in range(5):
            hyps.append(MotionHypothesis(
                primitive=MotionPrimitive.TRANSLATE_X,
                easing=EasingType.LINEAR, from_value=0, to_value=100,
                duration_frames=20, start_frame=0,
                raw_curve=[0.0, 1.0], confidence=0.8 - i * 0.1,
            ))
        result = select_best_motion_hypotheses(hyps, max_per_primitive=2)
        tx = [h for h in result if h.primitive == MotionPrimitive.TRANSLATE_X]
        assert len(tx) <= 2

    def test_always_at_least_one_kept(self):
        hyps = [MotionHypothesis(
            primitive=MotionPrimitive.UNKNOWN, easing=EasingType.UNKNOWN,
            from_value=0, to_value=0, duration_frames=0, start_frame=0,
            raw_curve=[], confidence=0.01,
        )]
        result = select_best_motion_hypotheses(hyps)
        assert len(result) >= 1

    def test_sorted_by_confidence(self):
        hyps = [
            MotionHypothesis(primitive=MotionPrimitive.TRANSLATE_X,
                             easing=EasingType.LINEAR, from_value=0, to_value=100,
                             duration_frames=20, start_frame=0,
                             raw_curve=[0.0, 1.0], confidence=c)
            for c in [0.3, 0.9, 0.6]
        ]
        result = select_best_motion_hypotheses(hyps)
        confs = [h.confidence for h in result]
        assert confs == sorted(confs, reverse=True)

    def test_empty_returns_empty(self):
        assert select_best_motion_hypotheses([]) == []
