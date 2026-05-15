"""
test_role_diagnostics_error.py — Tests for role_assignment.py, diagnostics.py,
error_handler.py, provenance.py, segmentation.py, and reid.py
"""
from __future__ import annotations
import time
import numpy as np
import pytest

from backend.pipeline.role_assignment import assign_roles, score_roles, RoleEvidence
from backend.pipeline.diagnostics import DiagnosticReport, StageRecord
from backend.pipeline.error_handler import (
    ErrorAccumulator, safe_execute, wrap_exception, classify_exception
)
from backend.pipeline.provenance import (
    ProvenanceChain, ProvenanceRegistry, make_record
)
from backend.pipeline.segmentation import segment_element, segment_frame_elements
from backend.pipeline.reid import ReidentificationEngine, ReidCandidate
from backend.schemas import (
    BoundingBox, DetectedElement, ElementType, FailureMode,
    MotionElement, MotionHypothesis, MotionPrimitive, EasingType,
    ProvenanceRecord, StructuredError, TrackedElement, TypeCandidate,
)

CANVAS_W, CANVAS_H = 320, 240


def _make_track(tid, x=50, y=80, w=100, h=40, frames=20) -> TrackedElement:
    return TrackedElement(
        track_id=tid, element_ids=[],
        type_candidates=[TypeCandidate(type=ElementType.TEXT, confidence=0.75)],
        bbox_sequence=[(i, BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)))
                       for i in range(frames)],
        continuity_score=0.9,
    )


def _make_motion(tid) -> MotionElement:
    return MotionElement(
        track_id=tid,
        motion_hypotheses=[MotionHypothesis(
            primitive=MotionPrimitive.TRANSLATE_Y, easing=EasingType.EASE_OUT_CUBIC,
            from_value=300, to_value=80, duration_frames=15, start_frame=0,
            raw_curve=[0.0, 0.5, 1.0], confidence=0.75,
        )],
        raw_trajectory=[(i, 50.0, 80.0) for i in range(20)],
        motion_confidence=0.75,
    )


def _make_detection(eid, x=50, y=80, w=100, h=40, frame=0) -> DetectedElement:
    return DetectedElement(
        id=eid, frame_index=frame,
        bbox=BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)),
        type_candidates=[TypeCandidate(type=ElementType.TEXT, confidence=0.7)],
        features={"aspect_ratio": 2.5, "color_variance": 500.0},
        provenance=ProvenanceRecord(source_module="test", method="synthetic", confidence=0.7),
    )


# ── Role assignment tests ─────────────────────────────────────────────────────

class TestScoreRoles:
    def _ev(self, area=0.05, cx=0.5, cy=0.2, is_text=True, has_enter=True,
            temporal=0.5, layer=2, ar=3.0):
        return RoleEvidence(
            area_norm=area, cx_norm=cx, cy_norm=cy,
            layer=layer, dominant_type=ElementType.TEXT if is_text else ElementType.SHAPE,
            has_enter_motion=has_enter, has_exit_motion=False,
            temporal_fraction=temporal, is_text=is_text, is_shape=not is_text,
            is_overlay=False, aspect_ratio=ar,
        )

    def test_returns_list(self):
        ev = self._ev()
        roles = score_roles(ev)
        assert isinstance(roles, list)

    def test_at_least_one_role(self):
        ev = self._ev()
        roles = score_roles(ev)
        assert len(roles) >= 1

    def test_title_detected_for_top_text(self):
        ev = self._ev(area=0.05, cy=0.2, is_text=True, has_enter=True)
        roles = score_roles(ev)
        role_names = [r.role for r in roles]
        assert "title" in role_names

    def test_background_detected_for_large_element(self):
        ev = self._ev(area=0.7, layer=0, temporal=0.9, is_text=False)
        roles = score_roles(ev)
        role_names = [r.role for r in roles]
        assert "background" in role_names

    def test_all_scores_in_range(self):
        ev = self._ev()
        for r in score_roles(ev):
            assert 0.0 <= r.score <= 1.0

    def test_sorted_descending(self):
        ev = self._ev()
        roles = score_roles(ev)
        scores = [r.score for r in roles]
        assert scores == sorted(scores, reverse=True)


class TestAssignRoles:
    def test_returns_dict(self):
        tracks = [_make_track("t1"), _make_track("t2")]
        result, errors = assign_roles(tracks, [], CANVAS_W, CANVAS_H, 100)
        assert isinstance(result, dict)

    def test_all_tracks_assigned(self):
        tracks = [_make_track(f"t{i}") for i in range(3)]
        result, _ = assign_roles(tracks, [], CANVAS_W, CANVAS_H, 100)
        for t in tracks:
            assert t.track_id in result

    def test_roles_are_lists(self):
        tracks = [_make_track("t1")]
        result, _ = assign_roles(tracks, [], CANVAS_W, CANVAS_H, 100)
        assert isinstance(result["t1"], list)

    def test_roles_have_scores(self):
        tracks = [_make_track("t1")]
        motions = [_make_motion("t1")]
        result, _ = assign_roles(tracks, motions, CANVAS_W, CANVAS_H, 100)
        for r in result["t1"]:
            assert 0.0 <= r.score <= 1.0

    def test_empty_tracks_returns_empty(self):
        result, errors = assign_roles([], [], CANVAS_W, CANVAS_H, 100)
        assert result == {}
        assert errors == []

    def test_layer_map_used(self):
        tracks = [_make_track("t1", y=10, h=20)]
        result, _ = assign_roles(tracks, [], CANVAS_W, CANVAS_H, 100,
                                 layer_map={"t1": 8})
        assert isinstance(result["t1"], list)


# ── Diagnostics tests ─────────────────────────────────────────────────────────

class TestDiagnosticReport:
    def test_start_finish_stage(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("detection")
        time.sleep(0.01)
        diag.finish_stage("detection", output_count=5)
        assert "detection" in diag.stages
        assert diag.stages["detection"].duration_ms > 0
        assert diag.stages["detection"].output_count == 5

    def test_record_errors(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("tracking")
        err = StructuredError(
            failure_mode=FailureMode.TRACK_LOST, message="test error",
            stage="tracking", recoverable=True,
        )
        diag.record_errors([err])
        assert diag.total_errors == 1
        assert diag.stages["tracking"].error_count == 1

    def test_record_confidences(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("detection")
        diag.record_confidences("detection", [0.8, 0.6, 0.9])
        assert diag.stages["detection"].mean_confidence() is not None

    def test_finalize_sets_duration(self):
        diag = DiagnosticReport(video_path="test.mp4")
        time.sleep(0.01)
        diag.finalize()
        assert diag.total_duration_ms is not None
        assert diag.total_duration_ms > 0

    def test_to_dict_complete(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("ingest")
        diag.finish_stage("ingest", output_count=1)
        diag.finalize()
        d = diag.to_dict()
        assert "total_duration_ms" in d
        assert "stages" in d
        assert "ingest" in d["stages"]

    def test_bottleneck_stage(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("detection")
        for _ in range(3):
            diag.record_errors([StructuredError(
                failure_mode=FailureMode.DETECTION_FAILED,
                message="err", stage="detection", recoverable=True,
            )])
        diag.start_stage("tracking")
        assert diag.bottleneck_stage() == "detection"

    def test_confidence_summary(self):
        diag = DiagnosticReport(video_path="test.mp4")
        diag.start_stage("motion_analysis")
        diag.record_confidences("motion_analysis", [0.7, 0.8])
        summary = diag.confidence_summary()
        assert "motion_analysis" in summary


# ── Error handler tests ───────────────────────────────────────────────────────

class TestErrorAccumulator:
    def test_add_and_retrieve(self):
        acc = ErrorAccumulator()
        err = StructuredError(failure_mode=FailureMode.TRACK_LOST,
                              message="test", stage="tracking", recoverable=True)
        acc.add(err)
        assert acc.count() == 1
        assert acc.all()[0].message == "test"

    def test_add_all(self):
        acc = ErrorAccumulator()
        errs = [StructuredError(failure_mode=FailureMode.TRACK_LOST,
                                message=f"e{i}", stage="tracking", recoverable=True)
                for i in range(5)]
        acc.add_all(errs)
        assert acc.count() == 5

    def test_by_stage_filter(self):
        acc = ErrorAccumulator()
        acc.add(StructuredError(failure_mode=FailureMode.DETECTION_FAILED,
                                message="det", stage="detection", recoverable=True))
        acc.add(StructuredError(failure_mode=FailureMode.TRACK_LOST,
                                message="trk", stage="tracking", recoverable=True))
        assert len(acc.by_stage("detection")) == 1
        assert len(acc.by_stage("tracking")) == 1

    def test_has_fatal(self):
        acc = ErrorAccumulator()
        acc.add(StructuredError(failure_mode=FailureMode.INGEST_FAILED,
                                message="fatal", stage="ingest", recoverable=False))
        assert acc.has_fatal() is True

    def test_no_fatal_initially(self):
        acc = ErrorAccumulator()
        assert acc.has_fatal() is False

    def test_summary_structure(self):
        acc = ErrorAccumulator()
        acc.add(StructuredError(failure_mode=FailureMode.TRACK_LOST,
                                message="m", stage="tracking", recoverable=True))
        s = acc.summary()
        assert "total" in s
        assert "by_stage" in s
        assert "by_mode" in s


class TestSafeExecute:
    def test_returns_value_on_success(self):
        acc = ErrorAccumulator()
        result = safe_execute(lambda: 42, "test", acc, -1)
        assert result == 42
        assert acc.count() == 0

    def test_returns_fallback_on_exception(self):
        acc = ErrorAccumulator()
        result = safe_execute(lambda: 1/0, "test", acc, -1)
        assert result == -1
        assert acc.count() == 1

    def test_error_contains_stage(self):
        acc = ErrorAccumulator()
        safe_execute(lambda: (_ for _ in ()).throw(ValueError("bad")),
                     "my_stage", acc, None)
        assert acc.all()[0].stage == "my_stage"

    def test_fallback_can_be_empty_list(self):
        acc = ErrorAccumulator()
        result = safe_execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                              "stage", acc, [])
        assert result == []


class TestWrapException:
    def test_wraps_file_not_found(self):
        err = wrap_exception(FileNotFoundError("missing"), "ingest")
        assert err.failure_mode == FailureMode.INGEST_FAILED

    def test_contains_traceback(self):
        try:
            raise ValueError("test")
        except ValueError as e:
            err = wrap_exception(e, "detection")
        assert "traceback" in err.details

    def test_stage_preserved(self):
        err = wrap_exception(RuntimeError("r"), "schema_builder")
        assert err.stage == "schema_builder"


# ── Provenance tests ──────────────────────────────────────────────────────────

class TestProvenanceChain:
    def test_add_step(self):
        chain = ProvenanceChain("type")
        chain.add_step("detection", "contour", 0.7, (0, 10))
        assert len(chain.steps) == 1

    def test_final_confidence(self):
        chain = ProvenanceChain("type")
        chain.add_step("d1", "m1", 0.6)
        chain.add_step("d2", "m2", 0.8)
        assert chain.final_confidence() == 0.8

    def test_origin_module(self):
        chain = ProvenanceChain("layout")
        chain.add_step("detection", "bbox", 0.9)
        chain.add_step("tracking", "median", 0.8)
        assert chain.origin_module() == "detection"

    def test_to_record(self):
        chain = ProvenanceChain("motion")
        chain.add_step("motion_analysis", "curve_fit", 0.75, (0, 30))
        record = chain.to_record()
        assert record.source_module == "motion_analysis"
        assert record.confidence == 0.75


class TestProvenanceRegistry:
    def test_record_and_retrieve(self):
        reg = ProvenanceRegistry()
        reg.record("elem_1", "type", "detection", "contour", 0.8)
        record = reg.get_record("elem_1", "type")
        assert record.confidence == 0.8

    def test_has_provenance(self):
        reg = ProvenanceRegistry()
        assert not reg.has_provenance("e1", "type")
        reg.record("e1", "type", "detection", "method", 0.7)
        assert reg.has_provenance("e1", "type")

    def test_missing_fields(self):
        reg = ProvenanceRegistry()
        reg.record("e1", "type", "detection", "m", 0.7)
        missing = reg.missing_fields("e1", ["type", "layout", "motion"])
        assert "layout" in missing
        assert "motion" in missing
        assert "type" not in missing

    def test_unknown_element_returns_safe_record(self):
        reg = ProvenanceRegistry()
        record = reg.get_record("nonexistent", "type")
        assert record.source_module == "unknown"

    def test_multiple_steps_chain(self):
        reg = ProvenanceRegistry()
        reg.record("e1", "motion", "motion_analysis", "trajectory", 0.6)
        reg.record("e1", "motion", "curve_fitting", "easing_fit", 0.8)
        chain = reg.get_chain("e1", "motion")
        assert len(chain.steps) == 2
        assert chain.final_confidence() == 0.8

    def test_summary(self):
        reg = ProvenanceRegistry()
        reg.record("e1", "type", "d", "m", 0.8)
        reg.record("e2", "layout", "d", "m", 0.7)
        s = reg.summary()
        assert s["total_chains"] == 2


# ── Segmentation tests ────────────────────────────────────────────────────────

class TestSegmentElement:
    def _frame(self):
        import cv2
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        f[:] = (40, 50, 60)
        cv2.rectangle(f, (50, 50), (150, 100), (220, 220, 255), -1)
        return f

    def test_returns_segmentation_result(self):
        from backend.pipeline.segmentation import SegmentationResult
        frame = self._frame()
        bbox = BoundingBox(x=50, y=50, w=100, h=50)
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert isinstance(result, SegmentationResult)

    def test_confidence_in_range(self):
        frame = self._frame()
        bbox = BoundingBox(x=50, y=50, w=100, h=50)
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert 0.0 <= result.mask_confidence <= 1.0

    def test_partial_bbox_detected(self):
        frame = self._frame()
        bbox = BoundingBox(x=280, y=200, w=100, h=100)  # extends past frame
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert result.is_partial is True

    def test_in_bounds_not_partial(self):
        frame = self._frame()
        bbox = BoundingBox(x=50, y=50, w=100, h=50)
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert result.is_partial is False

    def test_boundary_sharpness_in_range(self):
        frame = self._frame()
        bbox = BoundingBox(x=50, y=50, w=100, h=50)
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert 0.0 <= result.boundary_sharpness <= 1.0

    def test_tiny_bbox_returns_result(self):
        frame = self._frame()
        bbox = BoundingBox(x=50, y=50, w=3, h=3)
        result = segment_element(frame, "e1", bbox, use_grabcut=False)
        assert result is not None


class TestSegmentFrameElements:
    def test_returns_dict_and_errors(self):
        import cv2
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = [_make_detection("d1", x=50, y=50, w=100, h=40)]
        results, errors = segment_frame_elements(frame, dets, use_grabcut=False)
        assert isinstance(results, dict)
        assert isinstance(errors, list)

    def test_keys_match_detection_ids(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = [_make_detection("det_a"), _make_detection("det_b", x=150)]
        results, _ = segment_frame_elements(frame, dets, use_grabcut=False)
        assert "det_a" in results
        assert "det_b" in results

    def test_empty_detections_returns_empty(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        results, errors = segment_frame_elements(frame, [], use_grabcut=False)
        assert results == {}
        assert errors == []


# ── Re-ID tests ───────────────────────────────────────────────────────────────

class TestReidentificationEngine:
    def _det(self, eid, x=50, y=80, w=100, h=40, frame=5):
        return _make_detection(eid, x=x, y=y, w=w, h=h, frame=frame)

    def test_attempt_reid_empty_lost(self):
        engine = ReidentificationEngine(320, 240)
        dets = [self._det("d1")]
        candidates, errors = engine.attempt_reid(5, dets)
        assert candidates == []
        assert errors == []

    def test_registers_lost_track(self):
        engine = ReidentificationEngine(320, 240)
        engine.register_lost("trk_1", 0,
                             BoundingBox(x=50, y=80, w=100, h=40),
                             {"aspect_ratio": 2.5, "color_variance": 100.0,
                              "mean_color": [200, 200, 200]},
                             (3.0, 0.0),
                             [TypeCandidate(type=ElementType.TEXT, confidence=0.8)])
        assert engine.get_lost_count() == 1

    def test_reid_candidate_returned_for_close_detection(self):
        engine = ReidentificationEngine(320, 240)
        engine.register_lost("trk_1", 0,
                             BoundingBox(x=50, y=80, w=100, h=40),
                             {"aspect_ratio": 2.5, "color_variance": 100.0,
                              "mean_color": [200, 200, 200],
                              "color_histogram": [0.01]*128},
                             (2.0, 0.0),
                             [TypeCandidate(type=ElementType.TEXT, confidence=0.8)])
        # Detection close to where the lost track would have moved
        det = self._det("d_new", x=62, y=80, w=100, h=40, frame=6)
        det.features = {"aspect_ratio": 2.5, "color_variance": 100.0,
                        "mean_color": [200, 200, 200], "color_histogram": [0.01]*128}
        candidates, _ = engine.attempt_reid(6, [det])
        assert isinstance(candidates, list)

    def test_confirm_reid_removes_from_lost(self):
        engine = ReidentificationEngine(320, 240)
        engine.register_lost("trk_1", 0,
                             BoundingBox(x=50, y=80, w=100, h=40),
                             {}, (0.0, 0.0), [])
        engine.confirm_reid("trk_1")
        assert engine.get_lost_count() == 0

    def test_expired_tracks_cleaned(self):
        engine = ReidentificationEngine(320, 240)
        engine.register_lost("old_track", 0,
                             BoundingBox(x=50, y=80, w=100, h=40),
                             {}, (0.0, 0.0), [])
        from backend.pipeline.reid import MAX_REID_FRAME_GAP
        # Attempt at frame far beyond gap
        engine.attempt_reid(MAX_REID_FRAME_GAP + 5, [])
        assert engine.get_lost_count() == 0
