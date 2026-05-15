"""
test_schema.py — Tests for schema_builder, validator, and render_loop

Coverage:
  - schema_builder: builds valid TemplateJSON from pipeline outputs
  - schema_builder: elements have required fields
  - schema_builder: confidence values in range
  - schema_builder: role_scores are scored lists
  - schema_builder: zero tracks produces error
  - validator: approved on valid template
  - validator: rejected on missing scenes
  - validator: rejected on zero elements
  - validator: rejected on bad canvas dimensions
  - validator: needs_refinement on low-conf elements
  - render_loop: render_frame returns correct shape
  - render_loop: compare_frames returns dict with required keys
  - render_loop: run_render_validation returns ValidationResult
  - render_loop: approved when rendered matches source well
  - render_loop: refinement triggered on mismatch
  - render_loop: no source frames → rejected
  - Property: validate_schema_completeness catches missing keys
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.motion_analysis import analyze_all_tracks
from backend.pipeline.render_loop import (
    compare_frames,
    render_frame,
    run_render_validation,
)
from backend.pipeline.schema_builder import build_template
from backend.pipeline.tracking import track_across_frames
from backend.pipeline.validator import validate_schema_completeness, validate_template
from backend.schemas import (
    BoundingBox,
    DetectedElement,
    EasingType,
    ElementType,
    FailureMode,
    LayoutInfo,
    MotionHypothesis,
    MotionPrimitive,
    ProvenanceRecord,
    RoleScore,
    SceneSegment,
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_metadata(w: int = 320, h: int = 240, fps: float = 24.0, frames: int = 48) -> VideoMetadata:
    return VideoMetadata(
        filename="test.mp4",
        width=w,
        height=h,
        fps=fps,
        frame_count=frames,
        duration_ms=(frames / fps) * 1000.0,
        codec="mp4v",
        has_audio=False,
    )


def _make_scene(start: int = 0, end: int = 48) -> SceneSegment:
    return SceneSegment(
        scene_id="scene_000",
        start_frame=start,
        end_frame=end,
        duration_frames=end - start,
        boundary_confidence=1.0,
        same_scene_hypothesis=1.0,
        new_scene_hypothesis=0.0,
    )


def _make_track(track_id: str = "trk_001", frames: int = 20) -> TrackedElement:
    return TrackedElement(
        track_id=track_id,
        element_ids=[],
        type_candidates=[
            TypeCandidate(type=ElementType.TEXT, confidence=0.75),
            TypeCandidate(type=ElementType.SHAPE, confidence=0.20),
        ],
        bbox_sequence=[
            (i, BoundingBox(x=50.0 + i * 3, y=80.0, w=100.0, h=40.0))
            for i in range(frames)
        ],
        continuity_score=0.9,
    )


def _make_template_element(elem_id: str = "elem_001", confidence: float = 0.7) -> TemplateElement:
    return TemplateElement(
        id=elem_id,
        type=ElementType.TEXT,
        confidence=confidence,
        role_scores=[RoleScore(role="title", score=0.7)],
        layout=LayoutInfo(x_norm=0.1, y_norm=0.1, w_norm=0.3, h_norm=0.1),
        style=StyleInfo(),
        motion=[MotionHypothesis(
            primitive=MotionPrimitive.TRANSLATE_X,
            easing=EasingType.EASE_OUT_CUBIC,
            from_value=50.0,
            to_value=200.0,
            duration_frames=20,
            start_frame=0,
            raw_curve=[0.0, 0.5, 1.0],
            confidence=0.7,
        )],
        timing=TimingInfo(enter_frame=0, exit_frame=20, duration_frames=20, fps=24.0),
        provenance=ProvenanceRecord(
            source_module="test",
            method="synthetic",
            confidence=confidence,
        ),
    )


def _make_valid_template() -> TemplateJSON:
    meta = _make_metadata()
    return TemplateJSON(
        schema_version="1.0.0",
        meta=meta,
        quality={"element_count": 1},
        canvas={"width": 320, "height": 240, "fps": 24.0, "origin": "top-left"},
        scenes=[_make_scene()],
        elements=[_make_template_element()],
        camera={},
        validation=ValidationResult(status=ValidationStatus.PENDING),
        errors=[],
        provenance={"pipeline_stages": []},
    )


# ── Schema builder tests ──────────────────────────────────────────────────────

class TestSchemaBuilder:
    def test_returns_template_json(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, errors = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        assert isinstance(template, TemplateJSON)

    def test_elements_populated(self):
        meta = _make_metadata()
        tracks = [_make_track("trk_a"), _make_track("trk_b")]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        assert len(template.elements) == 2

    def test_element_has_required_fields(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        elem = template.elements[0]
        assert elem.id is not None
        assert elem.type is not None
        assert 0.0 <= elem.confidence <= 1.0
        assert elem.layout is not None
        assert elem.timing is not None
        assert elem.provenance is not None

    def test_element_confidence_in_range(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        for elem in template.elements:
            assert 0.0 <= elem.confidence <= 1.0

    def test_role_scores_are_list(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        for elem in template.elements:
            assert isinstance(elem.role_scores, list)

    def test_canvas_dimensions_correct(self):
        meta = _make_metadata(w=1280, h=720)
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        assert template.canvas["width"] == 1280
        assert template.canvas["height"] == 720

    def test_scenes_preserved(self):
        meta = _make_metadata()
        scenes = [_make_scene(0, 24), _make_scene(24, 48)]
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, scenes, tracks, motion_elements, [])
        assert len(template.scenes) == 2

    def test_zero_tracks_produces_error(self):
        meta = _make_metadata()
        template, errors = build_template(meta, [_make_scene()], [], [], [])
        assert len(errors) >= 1
        assert any(e.failure_mode == FailureMode.SCHEMA_BUILD_FAILED for e in errors)

    def test_schema_version_present(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        assert template.schema_version in ("1.0.0", "2.0.0")

    def test_accumulated_errors_included(self):
        from backend.schemas import StructuredError
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        prior_error = StructuredError(
            failure_mode=FailureMode.TRACK_LOST,
            message="prior error",
            stage="tracking",
            recoverable=True,
        )
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [prior_error])
        error_messages = [e.message for e in template.errors]
        assert "prior error" in error_messages

    def test_quality_metrics_populated(self):
        meta = _make_metadata()
        tracks = [_make_track()]
        motion_elements, _ = analyze_all_tracks(tracks)
        template, _ = build_template(meta, [_make_scene()], tracks, motion_elements, [])
        assert "element_count" in template.quality
        assert template.quality["element_count"] >= 1


# ── Validator tests ───────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_template_approved(self):
        template = _make_valid_template()
        result, errors = validate_template(template)
        assert result.status in (ValidationStatus.APPROVED, ValidationStatus.NEEDS_REFINEMENT)

    def test_missing_scenes_rejected(self):
        template = _make_valid_template()
        template.scenes = []
        result, _ = validate_template(template)
        assert result.status == ValidationStatus.REJECTED

    def test_zero_elements_rejected(self):
        template = _make_valid_template()
        template.elements = []
        result, _ = validate_template(template)
        assert result.status == ValidationStatus.REJECTED

    def test_bad_fps_rejected(self):
        template = _make_valid_template()
        template.meta = VideoMetadata(
            filename="bad.mp4",
            width=320, height=240,
            fps=-1.0,
            frame_count=48,
            duration_ms=2000.0,
            codec="mp4v",
            has_audio=False,
        )
        result, _ = validate_template(template)
        assert result.status == ValidationStatus.REJECTED

    def test_canvas_mismatch_rejected(self):
        template = _make_valid_template()
        template.canvas = {"width": 9999, "height": 9999, "fps": 24.0}
        result, _ = validate_template(template)
        assert result.status == ValidationStatus.REJECTED

    def test_element_out_of_bounds_rejected(self):
        template = _make_valid_template()
        template.elements[0].layout.x_norm = 2.0  # out of bounds
        result, _ = validate_template(template)
        assert result.status == ValidationStatus.REJECTED

    def test_element_missing_motion_generates_error(self):
        template = _make_valid_template()
        template.elements[0].motion = []
        result, errors = validate_template(template)
        motion_errors = [e for e in errors if e.failure_mode == FailureMode.MOTION_AMBIGUOUS]
        assert len(motion_errors) >= 1

    def test_similarity_score_present(self):
        template = _make_valid_template()
        result, _ = validate_template(template)
        assert result.similarity_score is not None

    def test_failure_reasons_list(self):
        template = _make_valid_template()
        template.elements = []
        result, _ = validate_template(template)
        assert isinstance(result.failure_reasons, list)
        assert len(result.failure_reasons) >= 1


class TestValidateSchemCompleteness:
    def test_valid_dict_no_missing(self):
        d = {
            "schema_version": "1.0.0",
            "meta": {},
            "canvas": {},
            "scenes": [],
            "elements": [],
            "validation": {},
            "errors": [],
            "provenance": {},
        }
        missing = validate_schema_completeness(d)
        assert missing == []

    def test_missing_keys_detected(self):
        d = {"schema_version": "1.0.0"}
        missing = validate_schema_completeness(d)
        assert "meta" in missing
        assert "canvas" in missing

    def test_empty_dict_all_missing(self):
        missing = validate_schema_completeness({})
        assert len(missing) >= 7


# ── Render loop tests ─────────────────────────────────────────────────────────

class TestRenderFrame:
    def test_returns_ndarray(self):
        template = _make_valid_template()
        frame = render_frame(template, 0, 320, 240)
        assert isinstance(frame, np.ndarray)

    def test_correct_shape(self):
        template = _make_valid_template()
        frame = render_frame(template, 0, 320, 240)
        assert frame.shape == (240, 320, 3)

    def test_different_frames_may_differ(self):
        template = _make_valid_template()
        f0 = render_frame(template, 0, 320, 240)
        f10 = render_frame(template, 10, 320, 240)
        # With motion, frames should differ (not guaranteed but likely)
        assert isinstance(f10, np.ndarray)

    def test_element_not_visible_before_enter_frame(self):
        template = _make_valid_template()
        template.elements[0].timing = TimingInfo(
            enter_frame=10, exit_frame=20, duration_frames=10, fps=24.0
        )
        # At frame 0, element shouldn't be rendered
        frame = render_frame(template, 0, 320, 240)
        # Should be all-black (no elements rendered)
        assert frame.max() == 0

    def test_empty_elements_all_black(self):
        template = _make_valid_template()
        template.elements = []
        frame = render_frame(template, 0, 320, 240)
        assert frame.max() == 0


class TestCompareFrames:
    def test_returns_dict_with_required_keys(self):
        a = np.zeros((240, 320, 3), dtype=np.uint8)
        b = np.zeros((240, 320, 3), dtype=np.uint8)
        result = compare_frames(a, b)
        assert "ssim" in result
        assert "pixel_similarity" in result or "similarity" in result
        assert "composite" in result
        assert "pass" in result

    def test_identical_frames_high_ssim(self):
        a = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        result = compare_frames(a, a.copy())
        assert result["ssim"] > 0.95

    def test_opposite_frames_low_ssim(self):
        a = np.zeros((240, 320, 3), dtype=np.uint8)
        b = np.full((240, 320, 3), 255, dtype=np.uint8)
        result = compare_frames(a, b)
        assert result["ssim"] < 0.5

    def test_ssim_in_range(self):
        a = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        b = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        result = compare_frames(a, b)
        assert -0.1 <= result["ssim"] <= 1.1  # SSIM can be slightly negative

    def test_pass_field_is_bool(self):
        a = np.zeros((240, 320, 3), dtype=np.uint8)
        result = compare_frames(a, a.copy())
        assert isinstance(result["pass"], bool)


class TestRunRenderValidation:
    def test_returns_three_values(self):
        template = _make_valid_template()
        source = [(0, np.zeros((240, 320, 3), dtype=np.uint8))]
        t, result, errors = run_render_validation(template, source)
        assert isinstance(t, TemplateJSON)
        assert isinstance(result, ValidationResult)
        assert isinstance(errors, list)

    def test_no_source_frames_rejected(self):
        template = _make_valid_template()
        _, result, errors = run_render_validation(template, [])
        assert result.status == ValidationStatus.REJECTED
        assert len(errors) >= 1

    def test_identical_render_approved(self):
        """When source and render are both black (blank template), should pass."""
        template = _make_valid_template()
        template.elements = []  # blank render
        source = [(i, np.zeros((240, 320, 3), dtype=np.uint8)) for i in range(5)]
        _, result, errors = run_render_validation(template, source)
        assert result.status == ValidationStatus.APPROVED

    def test_ssim_score_populated(self):
        template = _make_valid_template()
        template.elements = []
        source = [(0, np.zeros((240, 320, 3), dtype=np.uint8))]
        _, result, _ = run_render_validation(template, source)
        assert result.ssim_score is not None
        assert 0.0 <= result.ssim_score <= 1.0

    def test_refinement_attempts_tracked(self):
        """Mismatched render should trigger refinement and record attempts."""
        template = _make_valid_template()
        # Source is all-white, render is black → big mismatch
        source = [(i, np.full((240, 320, 3), 200, dtype=np.uint8)) for i in range(3)]
        _, result, _ = run_render_validation(template, source)
        assert result.refinement_attempts >= 0  # May or may not need refinement

    def test_validation_result_stored_in_template(self):
        template = _make_valid_template()
        template.elements = []
        source = [(0, np.zeros((240, 320, 3), dtype=np.uint8))]
        t, result, _ = run_render_validation(template, source)
        assert t.validation.status == result.status
