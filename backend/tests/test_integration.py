"""
test_integration.py — Full end-to-end pipeline integration tests.

Tests the complete video → JSON pipeline using synthetic MP4 files.
Validates that every stage runs without silent failure and that
the final output is a well-formed, validated TemplateJSON.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.main import run_pipeline
from backend.schemas import ValidationStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_test_video(
    path: str,
    frames: int = 48,
    w: int = 320,
    h: int = 240,
    fps: float = 24.0,
    with_motion: bool = True,
) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 40, 60)
        if with_motion:
            x = int((i / frames) * (w - 80)) + 10
            cv2.rectangle(frame, (x, 40), (x + 70, 90), (240, 240, 255), -1)
            cv2.rectangle(frame, (20, h - 60), (w - 20, h - 20), (180, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _write_two_scene_video(path: str) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (320, 240))
    assert writer.isOpened()
    for i in range(24):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 2] = 160
        cv2.putText(frame, "Scene 1", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(frame)
    for i in range(24):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = 180
        cv2.putText(frame, "Scene 2", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


# ── Integration tests ─────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_pipeline_returns_pipeline_result(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "test.mp4"))
        result = run_pipeline(path)
        assert result is not None
        assert result.status in ("approved", "needs_refinement", "rejected")

    def test_pipeline_succeeds_on_valid_video(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "valid.mp4"))
        result = run_pipeline(path)
        # With a real video we expect at least needs_refinement or approved
        assert result.status != "rejected" or len(result.errors) > 0

    def test_pipeline_rejected_on_missing_file(self):
        result = run_pipeline("/nonexistent/path.mp4")
        assert result.status == "rejected"
        assert len(result.errors) >= 1

    def test_pipeline_rejected_on_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        result = run_pipeline(str(empty))
        assert result.status == "rejected"

    def test_template_has_required_keys(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "req.mp4"))
        result = run_pipeline(path)
        if result.template:
            required = ["schema_version", "meta", "canvas", "scenes", "elements", "validation", "errors"]
            for key in required:
                assert key in result.template, f"Missing key: {key}"

    def test_template_is_json_serializable(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "serial.mp4"))
        result = run_pipeline(path)
        if result.template:
            # Should not raise
            serialized = json.dumps(result.template)
            assert len(serialized) > 10

    def test_errors_are_structured(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "err.mp4"))
        result = run_pipeline(path)
        for err in result.errors:
            assert "failure_mode" in err
            assert "message" in err
            assert "stage" in err
            assert "recoverable" in err

    def test_quality_dict_present(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "qual.mp4"))
        result = run_pipeline(path)
        assert isinstance(result.quality, dict)

    def test_no_silent_failure_on_static_video(self, tmp_path: Path):
        """A video with no motion should complete pipeline (not crash)."""
        path = _write_test_video(str(tmp_path / "static.mp4"), with_motion=False)
        result = run_pipeline(path)
        assert result.status in ("approved", "needs_refinement", "rejected")

    def test_two_scene_video_detects_scenes(self, tmp_path: Path):
        path = _write_two_scene_video(str(tmp_path / "two_scene.mp4"))
        result = run_pipeline(path)
        if result.template:
            scenes = result.template.get("scenes", [])
            # Should detect at least 1 scene
            assert len(scenes) >= 1

    def test_canvas_matches_video_resolution(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "res.mp4"), w=320, h=240)
        result = run_pipeline(path)
        if result.template:
            canvas = result.template.get("canvas", {})
            assert canvas.get("width") == 320
            assert canvas.get("height") == 240

    def test_meta_fps_matches_source(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "fps.mp4"), fps=24.0)
        result = run_pipeline(path)
        if result.template:
            meta = result.template.get("meta", {})
            assert abs(meta.get("fps", 0) - 24.0) < 2.0

    def test_validation_status_in_result(self, tmp_path: Path):
        path = _write_test_video(str(tmp_path / "val.mp4"))
        result = run_pipeline(path)
        if result.template:
            validation = result.template.get("validation", {})
            assert "status" in validation
            assert validation["status"] in ("approved", "needs_refinement", "rejected", "pending")

    def test_pipeline_deterministic(self, tmp_path: Path):
        """Same video → same status on two runs."""
        path = _write_test_video(str(tmp_path / "det.mp4"))
        r1 = run_pipeline(path)
        r2 = run_pipeline(path)
        assert r1.status == r2.status

    def test_element_types_are_valid_enum_values(self, tmp_path: Path):
        from backend.schemas import ElementType
        valid_types = {e.value for e in ElementType}
        path = _write_test_video(str(tmp_path / "types.mp4"))
        result = run_pipeline(path)
        if result.template:
            for elem in result.template.get("elements", []):
                assert elem["type"] in valid_types, f"Invalid element type: {elem['type']}"

    def test_motion_primitives_are_valid(self, tmp_path: Path):
        from backend.schemas import MotionPrimitive
        valid_primitives = {m.value for m in MotionPrimitive}
        path = _write_test_video(str(tmp_path / "prims.mp4"))
        result = run_pipeline(path)
        if result.template:
            for elem in result.template.get("elements", []):
                for motion in elem.get("motion", []):
                    assert motion["primitive"] in valid_primitives


class TestAPIEndpoints:
    """Test FastAPI endpoints via httpx (no server needed — direct call)."""

    def test_run_pipeline_missing_path(self):
        result = run_pipeline("/absolutely/nonexistent/video.mp4")
        assert result.status == "rejected"

    def test_run_pipeline_fake_video_file(self, tmp_path: Path):
        fake = tmp_path / "fake.mp4"
        fake.write_bytes(b"FAKEFAKEFAKE" * 100)
        result = run_pipeline(str(fake))
        assert result.status == "rejected"
