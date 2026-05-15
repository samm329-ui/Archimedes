"""
test_metadata.py — Tests for backend/pipeline/metadata.py
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pytest
from backend.pipeline.metadata import (
    extract_metadata, _classify_aspect_ratio, ExtendedMetadata
)


def _write_video(path: str, w=320, h=240, fps=24.0, frames=24):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(frames):
        wr.write(np.zeros((h, w, 3), dtype=np.uint8))
    wr.release()
    return path


class TestMetadataSuccess:
    def test_returns_extended_metadata(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        assert isinstance(result, ExtendedMetadata)

    def test_video_sub_object_present(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        assert result.video is not None
        assert result.video.width == 320
        assert result.video.height == 240

    def test_aspect_ratio_label_set(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        assert isinstance(result.aspect_ratio_label, str)
        assert len(result.aspect_ratio_label) > 0

    def test_aspect_ratio_value_positive(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        assert result.aspect_ratio_value > 0

    def test_no_fatal_errors_on_valid_file(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        fatal = [e for e in result.errors if not e.recoverable]
        assert len(fatal) == 0

    def test_16x9_classified_correctly(self, tmp_path):
        p = _write_video(str(tmp_path / "hd.mp4"), w=1280, h=720)
        result = extract_metadata(p)
        assert result.aspect_ratio_label == "16:9"

    def test_container_format_set(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        result = extract_metadata(p)
        assert isinstance(result.container_format, str)


class TestMetadataFailure:
    def test_missing_file_returns_error(self):
        result = extract_metadata("/nonexistent/file.mp4")
        assert len(result.errors) >= 1
        assert not result.errors[0].recoverable

    def test_empty_file_returns_error(self, tmp_path):
        p = tmp_path / "empty.mp4"
        p.write_bytes(b"")
        result = extract_metadata(str(p))
        assert len(result.errors) >= 1

    def test_invalid_video_data_returns_error(self, tmp_path):
        p = tmp_path / "fake.mp4"
        p.write_bytes(b"NOT A VIDEO FILE" * 100)
        result = extract_metadata(str(p))
        assert len(result.errors) >= 1


class TestAspectRatioClassifier:
    def test_16x9(self):
        label, val = _classify_aspect_ratio(1920, 1080)
        assert label == "16:9"
        assert abs(val - 1920/1080) < 0.01

    def test_4x3(self):
        label, _ = _classify_aspect_ratio(640, 480)
        assert label == "4:3"

    def test_1x1(self):
        label, _ = _classify_aspect_ratio(500, 500)
        assert label == "1:1"

    def test_9x16_vertical(self):
        label, _ = _classify_aspect_ratio(1080, 1920)
        assert label == "9:16"

    def test_custom_ratio(self):
        label, _ = _classify_aspect_ratio(333, 100)
        assert "custom" in label

    def test_zero_height_safe(self):
        label, val = _classify_aspect_ratio(100, 0)
        assert label == "unknown"
        assert val == 0.0
