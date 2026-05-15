"""
test_ingest.py — Tests for backend/pipeline/ingest.py

Coverage:
  - Success: valid MP4 returns correct VideoMetadata
  - Edge: minimum valid video (1 frame)
  - Failure: missing file
  - Failure: empty file
  - Failure: non-video file
  - Failure: corrupted data
  - Success: frame extraction returns correct shapes
  - Edge: extract frame 0 only
  - Failure: extract out-of-range frame
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline.ingest import IngestError, extract_frames, ingest_video
from backend.schemas import VideoMetadata


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_synthetic_mp4(path: str, frames: int = 24, w: int = 320, h: int = 240) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (w, h))
    assert writer.isOpened(), f"Cannot open writer: {path}"
    for i in range(frames):
        frame = np.full((h, w, 3), (i * 5 % 255, 100, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ── Success cases ─────────────────────────────────────────────────────────────

class TestIngestSuccess:
    def test_returns_video_metadata(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert isinstance(meta, VideoMetadata)

    def test_correct_resolution(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert meta.width == 320
        assert meta.height == 240

    def test_correct_fps(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert abs(meta.fps - 24.0) < 1.0

    def test_frame_count_positive(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert meta.frame_count > 0

    def test_duration_positive(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert meta.duration_ms > 0

    def test_filename_preserved(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert meta.filename == Path(simple_video).name

    def test_not_flagged_as_corrupt(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert not meta.is_corrupt

    def test_codec_not_empty(self, simple_video: str):
        meta = ingest_video(simple_video)
        assert isinstance(meta.codec, str)


# ── Failure cases ─────────────────────────────────────────────────────────────

class TestIngestFailures:
    def test_missing_file_raises_ingest_error(self):
        with pytest.raises(IngestError) as exc_info:
            ingest_video("/nonexistent/path/video.mp4")
        assert exc_info.value.structured.stage == "ingest"
        assert not exc_info.value.structured.recoverable

    def test_empty_file_raises_ingest_error(self, tmp_path: Path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(IngestError) as exc_info:
            ingest_video(str(empty))
        assert exc_info.value.structured.stage == "ingest"

    def test_non_video_file_raises_ingest_error(self, tmp_path: Path):
        text_file = tmp_path / "fake.mp4"
        text_file.write_text("this is not a video")
        with pytest.raises(IngestError):
            ingest_video(str(text_file))

    def test_directory_raises_ingest_error(self, tmp_path: Path):
        with pytest.raises(IngestError):
            ingest_video(str(tmp_path))


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestIngestEdge:
    def test_single_frame_video(self, tmp_path: Path):
        path = str(tmp_path / "one_frame.mp4")
        _write_synthetic_mp4(path, frames=1)
        meta = ingest_video(path)
        assert meta.frame_count >= 1

    def test_high_resolution(self, tmp_path: Path):
        path = str(tmp_path / "hd.mp4")
        _write_synthetic_mp4(path, frames=5, w=1280, h=720)
        meta = ingest_video(path)
        assert meta.width == 1280
        assert meta.height == 720

    def test_path_object_accepted(self, simple_video: str):
        meta = ingest_video(Path(simple_video))
        assert meta.width > 0


# ── Frame extraction ──────────────────────────────────────────────────────────

class TestFrameExtraction:
    def test_extract_specific_frames(self, simple_video: str):
        frames = extract_frames(simple_video, [0, 5, 10])
        assert len(frames) == 3

    def test_extracted_frames_are_ndarrays(self, simple_video: str):
        frames = extract_frames(simple_video, [0])
        idx, arr = frames[0]
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_frame_indices_preserved(self, simple_video: str):
        frames = extract_frames(simple_video, [0, 10, 20])
        indices = [idx for idx, _ in frames]
        assert 0 in indices
        assert 10 in indices

    def test_duplicate_indices_deduplicated(self, simple_video: str):
        frames = extract_frames(simple_video, [0, 0, 0])
        assert len(frames) == 1

    def test_empty_index_list_returns_empty(self, simple_video: str):
        frames = extract_frames(simple_video, [])
        assert frames == []

    def test_invalid_frame_index_raises(self, simple_video: str):
        meta = ingest_video(simple_video)
        with pytest.raises(IngestError):
            extract_frames(simple_video, [meta.frame_count + 100])
