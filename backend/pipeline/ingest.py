"""
ingest.py — Video ingestion, metadata extraction, frame extraction.

PRD §9: FFmpeg is in the tech stack for video decoding.
This module uses FFmpeg as primary frame extractor when available,
falling back to OpenCV. Both paths produce identical output contracts.

Rules:
  - Never silently pass corrupt media downstream
  - All failures produce StructuredError with recoverable flag
  - FFmpeg path: faster, supports more codecs
  - OpenCV path: fallback, always available
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from backend.schemas import FailureMode, StructuredError, VideoMetadata


class IngestError(RuntimeError):
    def __init__(self, error: StructuredError):
        self.structured = error
        super().__init__(error.message)


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ffmpeg_extract_frame(
    video_path: str,
    frame_index: int,
    fps: float,
) -> np.ndarray | None:
    """Extract a single frame via FFmpeg at the exact frame index."""
    timestamp = frame_index / max(fps, 1.0)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-ss", f"{timestamp:.6f}",
                "-i", video_path,
                "-frames:v", "1",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        # Determine frame size from the video
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w <= 0 or h <= 0:
            return None
        expected = w * h * 3
        if len(result.stdout) < expected:
            return None
        frame = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape((h, w, 3))
        return frame.copy()
    except Exception:
        return None


# ── Codec detection ───────────────────────────────────────────────────────────

def _detect_codec(cap: cv2.VideoCapture) -> str:
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    codec = "".join(c for c in chars if c.isalnum())
    return codec if codec else "unknown"


def _probe_corruption(cap: cv2.VideoCapture, sample_count: int = 8) -> bool:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return True
    positions = [int(i * total / sample_count) for i in range(sample_count)]
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return False


def _detect_audio(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_video(video_path: str | Path) -> VideoMetadata:
    """
    Ingest a video file and return canonical VideoMetadata.
    Raises IngestError on any unrecoverable failure.
    """
    path = Path(video_path)

    if not path.exists():
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"File not found: {path}",
            stage="ingest", recoverable=False,
            details={"path": str(path)},
        ))
    if not path.is_file():
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Not a file: {path}",
            stage="ingest", recoverable=False,
        ))
    if path.stat().st_size == 0:
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Empty file: {path}",
            stage="ingest", recoverable=False,
            details={"size_bytes": 0},
        ))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Cannot open with OpenCV: {path}",
            stage="ingest", recoverable=False,
        ))

    try:
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec       = _detect_codec(cap)
        is_corrupt  = _probe_corruption(cap)
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Invalid resolution {width}x{height}",
            stage="ingest", recoverable=False,
        ))
    if fps <= 0:
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Invalid FPS: {fps}",
            stage="ingest", recoverable=False,
        ))
    if frame_count <= 0:
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"No frames: {frame_count}",
            stage="ingest", recoverable=False,
        ))

    return VideoMetadata(
        filename=path.name,
        width=width, height=height,
        fps=float(fps), frame_count=frame_count,
        duration_ms=(frame_count / fps) * 1000.0,
        codec=codec,
        has_audio=_detect_audio(path),
        color_space="bgr",
        is_corrupt=is_corrupt,
    )


def extract_frames(
    video_path: str | Path,
    frame_indices: list[int],
) -> list[tuple[int, np.ndarray]]:
    """
    Extract specific frames by index.

    Strategy:
      1. Try FFmpeg for each frame if available (handles more codecs)
      2. Fall back to OpenCV seek per frame
      3. Any frame that cannot be decoded → IngestError (recoverable=True)

    Returns list of (frame_index, bgr_array) tuples, sorted by index.
    """
    if not frame_indices:
        return []

    path = Path(video_path)
    sorted_unique = sorted(set(frame_indices))
    use_ffmpeg = _ffmpeg_available()

    # Get fps for FFmpeg timestamp calculation
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Cannot open for extraction: {path}",
            stage="ingest.extract_frames", recoverable=False,
        ))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results: list[tuple[int, np.ndarray]] = []

    for idx in sorted_unique:
        if idx >= total_frames:
            cap.release()
            raise IngestError(StructuredError(
                failure_mode=FailureMode.INGEST_FAILED,
                message=f"Frame {idx} out of range (total={total_frames})",
                stage="ingest.extract_frames", recoverable=True,
                details={"frame_index": idx, "total_frames": total_frames},
            ))

        frame = None

        # Try FFmpeg first
        if use_ffmpeg:
            frame = _ffmpeg_extract_frame(str(path), idx, fps)

        # Fall back to OpenCV
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, cv_frame = cap.read()
            if ok and cv_frame is not None and cv_frame.size > 0:
                frame = cv_frame

        if frame is None:
            cap.release()
            raise IngestError(StructuredError(
                failure_mode=FailureMode.INGEST_FAILED,
                message=f"Failed to decode frame {idx}",
                stage="ingest.extract_frames", recoverable=True,
                details={"frame_index": idx, "ffmpeg_tried": use_ffmpeg},
            ))

        results.append((idx, frame))

    cap.release()
    return results
