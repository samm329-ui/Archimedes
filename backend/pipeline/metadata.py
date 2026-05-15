"""
metadata.py — Dedicated metadata extraction and normalization.

PRD §11: Every uploaded video must be normalized and validated before
deeper analysis. This module owns that contract entirely.

Extracts:
  - container format, codec
  - resolution, aspect ratio
  - fps, duration, frame count
  - audio presence
  - color space
  - corruption / decoding flags
  - aspect ratio classification
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from backend.schemas import FailureMode, StructuredError, VideoMetadata


# ── Aspect ratio classes ──────────────────────────────────────────────────────

ASPECT_RATIO_CLASSES = {
    "1:1":   (0.95, 1.05),
    "4:3":   (1.28, 1.38),
    "16:9":  (1.73, 1.82),
    "9:16":  (0.54, 0.58),
    "21:9":  (2.30, 2.40),
    "4:5":   (0.77, 0.83),
}

# Extreme resolutions that may cause pipeline instability
MAX_DIMENSION = 7680     # 8K
MIN_DIMENSION = 32


@dataclass
class ExtendedMetadata:
    """Full metadata including fields beyond VideoMetadata schema."""
    video: VideoMetadata
    aspect_ratio_label: str
    aspect_ratio_value: float
    container_format: str
    bit_rate_kbps: Optional[float]
    color_primaries: str
    has_variable_fps: bool
    total_frames_reliable: bool
    ffprobe_available: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)


def _classify_aspect_ratio(w: int, h: int) -> tuple[str, float]:
    if h == 0:
        return "unknown", 0.0
    ratio = w / h
    for label, (lo, hi) in ASPECT_RATIO_CLASSES.items():
        if lo <= ratio <= hi:
            return label, round(ratio, 3)
    return f"custom:{w}:{h}", round(ratio, 3)


def _probe_with_ffprobe(path: str) -> dict:
    """
    Run ffprobe to get stream-level metadata.
    Returns empty dict if ffprobe unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return {}


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


def _has_audio_stream(ffprobe_data: dict) -> bool:
    streams = ffprobe_data.get("streams", [])
    return any(s.get("codec_type") == "audio" for s in streams)


def _get_bit_rate(ffprobe_data: dict) -> Optional[float]:
    fmt = ffprobe_data.get("format", {})
    try:
        return float(fmt.get("bit_rate", 0)) / 1000.0
    except (TypeError, ValueError):
        return None


def _get_color_primaries(ffprobe_data: dict) -> str:
    streams = ffprobe_data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            return s.get("color_primaries", "unknown")
    return "unknown"


def _detect_variable_fps(ffprobe_data: dict) -> bool:
    streams = ffprobe_data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            r_frame_rate = s.get("r_frame_rate", "0/1")
            avg_frame_rate = s.get("avg_frame_rate", "0/1")
            return r_frame_rate != avg_frame_rate
    return False


def extract_metadata(video_path: str | Path) -> ExtendedMetadata:
    """
    Extract complete metadata from a video file.

    Returns ExtendedMetadata with the VideoMetadata sub-object and all
    extended fields needed by downstream pipeline stages.

    Never raises — always returns with errors embedded.
    """
    path = Path(video_path)
    warnings: list[str] = []
    errors: list[StructuredError] = []

    # ── Existence / size check ─────────────────────────────────────────────
    if not path.exists() or not path.is_file():
        err = StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"File not found or not a file: {path}",
            stage="metadata",
            recoverable=False,
        )
        return _empty_metadata(str(path.name), errors=[err])

    if path.stat().st_size == 0:
        err = StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"File is empty: {path}",
            stage="metadata",
            recoverable=False,
        )
        return _empty_metadata(str(path.name), errors=[err])

    # ── FFprobe pass (optional enrichment) ────────────────────────────────
    ffprobe_data = _probe_with_ffprobe(str(path))
    ffprobe_available = bool(ffprobe_data)

    # ── OpenCV pass (authoritative for frame-level data) ──────────────────
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        err = StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"OpenCV cannot open: {path}",
            stage="metadata",
            recoverable=False,
        )
        return _empty_metadata(str(path.name), errors=[err])

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _detect_codec(cap)
        is_corrupt = _probe_corruption(cap)
    finally:
        cap.release()

    # ── Sanity checks ──────────────────────────────────────────────────────
    if width <= 0 or height <= 0:
        errors.append(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Invalid resolution: {width}x{height}",
            stage="metadata",
            recoverable=False,
        ))
    if fps <= 0:
        errors.append(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"Invalid FPS: {fps}",
            stage="metadata",
            recoverable=False,
        ))
    if frame_count <= 0:
        errors.append(StructuredError(
            failure_mode=FailureMode.INGEST_FAILED,
            message=f"No frames: {frame_count}",
            stage="metadata",
            recoverable=False,
        ))

    if errors:
        return _empty_metadata(str(path.name), errors=errors)

    # ── Dimension warnings ─────────────────────────────────────────────────
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        warnings.append(f"Extreme resolution {width}x{height} may slow pipeline")
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        warnings.append(f"Very small resolution {width}x{height}")

    # ── Frame count reliability ────────────────────────────────────────────
    # OpenCV frame count can be unreliable for variable-fps or damaged files
    total_frames_reliable = (frame_count > 0 and not is_corrupt)

    duration_ms = (frame_count / fps) * 1000.0
    ar_label, ar_value = _classify_aspect_ratio(width, height)

    video_meta = VideoMetadata(
        filename=path.name,
        width=width,
        height=height,
        fps=float(fps),
        frame_count=frame_count,
        duration_ms=duration_ms,
        codec=codec,
        has_audio=_has_audio_stream(ffprobe_data) if ffprobe_available else False,
        color_space=_get_color_primaries(ffprobe_data) if ffprobe_available else "bgr",
        is_corrupt=is_corrupt,
    )

    container_format = path.suffix.lstrip(".").lower()
    if ffprobe_available:
        fmt_name = ffprobe_data.get("format", {}).get("format_name", "")
        if fmt_name:
            container_format = fmt_name.split(",")[0]

    return ExtendedMetadata(
        video=video_meta,
        aspect_ratio_label=ar_label,
        aspect_ratio_value=ar_value,
        container_format=container_format,
        bit_rate_kbps=_get_bit_rate(ffprobe_data) if ffprobe_available else None,
        color_primaries=_get_color_primaries(ffprobe_data) if ffprobe_available else "unknown",
        has_variable_fps=_detect_variable_fps(ffprobe_data) if ffprobe_available else False,
        total_frames_reliable=total_frames_reliable,
        ffprobe_available=ffprobe_available,
        warnings=warnings,
        errors=errors,
    )


def _empty_metadata(filename: str, errors: list[StructuredError]) -> ExtendedMetadata:
    return ExtendedMetadata(
        video=VideoMetadata(
            filename=filename,
            width=0, height=0, fps=0.0,
            frame_count=0, duration_ms=0.0,
            codec="unknown", has_audio=False,
            is_corrupt=True,
        ),
        aspect_ratio_label="unknown",
        aspect_ratio_value=0.0,
        container_format="unknown",
        bit_rate_kbps=None,
        color_primaries="unknown",
        has_variable_fps=False,
        total_frames_reliable=False,
        ffprobe_available=False,
        errors=errors,
    )
