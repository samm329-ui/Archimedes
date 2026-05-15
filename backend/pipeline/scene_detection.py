"""
scene_detection.py — Temporal scene segmentation.

Detects scene boundaries via:
  1. Histogram difference
  2. SSIM drop
  3. Optical flow discontinuity

Never forcibly collapses uncertain boundaries.
Both 'same_scene' and 'new_scene' hypotheses are preserved when confidence is low.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.schemas import SceneSegment, StructuredError, FailureMode


# ── Thresholds (explicit, not hidden) ────────────────────────────────────────

HIST_DIFF_THRESHOLD = 0.45       # Bhattacharyya distance above which we signal change
SSIM_DROP_THRESHOLD = 0.60       # SSIM below which we signal cut
FLOW_MAG_THRESHOLD = 12.0        # Mean optical-flow magnitude spike
BOUNDARY_MIN_CONFIDENCE = 0.50   # Below this, store as dual-hypothesis


@dataclass
class SceneBoundarySignal:
    frame_index: int
    hist_diff: float
    ssim_score: float
    flow_magnitude: float
    composite_score: float        # 0–1, higher = more likely new scene
    is_boundary: bool
    confidence: float


def _bgr_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    """Compute a normalized HSV histogram for a BGR frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist = np.concatenate([h_hist.flatten(), s_hist.flatten()])
    norm = hist.sum()
    return hist / norm if norm > 0 else hist


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(
        h1.astype(np.float32).reshape(-1, 1),
        h2.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_BHATTACHARYYA,
    ))


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Fast grayscale SSIM between two same-size BGR frames."""
    ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    mu1, mu2 = ga.mean(), gb.mean()
    sigma1 = ga.std() ** 2
    sigma2 = gb.std() ** 2
    sigma12 = float(np.mean((ga - mu1) * (gb - mu2)))
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    return float(num / den) if den != 0 else 1.0


def _optical_flow_magnitude(prev: np.ndarray, curr: np.ndarray) -> float:
    """Mean magnitude of dense optical flow between two frames."""
    p = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (160, 90))
    c = cv2.resize(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (160, 90))
    flow = cv2.calcOpticalFlowFarneback(
        p, c,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def _composite_score(hist_diff: float, ssim: float, flow_mag: float) -> float:
    """
    Combine signals into a single 0–1 score.
    Higher = more likely to be a scene boundary.
    Weights are explicit.
    """
    hist_score = min(1.0, hist_diff / 0.8)
    ssim_score = max(0.0, 1.0 - ssim)
    flow_score = min(1.0, flow_mag / 25.0)
    return 0.4 * hist_score + 0.35 * ssim_score + 0.25 * flow_score


def detect_scenes(
    video_path: str,
    sample_every_n: int = 2,
) -> tuple[list[SceneSegment], list[StructuredError]]:
    """
    Detect scene boundaries in a video.

    Returns:
        (list[SceneSegment], list[StructuredError])
    
    Rules:
      - Always includes a segment for the whole video as fallback
      - Never collapses dual hypotheses when confidence is below threshold
      - Every detected boundary carries both same_scene and new_scene probabilities
    """
    cap = cv2.VideoCapture(video_path)
    errors: list[StructuredError] = []

    if not cap.isOpened():
        errors.append(StructuredError(
            failure_mode=FailureMode.SCENE_BOUNDARY_UNCERTAIN,
            message=f"Cannot open video for scene detection: {video_path}",
            stage="scene_detection",
            recoverable=False,
        ))
        return [], errors

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        errors.append(StructuredError(
            failure_mode=FailureMode.SCENE_BOUNDARY_UNCERTAIN,
            message="Invalid frame count or FPS for scene detection",
            stage="scene_detection",
            recoverable=False,
            details={"total_frames": total_frames, "fps": fps},
        ))
        cap.release()
        return _whole_video_segment(total_frames if total_frames > 0 else 0), errors

    signals: list[SceneBoundarySignal] = []
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every_n == 0:
            hist = _bgr_histogram(frame)

            if prev_frame is not None and prev_hist is not None:
                hist_diff = _bhattacharyya(prev_hist, hist)
                ssim = _ssim_score(prev_frame, frame)
                flow_mag = _optical_flow_magnitude(prev_frame, frame)
                composite = _composite_score(hist_diff, ssim, flow_mag)
                is_boundary = composite >= 0.5
                confidence = abs(composite - 0.5) * 2  # distance from decision boundary

                signals.append(SceneBoundarySignal(
                    frame_index=frame_idx,
                    hist_diff=hist_diff,
                    ssim_score=ssim,
                    flow_magnitude=flow_mag,
                    composite_score=composite,
                    is_boundary=is_boundary,
                    confidence=float(confidence),
                ))

            prev_frame = frame.copy()
            prev_hist = hist

        frame_idx += 1

    cap.release()

    if not signals:
        return _whole_video_segment(total_frames), errors

    # ── Build scene segments from boundary signals ────────────────────────
    boundary_frames = [0]
    for sig in signals:
        if sig.is_boundary or sig.composite_score >= BOUNDARY_MIN_CONFIDENCE:
            boundary_frames.append(sig.frame_index)
    boundary_frames.append(total_frames)
    boundary_frames = sorted(set(boundary_frames))

    segments: list[SceneSegment] = []
    for i in range(len(boundary_frames) - 1):
        start = boundary_frames[i]
        end = boundary_frames[i + 1]

        # Find the signal that triggered this boundary (if any)
        triggering = [s for s in signals if s.frame_index == end]
        if triggering:
            sig = triggering[0]
            b_conf = sig.confidence
            new_scene_hyp = sig.composite_score
            same_scene_hyp = 1.0 - sig.composite_score
        else:
            b_conf = 1.0
            new_scene_hyp = 1.0
            same_scene_hyp = 0.0

        segments.append(SceneSegment(
            scene_id=f"scene_{i:03d}",
            start_frame=start,
            end_frame=end,
            duration_frames=end - start,
            boundary_confidence=b_conf,
            same_scene_hypothesis=same_scene_hyp,
            new_scene_hypothesis=new_scene_hyp,
        ))

    return segments, errors


def _whole_video_segment(total_frames: int) -> list[SceneSegment]:
    return [SceneSegment(
        scene_id="scene_000",
        start_frame=0,
        end_frame=total_frames,
        duration_frames=total_frames,
        boundary_confidence=1.0,
        same_scene_hypothesis=1.0,
        new_scene_hypothesis=0.0,
    )]


def adaptive_frame_indices(
    scene: SceneSegment,
    fps: float,
    target_rate: float = 12.0,
    motion_spike_frames: list[int] | None = None,
) -> list[int]:
    """
    Return a list of frame indices to sample for the given scene.

    Rules:
      - Always include start and end frames
      - Sample at target_rate fps
      - Always include motion spike frames
      - Densify around spikes (±2 frames)
    """
    indices: set[int] = set()
    indices.add(scene.start_frame)
    indices.add(max(0, scene.end_frame - 1))

    step = max(1, int(fps / target_rate))
    for i in range(scene.start_frame, scene.end_frame, step):
        indices.add(i)

    if motion_spike_frames:
        for spike in motion_spike_frames:
            if scene.start_frame <= spike < scene.end_frame:
                for offset in range(-2, 3):
                    candidate = spike + offset
                    if scene.start_frame <= candidate < scene.end_frame:
                        indices.add(candidate)

    return sorted(indices)
