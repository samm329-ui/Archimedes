"""
frame_sampling.py — Adaptive frame sampling engine.

PRD §13 requirements:
  - Sample adaptively, not uniformly only
  - Always include scene boundaries
  - Always include motion spikes
  - Densify frames where motion is rapid
  - Reduce density where scene is static
  - Retain raw frame indexes for reconstruction traceability

Never silently under-samples fast motion — uses optical flow magnitude
to detect spikes and bursts around them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.schemas import FailureMode, SceneSegment, StructuredError


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TARGET_FPS = 12.0          # Target sampling rate in frames/sec
MOTION_SPIKE_THRESHOLD = 8.0       # Mean optical-flow magnitude → motion spike
MOTION_BURST_RADIUS = 3            # Frames around a spike to also include
STATIC_SCENE_REDUCTION = 0.4       # Fraction of frames to keep in static regions
MAX_FRAMES_PER_PIPELINE = 300      # Hard cap to prevent memory overload
MIN_FRAMES_PER_SCENE = 5           # Always sample at least this many frames/scene


@dataclass
class SamplingPlan:
    """
    The complete sampling plan for a video.
    Carries provenance so every sampled index is traceable.
    """
    frame_indices: list[int]
    scene_plans: list[dict]          # per-scene breakdown
    motion_spike_frames: list[int]
    total_available_frames: int
    sampling_rate_used: float
    capped: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)


def _optical_flow_magnitudes(
    video_path: str,
    scene: SceneSegment,
    sample_step: int = 3,
) -> list[tuple[int, float]]:
    """
    Compute mean optical-flow magnitude between consecutive sampled frames
    within a scene. Returns list of (frame_index, magnitude).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    results: list[tuple[int, float]] = []
    prev_gray: np.ndarray | None = None
    prev_idx: int = -1

    try:
        for idx in range(scene.start_frame, scene.end_frame, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(
                cv2.resize(frame, (160, 90)),
                cv2.COLOR_BGR2GRAY,
            )

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=13,
                    iterations=3, poly_n=5, poly_sigma=1.1,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                results.append((idx, float(mag.mean())))

            prev_gray = gray
            prev_idx = idx
    finally:
        cap.release()

    return results


def _detect_motion_spikes(
    magnitudes: list[tuple[int, float]],
    threshold: float = MOTION_SPIKE_THRESHOLD,
) -> list[int]:
    """Return frame indices where motion magnitude exceeds threshold."""
    return [idx for idx, mag in magnitudes if mag >= threshold]


def plan_scene_sampling(
    scene: SceneSegment,
    fps: float,
    target_rate: float = DEFAULT_TARGET_FPS,
    motion_spike_frames: list[int] | None = None,
    is_static: bool = False,
) -> dict:
    """
    Build a sampling plan for a single scene.

    Returns dict with:
      - indices: list[int]
      - reason_map: {frame_index: reason_string}
      - spike_count: int
    """
    indices: set[int] = set()
    reason_map: dict[int, str] = {}

    # Always include boundary frames
    indices.add(scene.start_frame)
    reason_map[scene.start_frame] = "scene_boundary_start"
    last = max(scene.start_frame, scene.end_frame - 1)
    indices.add(last)
    reason_map[last] = "scene_boundary_end"

    # Uniform base sampling
    effective_rate = target_rate * STATIC_SCENE_REDUCTION if is_static else target_rate
    step = max(1, int(fps / effective_rate))
    for i in range(scene.start_frame, scene.end_frame, step):
        indices.add(i)
        if i not in reason_map:
            reason_map[i] = "uniform"

    # Motion spike bursts
    spike_count = 0
    if motion_spike_frames:
        for spike in motion_spike_frames:
            if scene.start_frame <= spike < scene.end_frame:
                spike_count += 1
                for offset in range(-MOTION_BURST_RADIUS, MOTION_BURST_RADIUS + 1):
                    candidate = spike + offset
                    if scene.start_frame <= candidate < scene.end_frame:
                        indices.add(candidate)
                        reason_map[candidate] = f"motion_burst@{spike}"

    # Enforce minimum
    if len(indices) < MIN_FRAMES_PER_SCENE:
        available = list(range(scene.start_frame, scene.end_frame))
        extra_step = max(1, len(available) // MIN_FRAMES_PER_SCENE)
        for i in available[::extra_step]:
            indices.add(i)
            if i not in reason_map:
                reason_map[i] = "min_coverage"

    sorted_indices = sorted(indices)
    return {
        "scene_id": scene.scene_id,
        "indices": sorted_indices,
        "reason_map": reason_map,
        "spike_count": spike_count,
        "is_static": is_static,
    }


def build_sampling_plan(
    video_path: str,
    scenes: list[SceneSegment],
    fps: float,
    target_rate: float = DEFAULT_TARGET_FPS,
) -> SamplingPlan:
    """
    Build the full adaptive sampling plan for an entire video.

    Steps:
      1. For each scene, compute optical flow magnitudes
      2. Detect motion spikes
      3. Build per-scene plan with spike bursts + uniform base
      4. Cap at MAX_FRAMES_PER_PIPELINE

    Returns SamplingPlan — every index is traceable to its reason.
    """
    errors: list[StructuredError] = []
    warnings: list[str] = []
    all_spike_frames: list[int] = []
    scene_plans: list[dict] = []
    all_indices: set[int] = set()

    total_frames = sum(s.duration_frames for s in scenes)

    for scene in scenes:
        # Detect motion spikes within this scene
        try:
            magnitudes = _optical_flow_magnitudes(video_path, scene, sample_step=3)
            spike_frames = _detect_motion_spikes(magnitudes)
            all_spike_frames.extend(spike_frames)
            # Classify as static if 80%+ of motion measurements are below threshold
            low_motion = sum(1 for _, m in magnitudes if m < MOTION_SPIKE_THRESHOLD)
            is_static = len(magnitudes) > 0 and (low_motion / len(magnitudes)) >= 0.8
        except Exception as exc:
            errors.append(StructuredError(
                failure_mode=FailureMode.SCENE_BOUNDARY_UNCERTAIN,
                message=f"Optical flow failed for scene {scene.scene_id}: {exc}",
                stage="frame_sampling",
                recoverable=True,
                details={"scene_id": scene.scene_id},
            ))
            spike_frames = []
            is_static = False

        plan = plan_scene_sampling(
            scene=scene,
            fps=fps,
            target_rate=target_rate,
            motion_spike_frames=spike_frames,
            is_static=is_static,
        )
        scene_plans.append(plan)
        all_indices.update(plan["indices"])

    sorted_indices = sorted(all_indices)

    # ── Hard cap ───────────────────────────────────────────────────────────
    capped = False
    if len(sorted_indices) > MAX_FRAMES_PER_PIPELINE:
        warnings.append(
            f"Sampling plan capped from {len(sorted_indices)} to "
            f"{MAX_FRAMES_PER_PIPELINE} frames"
        )
        step = len(sorted_indices) // MAX_FRAMES_PER_PIPELINE
        sorted_indices = sorted_indices[::step][:MAX_FRAMES_PER_PIPELINE]
        capped = True

    return SamplingPlan(
        frame_indices=sorted_indices,
        scene_plans=scene_plans,
        motion_spike_frames=all_spike_frames,
        total_available_frames=total_frames,
        sampling_rate_used=target_rate,
        capped=capped,
        warnings=warnings,
        errors=errors,
    )
