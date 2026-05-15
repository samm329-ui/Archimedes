"""
motion_analysis.py — Motion extraction with full primitive coverage.

PRD §18 required primitives:
  translateX, translateY, scale, rotate, opacity,
  maskReveal, pathFollow, static

PRD §19 rules:
  - Store raw trajectory AND fitted curve AND confidence
  - Never finalize motion early — multiple hypotheses
  - If fitting confidence is low → keep raw, mark uncertain
  - No silent failure
"""
from __future__ import annotations

import numpy as np

from backend.pipeline.curve_fitting import (
    build_motion_hypotheses,
    detect_opacity_fade,
    detect_rotation,
    smooth_trajectory,
)
from backend.schemas import (
    EasingType,
    FailureMode,
    MotionElement,
    MotionHypothesis,
    MotionPrimitive,
    StructuredError,
    TrackedElement,
)

# ── Thresholds (all explicit) ──────────────────────────────────────────────────

MIN_TRANSLATE_MAGNITUDE = 2.0       # px — below this = not translating
MIN_SCALE_CHANGE_PCT    = 0.04      # 4% area change → scale motion
MIN_ROTATION_AR_RANGE   = 0.08      # aspect-ratio range → rotation hint
MIN_OPACITY_DENSITY_RANGE = 0.015   # edge-density range → opacity fade hint
MIN_PATH_POINTS         = 8         # need this many trajectory points for path
PATH_CURVATURE_THRESHOLD = 0.15     # normalised curvature → non-linear path
MIN_HYPOTHESIS_CONFIDENCE = 0.10
STATIC_CONFIDENCE = 0.92


def _path_curvature(cx: np.ndarray, cy: np.ndarray) -> float:
    """
    Estimate trajectory curvature as mean angular change between steps.
    Returns 0 (straight line) → 1 (highly curved).
    """
    if len(cx) < 3:
        return 0.0
    dx = np.diff(cx)
    dy = np.diff(cy)
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(angles))
    # Wrap to [0, π]
    angle_changes = np.where(angle_changes > np.pi, 2 * np.pi - angle_changes, angle_changes)
    mean_change = float(np.mean(angle_changes))
    return min(1.0, mean_change / np.pi)


def _mask_reveal_score(
    bbox_sequence: list[tuple[int, object]],
) -> float:
    """
    Heuristic: detect mask reveal by checking if element grows from
    one edge consistently in one direction (unlike scale which grows from centre).
    Returns score 0–1.
    """
    if len(bbox_sequence) < 4:
        return 0.0
    sorted_seq = sorted(bbox_sequence, key=lambda x: x[0])
    # Check if width or height grows monotonically while opposite edge is stable
    widths  = np.array([b.w for _, b in sorted_seq])
    heights = np.array([b.h for _, b in sorted_seq])
    x_vals  = np.array([b.x for _, b in sorted_seq])
    y_vals  = np.array([b.y for _, b in sorted_seq])

    # Width grows while x stays ~constant → left-to-right reveal
    w_growth   = float(widths[-1] - widths[0]) / max(widths[0], 1.0)
    x_stable   = float(np.std(x_vals)) < 5.0
    # Height grows while y stays ~constant → top-to-bottom reveal
    h_growth   = float(heights[-1] - heights[0]) / max(heights[0], 1.0)
    y_stable   = float(np.std(y_vals)) < 5.0

    score = 0.0
    if w_growth > 0.15 and x_stable:
        score = max(score, min(1.0, w_growth * 0.8))
    if h_growth > 0.15 and y_stable:
        score = max(score, min(1.0, h_growth * 0.8))
    return score


def extract_motion_from_track(
    track: TrackedElement,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
) -> tuple[MotionElement, list[StructuredError]]:
    """
    Extract all motion hypotheses from a tracked element.

    Analyses all PRD §18 primitives:
      translateX, translateY, scale, rotate,
      opacity, maskReveal, pathFollow, static

    Returns MotionElement with hypotheses sorted by confidence.
    Never raises — all errors are returned as StructuredErrors.
    """
    errors: list[StructuredError] = []

    if not track.bbox_sequence:
        errors.append(StructuredError(
            failure_mode=FailureMode.MOTION_AMBIGUOUS,
            message=f"Track {track.track_id} has empty bbox_sequence",
            stage="motion_analysis",
            recoverable=False,
            details={"track_id": track.track_id},
        ))
        return _empty_motion(track.track_id), errors

    sorted_seq = sorted(track.bbox_sequence, key=lambda x: x[0])
    frames  = np.array([s[0] for s in sorted_seq], dtype=float)
    cx      = np.array([(s[1].x + s[1].w / 2) for s in sorted_seq], dtype=float)
    cy      = np.array([(s[1].y + s[1].h / 2) for s in sorted_seq], dtype=float)
    widths  = np.array([s[1].w for s in sorted_seq], dtype=float)
    heights = np.array([s[1].h for s in sorted_seq], dtype=float)
    areas   = widths * heights

    cx_s    = smooth_trajectory(cx)
    cy_s    = smooth_trajectory(cy)
    areas_s = smooth_trajectory(areas)
    widths_s  = smooth_trajectory(widths)
    heights_s = smooth_trajectory(heights)

    start_frame   = int(frames[0])
    raw_trajectory = [(int(frames[i]), float(cx[i]), float(cy[i])) for i in range(len(frames))]
    cx_range  = float(cx_s.max() - cx_s.min())
    cy_range  = float(cy_s.max() - cy_s.min())
    area_range = float(areas_s.max() - areas_s.min())

    all_hypotheses: list[MotionHypothesis] = []

    # ── STATIC ────────────────────────────────────────────────────────────
    if cx_range < MIN_TRANSLATE_MAGNITUDE and cy_range < MIN_TRANSLATE_MAGNITUDE:
        all_hypotheses.append(MotionHypothesis(
            primitive=MotionPrimitive.STATIC,
            easing=EasingType.LINEAR,
            from_value=float(cx_s[0]),
            to_value=float(cx_s[-1]),
            duration_frames=int(frames[-1] - frames[0]),
            start_frame=start_frame,
            raw_curve=[0.0] * min(len(frames), 10),
            confidence=STATIC_CONFIDENCE,
        ))
        return MotionElement(
            track_id=track.track_id,
            motion_hypotheses=all_hypotheses,
            raw_trajectory=raw_trajectory,
            motion_confidence=STATIC_CONFIDENCE,
        ), errors

    # ── TRANSLATE X ───────────────────────────────────────────────────────
    if cx_range >= MIN_TRANSLATE_MAGNITUDE:
        hyps = build_motion_hypotheses(
            frames, cx_s, MotionPrimitive.TRANSLATE_X,
            start_frame, magnitude_scale=50.0,
        )
        all_hypotheses.extend(hyps)

    # ── TRANSLATE Y ───────────────────────────────────────────────────────
    if cy_range >= MIN_TRANSLATE_MAGNITUDE:
        hyps = build_motion_hypotheses(
            frames, cy_s, MotionPrimitive.TRANSLATE_Y,
            start_frame, magnitude_scale=50.0,
        )
        all_hypotheses.extend(hyps)

    # ── SCALE ─────────────────────────────────────────────────────────────
    area_change_pct = area_range / max(float(areas_s[0]), 1.0)
    if area_change_pct >= MIN_SCALE_CHANGE_PCT:
        hyps = build_motion_hypotheses(
            frames, areas_s, MotionPrimitive.SCALE,
            start_frame, magnitude_scale=float(areas_s[0]) * 0.5,
        )
        all_hypotheses.extend(hyps)

    # ── ROTATE ────────────────────────────────────────────────────────────
    ar_values = widths_s / np.maximum(heights_s, 1.0)
    ar_range  = float(ar_values.max() - ar_values.min())
    if ar_range >= MIN_ROTATION_AR_RANGE:
        rot_hyps = detect_rotation(sorted_seq, frames)
        all_hypotheses.extend(rot_hyps)

    # ── OPACITY ───────────────────────────────────────────────────────────
    # Proxy: use edge-density from features if available
    edge_densities = []
    for frame_idx, _ in sorted_seq:
        feat = track.__dict__.get("_feature_cache", {}).get(frame_idx, {})
        ed = feat.get("edge_density", None)
        if ed is not None:
            edge_densities.append((frame_idx, float(ed)))

    if len(edge_densities) >= 4:
        density_vals = np.array([d for _, d in edge_densities])
        if float(np.ptp(density_vals)) >= MIN_OPACITY_DENSITY_RANGE:
            op_hyps = detect_opacity_fade(edge_densities)
            all_hypotheses.extend(op_hyps)
    else:
        # Fallback: if area shrinks significantly at start/end, infer opacity
        area_start = float(areas_s[:max(1, len(areas_s)//4)].mean())
        area_end   = float(areas_s[-(max(1, len(areas_s)//4)):].mean())
        fade_ratio = abs(area_end - area_start) / max(area_start, 1.0)
        if fade_ratio > 0.25:
            conf = min(0.55, fade_ratio * 0.6)
            if conf >= MIN_HYPOTHESIS_CONFIDENCE:
                all_hypotheses.append(MotionHypothesis(
                    primitive=MotionPrimitive.OPACITY,
                    easing=EasingType.LINEAR,
                    from_value=1.0 if area_end < area_start else 0.0,
                    to_value=0.0 if area_end < area_start else 1.0,
                    duration_frames=int(frames[-1] - frames[0]),
                    start_frame=start_frame,
                    raw_curve=list(np.linspace(
                        1.0 if area_end < area_start else 0.0,
                        0.0 if area_end < area_start else 1.0,
                        min(10, len(frames))
                    )),
                    confidence=round(conf, 4),
                ))

    # ── MASK REVEAL ───────────────────────────────────────────────────────
    mr_score = _mask_reveal_score(sorted_seq)
    if mr_score >= 0.20:
        # Build hypothesis: from_value=0 (hidden) to_value=1 (revealed)
        conf = min(0.75, mr_score * 0.85)
        if conf >= MIN_HYPOTHESIS_CONFIDENCE:
            reveal_vals = widths_s / max(float(widths_s.max()), 1.0)
            hyps = build_motion_hypotheses(
                frames, reveal_vals, MotionPrimitive.MASK_REVEAL,
                start_frame, magnitude_scale=0.5,
            )
            # Override confidence with mr_score-derived value
            for h in hyps:
                h.confidence = min(h.confidence, round(conf, 4))
            all_hypotheses.extend(hyps)

    # ── PATH FOLLOW ───────────────────────────────────────────────────────
    if len(frames) >= MIN_PATH_POINTS:
        curvature = _path_curvature(cx_s, cy_s)
        if curvature >= PATH_CURVATURE_THRESHOLD:
            # Parameterise path by arc length
            dists = np.sqrt(np.diff(cx_s) ** 2 + np.diff(cy_s) ** 2)
            arc_lengths = np.concatenate([[0.0], np.cumsum(dists)])
            total_arc = arc_lengths[-1]
            if total_arc > 10.0:
                arc_norm = arc_lengths / total_arc
                conf = min(0.70, curvature * 0.8)
                if conf >= MIN_HYPOTHESIS_CONFIDENCE:
                    all_hypotheses.append(MotionHypothesis(
                        primitive=MotionPrimitive.PATH_FOLLOW,
                        easing=EasingType.LINEAR,
                        from_value=0.0,
                        to_value=float(total_arc),
                        duration_frames=int(frames[-1] - frames[0]),
                        start_frame=start_frame,
                        raw_curve=list(arc_norm[:20]),
                        confidence=round(conf, 4),
                    ))

    # ── Fallback ───────────────────────────────────────────────────────────
    if not all_hypotheses:
        errors.append(StructuredError(
            failure_mode=FailureMode.MOTION_AMBIGUOUS,
            message=f"No confident motion hypotheses for track {track.track_id}",
            stage="motion_analysis",
            recoverable=True,
            details={"track_id": track.track_id, "cx_range": cx_range, "cy_range": cy_range},
        ))
        norm = list(((cx_s - cx_s.min()) / max(cx_s.max() - cx_s.min(), 1e-6))[:20])
        all_hypotheses.append(MotionHypothesis(
            primitive=MotionPrimitive.UNKNOWN,
            easing=EasingType.UNKNOWN,
            from_value=float(cx_s[0]),
            to_value=float(cx_s[-1]),
            duration_frames=int(frames[-1] - frames[0]),
            start_frame=start_frame,
            raw_curve=norm,
            confidence=0.05,
        ))

    # Sort by confidence descending
    all_hypotheses.sort(key=lambda h: -h.confidence)
    overall_confidence = max(h.confidence for h in all_hypotheses)

    return MotionElement(
        track_id=track.track_id,
        motion_hypotheses=all_hypotheses,
        raw_trajectory=raw_trajectory,
        motion_confidence=round(overall_confidence, 4),
    ), errors


def _empty_motion(track_id: str) -> MotionElement:
    return MotionElement(
        track_id=track_id,
        motion_hypotheses=[MotionHypothesis(
            primitive=MotionPrimitive.UNKNOWN,
            easing=EasingType.UNKNOWN,
            from_value=0.0, to_value=0.0,
            duration_frames=0, start_frame=0,
            raw_curve=[], confidence=0.0,
        )],
        raw_trajectory=[],
        motion_confidence=0.0,
    )


def analyze_all_tracks(
    tracks: list[TrackedElement],
    canvas_width: int = 1920,
    canvas_height: int = 1080,
) -> tuple[list[MotionElement], list[StructuredError]]:
    """Run motion analysis on all tracks."""
    all_errors: list[StructuredError] = []
    motion_elements: list[MotionElement] = []
    for track in tracks:
        me, errors = extract_motion_from_track(track, canvas_width, canvas_height)
        motion_elements.append(me)
        all_errors.extend(errors)
    return motion_elements, all_errors
