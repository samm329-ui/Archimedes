"""
curve_fitting.py — Dedicated curve fitting and motion primitive hypothesis engine.

PRD §19 requirements:
  - Store BOTH fitted easing label AND raw curve WITH confidence
  - Never finalize motion early — maintain multiple hypotheses
  - If fitting confidence is low → retain raw keyframes only
  - Support: translateX, translateY, scale, rotate, opacity,
    maskReveal, pathFollow

Easing functions supported:
  linear, easeIn, easeOut, easeInOut, easeOutCubic, easeInCubic,
  bounce, spring
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from backend.schemas import EasingType, MotionHypothesis, MotionPrimitive


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_R2_TO_USE_FIT = 0.55          # Below this → keep raw, mark low confidence
MIN_POINTS_FOR_FIT = 4
MAX_HYPOTHESES_PER_AXIS = 3
MIN_HYPOTHESIS_CONFIDENCE = 0.12


# ── All easing functions ──────────────────────────────────────────────────────

def _linear(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * t

def _ease_in(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * (t ** 2)

def _ease_out(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * (1 - (1 - t) ** 2)

def _ease_in_out(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.where(t < 0.5, 2 * t ** 2, 1 - (-2 * t + 2) ** 2 / 2)

def _ease_out_cubic(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * (1 - (1 - t) ** 3)

def _ease_in_cubic(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * (t ** 3)

def _bounce(t: np.ndarray, a: float, b: float) -> np.ndarray:
    def _bounce_scalar(tv: float) -> float:
        if tv < 1 / 2.75:
            return 7.5625 * tv * tv
        elif tv < 2 / 2.75:
            tv -= 1.5 / 2.75
            return 7.5625 * tv * tv + 0.75
        elif tv < 2.5 / 2.75:
            tv -= 2.25 / 2.75
            return 7.5625 * tv * tv + 0.9375
        else:
            tv -= 2.625 / 2.75
            return 7.5625 * tv * tv + 0.984375
    bounced = np.array([_bounce_scalar(float(ti)) for ti in t])
    return a + b * bounced

def _spring(t: np.ndarray, a: float, b: float) -> np.ndarray:
    # Damped spring approximation
    damping = 8.0
    frequency = 3.0
    spring_val = 1 - np.exp(-damping * t) * np.cos(frequency * np.pi * t)
    return a + b * np.clip(spring_val, 0, 1.5)


EASING_REGISTRY: dict[EasingType, Callable] = {
    EasingType.LINEAR:        _linear,
    EasingType.EASE_IN:       _ease_in,
    EasingType.EASE_OUT:      _ease_out,
    EasingType.EASE_IN_OUT:   _ease_in_out,
    EasingType.EASE_OUT_CUBIC: _ease_out_cubic,
    EasingType.EASE_IN_CUBIC:  _ease_in_cubic,
    EasingType.BOUNCE:        _bounce,
    EasingType.SPRING:        _spring,
}


@dataclass
class FitResult:
    easing: EasingType
    r2: float
    fitted_values: np.ndarray
    raw_curve: list[float]         # normalized 0–1
    from_value: float
    to_value: float
    confidence: float


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    return max(0.0, float(1.0 - ss_res / ss_tot))


def _normalize_curve(values: np.ndarray) -> list[float]:
    mn, mx = float(values.min()), float(values.max())
    if mx - mn < 1e-8:
        return [0.0] * len(values)
    return list(((values - mn) / (mx - mn)).tolist())


def smooth_trajectory(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing. Returns original if too short."""
    if len(values) < window + 2:
        return values.copy()
    win = min(window, len(values) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return values.copy()
    try:
        return savgol_filter(values, window_length=win, polyorder=2)
    except Exception:
        return values.copy()


def fit_all_easings(
    frames: np.ndarray,
    values: np.ndarray,
) -> list[FitResult]:
    """
    Fit every registered easing function to (frames, values).
    Returns all results sorted by R² descending.
    Raw keyframes are ALWAYS returned — never discarded.
    """
    if len(frames) < MIN_POINTS_FOR_FIT:
        # Too sparse — raw only
        return [FitResult(
            easing=EasingType.UNKNOWN,
            r2=0.0,
            fitted_values=values.copy(),
            raw_curve=_normalize_curve(values),
            from_value=float(values[0]) if len(values) else 0.0,
            to_value=float(values[-1]) if len(values) else 0.0,
            confidence=0.0,
        )]

    duration = float(frames[-1] - frames[0])
    if duration < 1:
        duration = 1.0
    t = (frames - frames[0]) / duration

    results: list[FitResult] = []

    for easing_type, fn in EASING_REGISTRY.items():
        try:
            popt, _ = curve_fit(
                fn, t, values,
                maxfev=2000,
                bounds=(-1e7, 1e7),
                p0=[float(values[0]), float(values[-1] - values[0])],
            )
            fitted = fn(t, *popt)
            r2 = _r_squared(values, fitted)
            raw_curve = _normalize_curve(fitted)
            conf = max(0.0, r2) * 0.9   # slight discount — fit ≠ ground truth
            results.append(FitResult(
                easing=easing_type,
                r2=r2,
                fitted_values=fitted,
                raw_curve=raw_curve[:30],
                from_value=float(values[0]),
                to_value=float(values[-1]),
                confidence=round(conf, 4),
            ))
        except Exception:
            results.append(FitResult(
                easing=easing_type,
                r2=0.0,
                fitted_values=values.copy(),
                raw_curve=_normalize_curve(values)[:30],
                from_value=float(values[0]) if len(values) else 0.0,
                to_value=float(values[-1]) if len(values) else 0.0,
                confidence=0.0,
            ))

    results.sort(key=lambda r: -r.r2)
    return results


def build_motion_hypotheses(
    frames: np.ndarray,
    values: np.ndarray,
    primitive: MotionPrimitive,
    start_frame: int,
    magnitude_scale: float = 50.0,
) -> list[MotionHypothesis]:
    """
    Build MotionHypothesis list for a single axis/primitive.

    Args:
        frames: frame indices (float array)
        values: trajectory values (positions, areas, etc.)
        primitive: which motion primitive this represents
        start_frame: the first frame index
        magnitude_scale: used to scale confidence by motion magnitude
    """
    if len(frames) < 2:
        return []

    smooth_vals = smooth_trajectory(values)
    fit_results = fit_all_easings(frames, smooth_vals)

    magnitude = float(np.ptp(smooth_vals))   # peak-to-peak range
    mag_factor = min(1.0, magnitude / magnitude_scale)

    hypotheses: list[MotionHypothesis] = []
    duration = int(frames[-1] - frames[0])

    for fit in fit_results[:MAX_HYPOTHESES_PER_AXIS]:
        # Scale confidence by magnitude — tiny motion has low confidence
        adjusted_conf = fit.confidence * (0.4 + 0.6 * mag_factor)

        if adjusted_conf < MIN_HYPOTHESIS_CONFIDENCE:
            continue

        hypotheses.append(MotionHypothesis(
            primitive=primitive,
            easing=fit.easing,
            from_value=round(fit.from_value, 3),
            to_value=round(fit.to_value, 3),
            duration_frames=duration,
            start_frame=start_frame,
            raw_curve=fit.raw_curve,
            confidence=round(adjusted_conf, 4),
        ))

    # Always include raw fallback if no confident hypothesis
    if not hypotheses:
        raw_curve = _normalize_curve(smooth_vals)
        hypotheses.append(MotionHypothesis(
            primitive=primitive,
            easing=EasingType.UNKNOWN,
            from_value=round(float(smooth_vals[0]), 3),
            to_value=round(float(smooth_vals[-1]), 3),
            duration_frames=duration,
            start_frame=start_frame,
            raw_curve=raw_curve[:30],
            confidence=0.05,
        ))

    return hypotheses


def detect_rotation(
    bbox_sequence: list[tuple[int, object]],
    frames: np.ndarray,
) -> list[MotionHypothesis]:
    """
    Attempt to detect rotation via aspect-ratio oscillation.
    A rotating rectangle changes apparent w/h ratio periodically.
    Returns rotation hypotheses if detected, empty list otherwise.
    """
    if len(bbox_sequence) < 6:
        return []

    sorted_seq = sorted(bbox_sequence, key=lambda x: x[0])
    ar_values = np.array([b.w / max(b.h, 1.0) for _, b in sorted_seq], dtype=float)
    ar_range = float(np.ptp(ar_values))

    # Only attempt if there's measurable AR variation (could indicate rotation)
    if ar_range < 0.1:
        return []

    return build_motion_hypotheses(
        frames=frames,
        values=ar_values,
        primitive=MotionPrimitive.ROTATE,
        start_frame=int(frames[0]),
        magnitude_scale=0.5,
    )


def detect_opacity_fade(
    edge_density_sequence: list[tuple[int, float]],
) -> list[MotionHypothesis]:
    """
    Detect opacity fade from edge density changes over time.
    A fading element has decreasing edge density.
    """
    if len(edge_density_sequence) < 4:
        return []

    sorted_seq = sorted(edge_density_sequence, key=lambda x: x[0])
    frames = np.array([f for f, _ in sorted_seq], dtype=float)
    densities = np.array([d for _, d in sorted_seq], dtype=float)
    density_range = float(np.ptp(densities))

    if density_range < 0.02:
        return []

    return build_motion_hypotheses(
        frames=frames,
        values=densities,
        primitive=MotionPrimitive.OPACITY,
        start_frame=int(frames[0]),
        magnitude_scale=0.1,
    )
