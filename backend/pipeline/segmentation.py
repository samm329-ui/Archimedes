"""
segmentation.py — Element mask generation and segmentation.

PRD §9: SAM (Segment Anything Model) is in the tech stack.
Since SAM requires large model weights not available here,
this module implements a deterministic OpenCV-based segmentation
pipeline that produces element masks, with a clean interface
that can be swapped for SAM when available.

Provides:
  - GrabCut-based foreground segmentation
  - Watershed-based region separation
  - Alpha/transparency detection
  - Mask quality scoring
  - Partial visibility detection (PRD §27 item 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from backend.schemas import BoundingBox, FailureMode, StructuredError


@dataclass
class SegmentationResult:
    """Mask and metadata for one segmented element."""
    element_id: str
    mask: Optional[np.ndarray]        # HxW uint8, 255=foreground, 0=background
    mask_confidence: float            # 0–1
    is_partial: bool                  # Element is partially out of frame
    has_transparency: bool            # Detected alpha/semi-transparency
    boundary_sharpness: float         # 0–1: sharp=foreground element, soft=shadow/blur
    mask_area_fraction: float         # fraction of bbox that is foreground
    failure_mode: Optional[str] = None


def _grabcut_mask(
    frame: np.ndarray,
    bbox: BoundingBox,
    iterations: int = 3,
) -> Optional[np.ndarray]:
    """
    GrabCut foreground segmentation within a bounding box.
    Returns foreground mask (255=fg, 0=bg) or None on failure.
    """
    h, w = frame.shape[:2]
    x = max(0, int(bbox.x))
    y = max(0, int(bbox.y))
    bw = min(int(bbox.w), w - x)
    bh = min(int(bbox.h), h - y)

    if bw < 10 or bh < 10:
        return None

    rect = (x, y, bw, bh)
    try:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, iterations,
                    cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        roi = fg_mask[y:y + bh, x:x + bw]
        return roi if roi.size > 0 else None
    except Exception:
        return None


def _watershed_segments(
    frame: np.ndarray,
    bbox: BoundingBox,
) -> Optional[np.ndarray]:
    """
    Watershed segmentation within a region.
    Returns a binary mask distinguishing dominant region.
    """
    h, w = frame.shape[:2]
    x = max(0, int(bbox.x))
    y = max(0, int(bbox.y))
    bw = min(int(bbox.w), w - x)
    bh = min(int(bbox.h), h - y)

    if bw < 8 or bh < 8:
        return None

    try:
        roi = frame[y:y + bh, x:x + bw].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        return sure_fg
    except Exception:
        return None


def _detect_transparency(frame: np.ndarray, bbox: BoundingBox) -> bool:
    """
    Heuristic: detect if an element appears semi-transparent.
    Checks if the color variance within bbox is unusually high
    relative to the overall frame, suggesting compositing.
    """
    h, w = frame.shape[:2]
    x = max(0, int(bbox.x))
    y = max(0, int(bbox.y))
    x2 = min(w, x + int(bbox.w))
    y2 = min(h, y + int(bbox.h))
    if x2 <= x or y2 <= y:
        return False

    roi = frame[y:y2, x:x2]
    frame_var = float(frame.var())
    roi_var = float(roi.var())
    if frame_var == 0:
        return False
    ratio = roi_var / frame_var
    return ratio > 1.8  # Unusually high variance → possible transparency


def _boundary_sharpness(frame: np.ndarray, bbox: BoundingBox) -> float:
    """
    Measure how sharp the boundary of a bounding box region is.
    Sharp boundary → foreground element; blurry → shadow/blur effect.
    Returns 0–1.
    """
    h, w = frame.shape[:2]
    x = max(0, int(bbox.x))
    y = max(0, int(bbox.y))
    x2 = min(w, x + int(bbox.w))
    y2 = min(h, y + int(bbox.h))
    if x2 <= x or y2 <= y:
        return 0.5

    roi = frame[y:y2, x:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Normalize: ~100 = moderately sharp, >500 = very sharp
    return min(1.0, laplacian_var / 300.0)


def _is_partial(bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
    """Check if element bbox extends beyond frame bounds."""
    return (
        bbox.x < 0 or bbox.y < 0 or
        bbox.x + bbox.w > frame_w or
        bbox.y + bbox.h > frame_h
    )


def segment_element(
    frame: np.ndarray,
    element_id: str,
    bbox: BoundingBox,
    use_grabcut: bool = True,
) -> SegmentationResult:
    """
    Segment a single element in a frame.

    Args:
        frame: BGR frame
        element_id: ID for provenance
        bbox: bounding box of detected element
        use_grabcut: if False, use watershed only (faster, lower quality)

    Returns SegmentationResult with mask and metadata.
    Never raises — returns failure result on error.
    """
    h, w = frame.shape[:2]

    # Check partial visibility
    is_partial = _is_partial(bbox, w, h)

    # Transparency detection
    has_transparency = _detect_transparency(frame, bbox)

    # Boundary sharpness
    sharpness = _boundary_sharpness(frame, bbox)

    # Mask generation
    mask: Optional[np.ndarray] = None
    mask_confidence = 0.0

    if use_grabcut and bbox.w >= 15 and bbox.h >= 15:
        mask = _grabcut_mask(frame, bbox, iterations=3)
        if mask is not None:
            fg_fraction = float(np.sum(mask > 0)) / max(mask.size, 1)
            mask_confidence = 0.5 + 0.3 * sharpness + 0.2 * fg_fraction
    
    if mask is None:
        mask = _watershed_segments(frame, bbox)
        if mask is not None:
            mask_confidence = 0.3 + 0.2 * sharpness

    # Area fraction
    mask_area_fraction = 0.5  # default
    if mask is not None and mask.size > 0:
        mask_area_fraction = float(np.sum(mask > 0)) / mask.size

    # Penalty for partial elements
    if is_partial:
        mask_confidence *= 0.7

    failure = None
    if mask is None:
        failure = "mask_generation_failed"
        mask_confidence = 0.0

    return SegmentationResult(
        element_id=element_id,
        mask=mask,
        mask_confidence=round(min(1.0, mask_confidence), 4),
        is_partial=is_partial,
        has_transparency=has_transparency,
        boundary_sharpness=round(sharpness, 4),
        mask_area_fraction=round(mask_area_fraction, 4),
        failure_mode=failure,
    )


def segment_frame_elements(
    frame: np.ndarray,
    detections: list,   # list[DetectedElement]
    use_grabcut: bool = True,
) -> tuple[dict[str, SegmentationResult], list[StructuredError]]:
    """
    Segment all elements detected in one frame.

    Returns:
        - results: {element_id: SegmentationResult}
        - errors
    """
    results: dict[str, SegmentationResult] = {}
    errors: list[StructuredError] = []

    for det in detections:
        try:
            seg = segment_element(frame, det.id, det.bbox, use_grabcut=use_grabcut)
            results[det.id] = seg

            if seg.failure_mode:
                errors.append(StructuredError(
                    failure_mode=FailureMode.MASK_INCOMPLETE,
                    message=f"Segmentation failed for {det.id}: {seg.failure_mode}",
                    stage="segmentation",
                    recoverable=True,
                    details={"element_id": det.id},
                ))
        except Exception as exc:
            errors.append(StructuredError(
                failure_mode=FailureMode.MASK_INCOMPLETE,
                message=f"Segmentation exception for {det.id}: {exc}",
                stage="segmentation",
                recoverable=True,
                details={"element_id": det.id},
            ))

    return results, errors
