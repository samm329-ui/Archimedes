"""
detection.py — Multi-model element detection.

Rules:
  - Never assign final type early
  - Every detection carries multiple TypeCandidates with confidence
  - Supports text heuristics, shape detection, and feature extraction
  - No silent failure — all failures produce StructuredErrors
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from backend.schemas import (
    BoundingBox,
    DetectedElement,
    ElementType,
    FailureMode,
    ProvenanceRecord,
    StructuredError,
    TypeCandidate,
)


# ── Thresholds (explicit) ─────────────────────────────────────────────────────

MIN_CONTOUR_AREA = 200        # px² — ignore noise
TEXT_ASPECT_RATIO_MIN = 0.5   # width/height ratio range for text blocks
TEXT_ASPECT_RATIO_MAX = 15.0
MIN_DETECTION_CONFIDENCE = 0.10


def detect_elements_in_frame(
    frame: np.ndarray,
    frame_index: int,
) -> tuple[list[DetectedElement], list[StructuredError]]:
    """
    Detect candidate elements in a single BGR frame.

    Returns:
        (list[DetectedElement], list[StructuredError])
    """
    errors: list[StructuredError] = []
    elements: list[DetectedElement] = []

    if frame is None or frame.size == 0:
        errors.append(StructuredError(
            failure_mode=FailureMode.DETECTION_FAILED,
            message="Empty or null frame received",
            stage="detection",
            recoverable=False,
            details={"frame_index": frame_index},
        ))
        return elements, errors

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # ── Step 2: Contour-based region detection ─────────────────────────────
    contour_elements, contour_errors = _detect_via_contours(
        frame, gray, frame_index, w, h
    )
    elements.extend(contour_elements)
    errors.extend(contour_errors)

    # ── Step 3: Edge-density text heuristic ───────────────────────────────
    text_elements, text_errors = _detect_text_regions(
        frame, gray, frame_index, w, h
    )
    # Merge text candidates with existing, deduplicate by IoU
    for te in text_elements:
        if not _is_duplicate(te, elements, iou_threshold=0.5):
            elements.append(te)
    errors.extend(text_errors)

    return elements, errors


def _detect_via_contours(
    frame: np.ndarray,
    gray: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
) -> tuple[list[DetectedElement], list[StructuredError]]:
    """Detect elements via edge contours."""
    errors: list[StructuredError] = []
    elements: list[DetectedElement] = []

    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            # Skip full-frame contours (background)
            if cw > width * 0.95 and ch > height * 0.95:
                continue

            bbox = BoundingBox(x=float(x), y=float(y), w=float(cw), h=float(ch))
            type_candidates = _classify_region(frame, gray, bbox, width, height)

            if not type_candidates or all(
                tc.confidence < MIN_DETECTION_CONFIDENCE for tc in type_candidates
            ):
                continue

            features = _extract_features(frame, gray, bbox)
            elem = DetectedElement(
                id=f"det_{frame_index}_{uuid.uuid4().hex[:8]}",
                frame_index=frame_index,
                bbox=bbox,
                type_candidates=type_candidates,
                mask_available=False,
                features=features,
                provenance=ProvenanceRecord(
                    source_module="detection.contours",
                    source_frame_range=(frame_index, frame_index),
                    method="canny_contour",
                    confidence=max(tc.confidence for tc in type_candidates),
                ),
            )
            elements.append(elem)

    except Exception as exc:
        errors.append(StructuredError(
            failure_mode=FailureMode.DETECTION_FAILED,
            message=f"Contour detection error: {exc}",
            stage="detection.contours",
            recoverable=True,
            details={"frame_index": frame_index},
        ))

    return elements, errors


def _detect_text_regions(
    frame: np.ndarray,
    gray: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
) -> tuple[list[DetectedElement], list[StructuredError]]:
    """
    Heuristic text detection using edge density and morphological operations.
    No OCR dependency — structural only.
    """
    errors: list[StructuredError] = []
    elements: list[DetectedElement] = []

    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0:
                continue

            ar = cw / ch
            if not (TEXT_ASPECT_RATIO_MIN <= ar <= TEXT_ASPECT_RATIO_MAX):
                continue

            if cw > width * 0.95:
                continue

            bbox = BoundingBox(x=float(x), y=float(y), w=float(cw), h=float(ch))
            text_conf = _text_confidence(frame, gray, bbox)

            candidates = [
                TypeCandidate(type=ElementType.TEXT, confidence=text_conf),
                TypeCandidate(type=ElementType.SHAPE, confidence=max(0.0, 0.4 - text_conf * 0.3)),
            ]

            features = _extract_features(frame, gray, bbox)
            features["edge_density"] = _edge_density(gray, bbox)

            elem = DetectedElement(
                id=f"det_txt_{frame_index}_{uuid.uuid4().hex[:8]}",
                frame_index=frame_index,
                bbox=bbox,
                type_candidates=candidates,
                mask_available=False,
                features=features,
                provenance=ProvenanceRecord(
                    source_module="detection.text_heuristic",
                    source_frame_range=(frame_index, frame_index),
                    method="morphological_text",
                    confidence=text_conf,
                ),
            )
            elements.append(elem)

    except Exception as exc:
        errors.append(StructuredError(
            failure_mode=FailureMode.TEXT_DETECTION_FAILED,
            message=f"Text detection error: {exc}",
            stage="detection.text",
            recoverable=True,
            details={"frame_index": frame_index},
        ))

    return elements, errors


def _classify_region(
    frame: np.ndarray,
    gray: np.ndarray,
    bbox: BoundingBox,
    width: int,
    height: int,
) -> list[TypeCandidate]:
    """
    Multi-signal classification. Returns scored TypeCandidates.
    Never returns a single certain answer.
    """
    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
    x2, y2 = min(x + w, width), min(y + h, height)

    if x2 <= x or y2 <= y:
        return [TypeCandidate(type=ElementType.UNKNOWN, confidence=0.1)]

    roi_gray = gray[y:y2, x:x2]
    roi_bgr = frame[y:y2, x:x2]

    ar = w / h if h > 0 else 1.0
    edge_dens = _edge_density(gray, bbox)
    color_var = _color_variance(roi_bgr)
    compactness = _contour_compactness(roi_gray)

    # Text score: high edge density, wide aspect ratio, low color variance
    text_score = 0.0
    if TEXT_ASPECT_RATIO_MIN <= ar <= TEXT_ASPECT_RATIO_MAX:
        text_score += 0.3
    if edge_dens > 0.08:
        text_score += 0.3
    if color_var < 2000:
        text_score += 0.2
    text_score = min(1.0, text_score)

    # Shape score: compact, moderate edge density
    shape_score = 0.0
    if compactness > 0.6:
        shape_score += 0.35
    if 0.03 < edge_dens < 0.25:
        shape_score += 0.3
    shape_score = min(1.0, shape_score)

    # Overlay/image: large region, varied colors
    overlay_score = 0.0
    if color_var > 3000 and w * h > 10000:
        overlay_score += 0.4
    overlay_score = min(1.0, overlay_score)

    candidates = [
        TypeCandidate(type=ElementType.TEXT, confidence=text_score),
        TypeCandidate(type=ElementType.SHAPE, confidence=shape_score),
        TypeCandidate(type=ElementType.OVERLAY, confidence=overlay_score),
        TypeCandidate(type=ElementType.UNKNOWN, confidence=0.1),
    ]
    # Filter to non-trivial
    return [c for c in candidates if c.confidence >= MIN_DETECTION_CONFIDENCE]


def _text_confidence(
    frame: np.ndarray,
    gray: np.ndarray,
    bbox: BoundingBox,
) -> float:
    """Compute a text-likelihood score for a bounding box."""
    h_img, w_img = gray.shape[:2]
    x, y = int(bbox.x), int(bbox.y)
    x2 = min(int(bbox.x + bbox.w), w_img)
    y2 = min(int(bbox.y + bbox.h), h_img)

    if x2 <= x or y2 <= y:
        return 0.0

    roi = gray[y:y2, x:x2]
    if roi.size == 0:
        return 0.0

    edge_dens = _edge_density(gray, bbox)
    ar = bbox.w / bbox.h if bbox.h > 0 else 1.0
    std = float(roi.std())

    score = 0.0
    if TEXT_ASPECT_RATIO_MIN <= ar <= TEXT_ASPECT_RATIO_MAX:
        score += 0.35
    if edge_dens > 0.07:
        score += 0.35
    if 20 < std < 100:
        score += 0.2

    return min(1.0, score)


def _edge_density(gray: np.ndarray, bbox: BoundingBox) -> float:
    """Fraction of pixels that are edges within the bounding box."""
    h_img, w_img = gray.shape[:2]
    x, y = int(bbox.x), int(bbox.y)
    x2 = min(int(bbox.x + bbox.w), w_img)
    y2 = min(int(bbox.y + bbox.h), h_img)

    if x2 <= x or y2 <= y:
        return 0.0

    roi = gray[y:y2, x:x2]
    if roi.size == 0:
        return 0.0

    edges = cv2.Canny(roi, 50, 150)
    return float(edges.sum() / 255) / roi.size


def _color_variance(roi: np.ndarray) -> float:
    """Mean per-channel variance."""
    if roi.size == 0:
        return 0.0
    return float(np.mean([roi[:, :, c].var() for c in range(roi.shape[2])]))


def _contour_compactness(roi: np.ndarray) -> float:
    """
    Compactness = 4π·Area / Perimeter².
    Returns 0–1; circle=1, complex shape→0.
    """
    if roi.size == 0:
        return 0.0
    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return min(1.0, (4 * np.pi * area) / (perimeter ** 2))


def _extract_features(
    frame: np.ndarray,
    gray: np.ndarray,
    bbox: BoundingBox,
) -> dict:
    """Extract feature dictionary for an element."""
    h_img, w_img = frame.shape[:2]
    x, y = int(bbox.x), int(bbox.y)
    x2 = min(int(bbox.x + bbox.w), w_img)
    y2 = min(int(bbox.y + bbox.h), h_img)

    features: dict = {
        "aspect_ratio": float(bbox.w / bbox.h) if bbox.h > 0 else 1.0,
        "area": float(bbox.w * bbox.h),
        "edge_density": _edge_density(gray, bbox),
    }

    if x2 > x and y2 > y:
        roi_bgr = frame[y:y2, x:x2]
        features["color_variance"] = _color_variance(roi_bgr)
        # Dominant color (mean BGR)
        features["mean_color"] = [
            int(roi_bgr[:, :, c].mean()) for c in range(3)
        ]

    return features


def _is_duplicate(
    candidate: DetectedElement,
    existing: list[DetectedElement],
    iou_threshold: float = 0.5,
) -> bool:
    """Check if a detection overlaps significantly with an existing one."""
    for elem in existing:
        if elem.frame_index != candidate.frame_index:
            continue
        if elem.bbox.iou(candidate.bbox) >= iou_threshold:
            return True
    return False
