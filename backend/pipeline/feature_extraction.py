"""
feature_extraction.py — Visual feature extraction from detected regions.

PRD §14: Every detection must carry visual, spatial, and temporal features.
This module owns all feature computation so detection.py and tracking.py
do not duplicate or hide feature logic.

Features computed:
  - Color histogram (HSV)
  - Edge density
  - Aspect ratio
  - Color variance
  - Mean color (BGR)
  - Contrast
  - Texture (LBP approximation)
  - Compactness
  - Spatial position (normalized)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Optional
import cv2
import numpy as np


@dataclass
class ElementFeatures:
    """
    All features for one element region.
    Every field is explicitly typed and documented.
    """

    # Geometry
    aspect_ratio: float  # w/h
    area_pixels: float  # w * h
    area_norm: float  # fraction of canvas

    # Color
    mean_color_bgr: list[int]  # [B, G, R]
    color_variance: float  # mean per-channel variance
    dominant_hue: float  # 0–180 (OpenCV HSV scale)
    saturation_mean: float  # 0–255

    # Texture / structure
    edge_density: float  # fraction of edge pixels
    contrast: float  # RMS contrast of grayscale ROI
    compactness: float  # 4π·A/P² (1 = circle, lower = complex)

    # Color histogram (flattened, normalized)
    color_histogram: list[float]  # 64-bin H + 64-bin S = 128 values

    # Spatial (normalized to canvas)
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float

    def to_dict(self) -> dict:
        return {
            "aspect_ratio": self.aspect_ratio,
            "area_pixels": self.area_pixels,
            "area_norm": self.area_norm,
            "mean_color": self.mean_color_bgr,
            "color_variance": self.color_variance,
            "dominant_hue": self.dominant_hue,
            "saturation_mean": self.saturation_mean,
            "edge_density": self.edge_density,
            "contrast": self.contrast,
            "compactness": self.compactness,
            "color_histogram": self.color_histogram,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "w_norm": self.w_norm,
            "h_norm": self.h_norm,
        }


def _safe_roi(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract BGR and grayscale ROI, clamped to frame bounds."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        blank = np.zeros((1, 1, 3), dtype=np.uint8)
        return blank, np.zeros((1, 1), dtype=np.uint8)
    roi_bgr = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return roi_bgr, roi_gray


def _edge_density(roi_gray: np.ndarray) -> float:
    if roi_gray.size == 0:
        return 0.0
    edges = cv2.Canny(roi_gray, 50, 150)
    return float(edges.sum() / 255) / roi_gray.size


def _compactness(roi_gray: np.ndarray) -> float:
    if roi_gray.size == 0:
        return 0.0
    _, thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return min(1.0, float((4 * np.pi * area) / (perimeter**2)))


def _color_histogram(roi_bgr: np.ndarray) -> list[float]:
    """64-bin H + 64-bin S normalized histogram."""
    if roi_bgr.size == 0:
        return [0.0] * 128
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
    combined = np.concatenate([h_hist, s_hist])
    total = combined.sum()
    if total > 0:
        combined /= total
    return combined.tolist()


def _dominant_hue(roi_bgr: np.ndarray) -> float:
    if roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    return float(np.argmax(h_hist))


def _text_content(roi_gray: np.ndarray, roi_bgr: np.ndarray) -> Optional[str]:
    try:
        import pytesseract

        if roi_gray.size == 0:
            return None
        roi_resized = cv2.resize(
            roi_gray, (max(roi_gray.shape[1] * 2, 100), max(roi_gray.shape[0] * 2, 30))
        )
        text = pytesseract.image_to_string(roi_resized, config="--psm 7").strip()
        if text:
            return text
    except Exception:
        pass
    return None


def extract_features(
    frame: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    canvas_width: int,
    canvas_height: int,
) -> ElementFeatures:
    """
    Extract all features for an element region defined by (x, y, w, h).

    Args:
        frame: full BGR frame
        x, y, w, h: bounding box in pixel coordinates
        canvas_width, canvas_height: full frame dimensions
    """
    xi, yi, wi, hi = int(x), int(y), int(w), int(h)
    roi_bgr, roi_gray = _safe_roi(frame, xi, yi, wi, hi)

    # Geometry
    aspect_ratio = float(w / h) if h > 0 else 1.0
    area_pixels = float(w * h)
    canvas_area = canvas_width * canvas_height
    area_norm = area_pixels / canvas_area if canvas_area > 0 else 0.0

    # Color
    if roi_bgr.size > 0:
        mean_bgr = [int(roi_bgr[:, :, c].mean()) for c in range(3)]
        color_variance = float(np.mean([roi_bgr[:, :, c].var() for c in range(3)]))
        sat_mean = float(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        dom_hue = _dominant_hue(roi_bgr)
    else:
        mean_bgr = [0, 0, 0]
        color_variance = 0.0
        sat_mean = 0.0
        dom_hue = 0.0

    # Texture
    edge_dens = _edge_density(roi_gray)
    contrast = float(roi_gray.std()) if roi_gray.size > 0 else 0.0
    compact = _compactness(roi_gray)

    # Histogram
    hist = _color_histogram(roi_bgr)

    # Spatial (normalized)
    x_norm = x / canvas_width if canvas_width > 0 else 0.0
    y_norm = y / canvas_height if canvas_height > 0 else 0.0
    w_norm = w / canvas_width if canvas_width > 0 else 0.0
    h_norm = h / canvas_height if canvas_height > 0 else 0.0

    return ElementFeatures(
        aspect_ratio=round(aspect_ratio, 4),
        area_pixels=round(area_pixels, 2),
        area_norm=round(area_norm, 6),
        mean_color_bgr=mean_bgr,
        color_variance=round(color_variance, 2),
        dominant_hue=round(dom_hue, 2),
        saturation_mean=round(sat_mean, 2),
        edge_density=round(edge_dens, 4),
        contrast=round(contrast, 2),
        compactness=round(compact, 4),
        color_histogram=hist,
        x_norm=round(x_norm, 4),
        y_norm=round(y_norm, 4),
        w_norm=round(w_norm, 4),
        h_norm=round(h_norm, 4),
    )


def histogram_similarity(hist1: list[float], hist2: list[float]) -> float:
    """
    Bhattacharyya distance between two normalized histograms.
    Returns similarity in [0, 1] (1 = identical).
    """
    if len(hist1) != len(hist2) or not hist1:
        return 0.0
    h1 = np.array(hist1, dtype=np.float32).reshape(-1, 1)
    h2 = np.array(hist2, dtype=np.float32).reshape(-1, 1)
    dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - dist))


def feature_distance(f1: dict, f2: dict) -> float:
    """
    Compute a normalized distance (0=identical, 1=completely different)
    between two feature dicts. Used by tracking re-ID.
    """
    scores = []

    # Aspect ratio distance
    ar1 = f1.get("aspect_ratio", 1.0)
    ar2 = f2.get("aspect_ratio", 1.0)
    ar_diff = abs(ar1 - ar2) / max(ar1, ar2, 1e-6)
    scores.append(min(1.0, ar_diff * 2.0))

    # Color variance distance
    cv1 = f1.get("color_variance", 0.0)
    cv2_val = f2.get("color_variance", 0.0)
    denom = max(cv1, cv2_val, 1.0)
    scores.append(min(1.0, abs(cv1 - cv2_val) / denom))

    # Mean color distance
    mc1 = f1.get("mean_color", [128, 128, 128])
    mc2 = f2.get("mean_color", [128, 128, 128])
    if mc1 and mc2:
        color_dist = float(np.linalg.norm(np.array(mc1) - np.array(mc2))) / 441.67
        scores.append(min(1.0, color_dist))

    # Histogram similarity → distance
    h1 = f1.get("color_histogram", [])
    h2 = f2.get("color_histogram", [])
    if h1 and h2:
        hist_sim = histogram_similarity(h1, h2)
        scores.append(1.0 - hist_sim)

    return float(np.mean(scores)) if scores else 0.5
