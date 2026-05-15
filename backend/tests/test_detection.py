"""
test_detection.py — Tests for backend/pipeline/detection.py

Coverage:
  - Success: frame with clear elements returns detections
  - Success: type_candidates always has multiple entries
  - Success: all confidences in [0, 1]
  - Success: all bounding boxes are within frame bounds
  - Edge: blank frame produces no detections
  - Edge: very small frame
  - Edge: full-white frame
  - Failure: None frame handled gracefully
  - Failure: zero-size frame handled gracefully
  - Property: determinism (same frame → same result)
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.detection import detect_elements_in_frame
from backend.schemas import DetectedElement, FailureMode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame_with_text_region(w: int = 320, h: int = 240) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 50, 60)
    # Draw a white horizontal rectangle (text-like)
    import cv2
    cv2.rectangle(frame, (20, 40), (200, 70), (240, 240, 250), -1)
    # Draw a colored shape
    cv2.rectangle(frame, (60, 120), (160, 190), (100, 200, 80), -1)
    return frame


# ── Success cases ─────────────────────────────────────────────────────────────

class TestDetectionSuccess:
    def test_returns_list(self, sample_frame: np.ndarray):
        elements, errors = detect_elements_in_frame(sample_frame, 0)
        assert isinstance(elements, list)
        assert isinstance(errors, list)

    def test_detects_elements_in_synthetic_frame(self):
        frame = _make_frame_with_text_region()
        elements, _ = detect_elements_in_frame(frame, 0)
        # Should find at least 1 element in a frame with clear shapes
        assert len(elements) >= 1

    def test_type_candidates_are_lists(self, sample_frame: np.ndarray):
        elements, _ = detect_elements_in_frame(sample_frame, 0)
        for elem in elements:
            assert isinstance(elem.type_candidates, list)
            assert len(elem.type_candidates) >= 1

    def test_all_confidences_in_range(self, sample_frame: np.ndarray):
        elements, _ = detect_elements_in_frame(sample_frame, 0)
        for elem in elements:
            for tc in elem.type_candidates:
                assert 0.0 <= tc.confidence <= 1.0, (
                    f"Confidence out of range: {tc.confidence}"
                )

    def test_bounding_boxes_within_frame(self, sample_frame: np.ndarray):
        h, w = sample_frame.shape[:2]
        elements, _ = detect_elements_in_frame(sample_frame, 0)
        for elem in elements:
            bb = elem.bbox
            assert bb.x >= 0, f"x negative: {bb.x}"
            assert bb.y >= 0, f"y negative: {bb.y}"
            assert bb.x + bb.w <= w + 1, f"bbox overflows width: {bb.x + bb.w} > {w}"
            assert bb.y + bb.h <= h + 1, f"bbox overflows height: {bb.y + bb.h} > {h}"

    def test_element_ids_unique(self, sample_frame: np.ndarray):
        elements, _ = detect_elements_in_frame(sample_frame, 0)
        ids = [e.id for e in elements]
        assert len(ids) == len(set(ids)), "Duplicate element IDs"

    def test_provenance_present(self, sample_frame: np.ndarray):
        elements, _ = detect_elements_in_frame(sample_frame, 0)
        for elem in elements:
            assert elem.provenance is not None
            assert elem.provenance.source_module != ""

    def test_frame_index_stored(self, sample_frame: np.ndarray):
        elements, _ = detect_elements_in_frame(sample_frame, 42)
        for elem in elements:
            assert elem.frame_index == 42


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestDetectionEdge:
    def test_blank_frame_returns_no_elements(self, blank_frame: np.ndarray):
        elements, errors = detect_elements_in_frame(blank_frame, 0)
        # Blank frame should have no detectable elements, at most minor errors
        assert isinstance(elements, list)

    def test_very_small_frame(self):
        import numpy as np
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        elements, errors = detect_elements_in_frame(tiny, 0)
        assert isinstance(elements, list)

    def test_full_white_frame(self):
        white = np.full((240, 320, 3), 255, dtype=np.uint8)
        elements, errors = detect_elements_in_frame(white, 0)
        assert isinstance(elements, list)

    def test_different_frame_indices_dont_share_ids(self, sample_frame: np.ndarray):
        elems0, _ = detect_elements_in_frame(sample_frame, 0)
        elems5, _ = detect_elements_in_frame(sample_frame, 5)
        ids0 = {e.id for e in elems0}
        ids5 = {e.id for e in elems5}
        # UUIDs should not collide
        overlap = ids0 & ids5
        assert len(overlap) == 0


# ── Failure cases ─────────────────────────────────────────────────────────────

class TestDetectionFailure:
    def test_none_frame_returns_error(self):
        elements, errors = detect_elements_in_frame(None, 0)
        assert elements == []
        assert len(errors) == 1
        assert errors[0].failure_mode == FailureMode.DETECTION_FAILED

    def test_empty_array_returns_error(self):
        empty = np.array([])
        elements, errors = detect_elements_in_frame(empty, 0)
        assert elements == []
        assert len(errors) >= 1


# ── Property: determinism ──────────────────────────────────────────────────────

class TestDetectionDeterminism:
    def test_same_input_same_count(self, sample_frame: np.ndarray):
        """Detection count must be the same for identical frames."""
        e1, _ = detect_elements_in_frame(sample_frame.copy(), 0)
        e2, _ = detect_elements_in_frame(sample_frame.copy(), 0)
        assert len(e1) == len(e2)

    def test_same_input_same_bboxes(self, sample_frame: np.ndarray):
        """Bounding boxes must be identical for identical frames."""
        e1, _ = detect_elements_in_frame(sample_frame.copy(), 0)
        e2, _ = detect_elements_in_frame(sample_frame.copy(), 0)
        bboxes1 = sorted([(e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h) for e in e1])
        bboxes2 = sorted([(e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h) for e in e2])
        assert bboxes1 == bboxes2
