"""
test_feature_extraction.py — Tests for backend/pipeline/feature_extraction.py
"""
from __future__ import annotations
import numpy as np
import pytest
from backend.pipeline.feature_extraction import (
    extract_features, feature_distance, histogram_similarity, ElementFeatures
)


def _solid_frame(color=(100, 150, 200), w=320, h=240):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = color
    return f


def _complex_frame(w=320, h=240):
    import cv2
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = (40, 50, 60)
    cv2.rectangle(f, (50, 50), (150, 100), (255, 255, 255), -1)
    cv2.rectangle(f, (10, 180), (200, 220), (200, 200, 240), -1)
    return f


class TestExtractFeatures:
    def test_returns_element_features(self):
        frame = _solid_frame()
        result = extract_features(frame, 50, 50, 100, 40, 320, 240)
        assert isinstance(result, ElementFeatures)

    def test_aspect_ratio_correct(self):
        frame = _solid_frame()
        result = extract_features(frame, 0, 0, 100, 50, 320, 240)
        assert abs(result.aspect_ratio - 2.0) < 0.01

    def test_area_pixels_correct(self):
        frame = _solid_frame()
        result = extract_features(frame, 0, 0, 100, 50, 320, 240)
        assert abs(result.area_pixels - 5000) < 1

    def test_area_norm_in_range(self):
        frame = _solid_frame()
        result = extract_features(frame, 0, 0, 100, 50, 320, 240)
        assert 0.0 <= result.area_norm <= 1.0

    def test_color_histogram_length(self):
        frame = _solid_frame()
        result = extract_features(frame, 0, 0, 100, 50, 320, 240)
        assert len(result.color_histogram) == 128

    def test_edge_density_in_range(self):
        frame = _complex_frame()
        result = extract_features(frame, 50, 50, 100, 50, 320, 240)
        assert 0.0 <= result.edge_density <= 1.0

    def test_compactness_in_range(self):
        frame = _complex_frame()
        result = extract_features(frame, 50, 50, 100, 50, 320, 240)
        assert 0.0 <= result.compactness <= 1.0

    def test_spatial_norms_in_range(self):
        frame = _solid_frame()
        result = extract_features(frame, 80, 60, 100, 50, 320, 240)
        assert 0.0 <= result.x_norm <= 1.0
        assert 0.0 <= result.y_norm <= 1.0
        assert 0.0 <= result.w_norm <= 1.0
        assert 0.0 <= result.h_norm <= 1.0

    def test_to_dict_returns_dict(self):
        frame = _solid_frame()
        result = extract_features(frame, 0, 0, 100, 50, 320, 240)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "aspect_ratio" in d
        assert "edge_density" in d

    def test_bbox_outside_frame_safe(self):
        frame = _solid_frame()
        # Bbox extending beyond frame — should not crash
        result = extract_features(frame, 300, 220, 100, 100, 320, 240)
        assert isinstance(result, ElementFeatures)

    def test_zero_size_bbox_safe(self):
        frame = _solid_frame()
        result = extract_features(frame, 100, 100, 0, 0, 320, 240)
        assert isinstance(result, ElementFeatures)

    def test_deterministic(self):
        frame = _complex_frame()
        r1 = extract_features(frame, 50, 50, 100, 40, 320, 240)
        r2 = extract_features(frame, 50, 50, 100, 40, 320, 240)
        assert r1.aspect_ratio == r2.aspect_ratio
        assert r1.edge_density == r2.edge_density


class TestHistogramSimilarity:
    def test_identical_histograms_score_1(self):
        hist = [0.01] * 128
        score = histogram_similarity(hist, hist)
        assert score > 0.95

    def test_different_histograms_lower_score(self):
        h1 = [0.0] * 64 + [0.02] * 64
        h2 = [0.02] * 64 + [0.0] * 64
        score = histogram_similarity(h1, h2)
        assert score < 0.8

    def test_empty_histograms_returns_zero(self):
        assert histogram_similarity([], []) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert histogram_similarity([0.1]*10, [0.1]*20) == 0.0

    def test_score_in_range(self):
        import random
        h1 = [random.random() for _ in range(128)]
        h2 = [random.random() for _ in range(128)]
        score = histogram_similarity(h1, h2)
        assert 0.0 <= score <= 1.0


class TestFeatureDistance:
    def test_same_features_zero_distance(self):
        f = {"aspect_ratio": 1.5, "color_variance": 1000.0,
             "mean_color": [100, 100, 100], "color_histogram": [0.01]*128}
        dist = feature_distance(f, f)
        assert dist < 0.1

    def test_different_colors_higher_distance(self):
        f1 = {"aspect_ratio": 1.5, "color_variance": 100.0,
              "mean_color": [255, 0, 0], "color_histogram": [0.01]*128}
        f2 = {"aspect_ratio": 1.5, "color_variance": 100.0,
              "mean_color": [0, 0, 255], "color_histogram": [0.01]*128}
        dist = feature_distance(f1, f2)
        assert dist > 0.2

    def test_distance_in_range(self):
        f1 = {"aspect_ratio": 2.0, "color_variance": 500.0, "mean_color": [80, 80, 80]}
        f2 = {"aspect_ratio": 1.0, "color_variance": 200.0, "mean_color": [200, 200, 200]}
        dist = feature_distance(f1, f2)
        assert 0.0 <= dist <= 1.0

    def test_empty_dicts_safe(self):
        dist = feature_distance({}, {})
        assert 0.0 <= dist <= 1.0
