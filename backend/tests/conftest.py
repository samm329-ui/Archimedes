"""
conftest.py — Shared pytest fixtures for all test modules.
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ── Synthetic video factory ───────────────────────────────────────────────────

def make_test_video(
    path: str,
    width: int = 320,
    height: int = 240,
    fps: float = 24.0,
    num_frames: int = 48,
    color_sequence: list[tuple[int, int, int]] | None = None,
) -> str:
    """
    Write a synthetic MP4 video using OpenCV VideoWriter.
    Returns the path.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open: {path}"

    colors = color_sequence or [(50 * (i % 5), 100, 200 - i * 3) for i in range(num_frames)]

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Fill background
        bg_color = colors[i % len(colors)]
        frame[:] = bg_color

        # Draw a moving rectangle (simulates a tracked element)
        x = int((i / num_frames) * (width - 60)) + 10
        y = height // 4
        cv2.rectangle(frame, (x, y), (x + 50, y + 40), (255, 255, 255), -1)

        # Draw text-like region
        cv2.rectangle(frame, (10, height - 60), (200, height - 20), (220, 220, 255), -1)

        writer.write(frame)

    writer.release()
    return path


def make_two_scene_video(path: str) -> str:
    """
    A video with a clear scene cut halfway through.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (320, 240))
    assert writer.isOpened()

    # Scene 1: red tones
    for _ in range(24):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 2] = 180  # red
        writer.write(frame)

    # Scene 2: blue tones (abrupt cut)
    for _ in range(24):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # blue
        writer.write(frame)

    writer.release()
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("videos")


@pytest.fixture(scope="session")
def simple_video(tmp_dir: Path) -> str:
    path = str(tmp_dir / "simple.mp4")
    return make_test_video(path)


@pytest.fixture(scope="session")
def two_scene_video(tmp_dir: Path) -> str:
    path = str(tmp_dir / "two_scene.mp4")
    return make_two_scene_video(path)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 320x240 synthetic BGR frame with a white rectangle and text region."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (50, 60, 70)
    cv2.rectangle(frame, (40, 30), (120, 80), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 180), (200, 220), (200, 200, 240), -1)
    return frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)
