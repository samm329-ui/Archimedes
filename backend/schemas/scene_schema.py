"""
scene_schema.py — Scene-level schemas.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from backend.schemas import SceneSegment


class SceneAnalysis(BaseModel):
    segment: SceneSegment
    frame_count_sampled: int = 0
    elements_detected: int = 0
    tracks_active: int = 0
    motion_spike_frames: list[int] = Field(default_factory=list)
    is_static: bool = False
    dominant_colors: list[list[int]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
