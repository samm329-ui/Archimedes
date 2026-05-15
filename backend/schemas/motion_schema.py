"""
motion_schema.py — Motion-level schemas with full hypothesis typing.
PRD §19: store easing label + raw curve + confidence. Always.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from backend.schemas import EasingType, MotionPrimitive


class CurveFitRecord(BaseModel):
    easing: EasingType
    r2_score: float = Field(ge=0.0, le=1.0)
    raw_curve: list[float] = Field(default_factory=list)
    from_value: float
    to_value: float
    confidence: float = Field(ge=0.0, le=1.0)


class MotionAxisAnalysis(BaseModel):
    primitive: MotionPrimitive
    raw_trajectory: list[tuple[int, float]]
    smoothed_trajectory: list[float] = Field(default_factory=list)
    fit_results: list[CurveFitRecord] = Field(default_factory=list)
    selected_easing: EasingType = EasingType.UNKNOWN
    motion_magnitude: float = 0.0
    is_static: bool = False
    low_confidence_raw_kept: bool = False
