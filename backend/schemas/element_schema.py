"""
element_schema.py — Element-level schemas with full field typing.
PRD §21: every field must have a known type, every inferred field carries confidence.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from backend.schemas import (
    ElementType,
    LayoutInfo,
    StyleInfo,
    MotionHypothesis,
    TimingInfo,
    RoleScore,
    TypeCandidate,
    ProvenanceRecord,
    FailureMode,
)


class GroupInfo(BaseModel):
    group_id: Optional[str] = None
    group_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hypothesis_group_id: Optional[str] = None


class LayerInfo(BaseModel):
    layer: int = 0
    layer_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis_only: bool = False
    reasoning: list[str] = Field(default_factory=list)


class FullElementSchema(BaseModel):
    id: str
    type: ElementType
    type_confidence: float = Field(ge=0.0, le=1.0)
    subtype: Optional[str] = None
    content: Optional[str] = None
    layout: LayoutInfo
    style: StyleInfo
    motion: list[MotionHypothesis] = Field(default_factory=list)
    timing: TimingInfo
    role_scores: list[RoleScore] = Field(default_factory=list)
    group: GroupInfo = Field(default_factory=GroupInfo)
    layer_info: LayerInfo = Field(default_factory=LayerInfo)
    alternatives: list[TypeCandidate] = Field(default_factory=list)
    provenance: ProvenanceRecord
    failure_modes: list[FailureMode] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    mask_available: bool = False
    mask_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_partial: bool = False
    has_transparency: bool = False
