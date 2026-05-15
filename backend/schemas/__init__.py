"""
Canonical data schemas for the reverse motion compiler.
All field types are explicit. No implicit defaults.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    SHAPE = "shape"
    IMAGE = "image"
    ICON = "icon"
    OVERLAY = "overlay"
    UNKNOWN = "unknown"


class MotionPrimitive(str, Enum):
    TRANSLATE_X = "translateX"
    TRANSLATE_Y = "translateY"
    SCALE = "scale"
    ROTATE = "rotate"
    OPACITY = "opacity"
    MASK_REVEAL = "maskReveal"
    PATH_FOLLOW = "pathFollow"
    STATIC = "static"
    UNKNOWN = "unknown"


class EasingType(str, Enum):
    LINEAR = "linear"
    EASE_IN = "easeIn"
    EASE_OUT = "easeOut"
    EASE_IN_OUT = "easeInOut"
    EASE_OUT_CUBIC = "easeOutCubic"
    EASE_IN_CUBIC = "easeInCubic"
    BOUNCE = "bounce"
    SPRING = "spring"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REFINEMENT = "needs_refinement"
    PENDING = "pending"


class FailureMode(str, Enum):
    TEXT_DETECTION_FAILED = "text_detection_failed"
    MASK_INCOMPLETE = "mask_incomplete"
    TRACK_LOST = "track_lost"
    SCENE_BOUNDARY_UNCERTAIN = "scene_boundary_uncertain"
    MOTION_AMBIGUOUS = "motion_ambiguous"
    LAYER_ORDER_CONFLICT = "layer_order_conflict"
    RENDER_MISMATCH = "render_mismatch"
    UNSUPPORTED_CLASS = "unsupported_class"
    INGEST_FAILED = "ingest_failed"
    DETECTION_FAILED = "detection_failed"
    CURVE_FIT_FAILED = "curve_fit_failed"
    SCHEMA_BUILD_FAILED = "schema_build_failed"


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: "BoundingBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


class TypeCandidate(BaseModel):
    type: ElementType
    confidence: float = Field(ge=0.0, le=1.0)


class MotionHypothesis(BaseModel):
    primitive: MotionPrimitive
    easing: EasingType
    from_value: float
    to_value: float
    duration_frames: int
    start_frame: int
    raw_curve: list[float]
    confidence: float = Field(ge=0.0, le=1.0)


class RoleScore(BaseModel):
    role: str
    score: float = Field(ge=0.0, le=1.0)


class LayoutInfo(BaseModel):
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float
    layer: int = 0
    layer_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StyleInfo(BaseModel):
    dominant_color: Optional[list[int]] = None
    has_glow: bool = False
    has_shadow: bool = False
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    blur_radius: float = 0.0


class TimingInfo(BaseModel):
    enter_frame: int
    exit_frame: int
    duration_frames: int
    fps: float


class ValidationResult(BaseModel):
    status: ValidationStatus = ValidationStatus.PENDING
    ssim_score: Optional[float] = None
    similarity_score: Optional[float] = None
    temporal_score: Optional[float] = None
    text_agreement_score: Optional[float] = None
    failure_reasons: list[str] = Field(default_factory=list)
    refinement_attempts: int = 0


class ProvenanceRecord(BaseModel):
    source_module: str
    source_frame_range: Optional[tuple[int, int]] = None
    method: str
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class StructuredError(BaseModel):
    failure_mode: FailureMode
    message: str
    stage: str
    recoverable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DetectedElement(BaseModel):
    id: str
    frame_index: int
    bbox: BoundingBox
    type_candidates: list[TypeCandidate]
    mask_available: bool = False
    features: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceRecord


class TrackedElement(BaseModel):
    track_id: str
    element_ids: list[str]
    type_candidates: list[TypeCandidate]
    bbox_sequence: list[tuple[int, BoundingBox]]
    continuity_score: float = Field(ge=0.0, le=1.0)
    occlusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reid_score: float = Field(default=1.0, ge=0.0, le=1.0)
    split_events: list[int] = Field(default_factory=list)
    merge_events: list[int] = Field(default_factory=list)


class MotionElement(BaseModel):
    track_id: str
    motion_hypotheses: list[MotionHypothesis]
    raw_trajectory: list[tuple[int, float, float]]
    motion_confidence: float = Field(ge=0.0, le=1.0)


class SceneSegment(BaseModel):
    scene_id: str
    start_frame: int
    end_frame: int
    duration_frames: int
    boundary_confidence: float = Field(ge=0.0, le=1.0)
    same_scene_hypothesis: float = Field(ge=0.0, le=1.0)
    new_scene_hypothesis: float = Field(ge=0.0, le=1.0)


class TemplateElement(BaseModel):
    id: str
    type: ElementType
    subtype: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    role_scores: list[RoleScore] = Field(default_factory=list)
    content: Optional[str] = None
    layout: LayoutInfo
    style: StyleInfo
    motion: list[MotionHypothesis]
    timing: TimingInfo
    group_id: Optional[str] = None
    layer: int = 0
    alternatives: list[TypeCandidate] = Field(default_factory=list)
    provenance: ProvenanceRecord
    failure_modes: list[FailureMode] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: float
    codec: str
    has_audio: bool
    color_space: str = "unknown"
    is_corrupt: bool = False


class TemplateJSON(BaseModel):
    schema_version: str = "1.0.0"
    meta: VideoMetadata
    quality: dict[str, Any] = Field(default_factory=dict)
    canvas: dict[str, Any] = Field(default_factory=dict)
    scenes: list[SceneSegment] = Field(default_factory=list)
    elements: list[TemplateElement] = Field(default_factory=list)
    camera: dict[str, Any] = Field(default_factory=dict)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    errors: list[StructuredError] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
