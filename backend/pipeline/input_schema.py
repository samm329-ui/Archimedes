"""
input_schema.py — Input contract validation schemas.

PRD §11: Every uploaded video must be normalized and validated.
This module owns the input-side contract.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_FILE_SIZE_MB = 500
MIN_FPS = 1.0
MAX_FPS = 240.0
MIN_DIMENSION = 32
MAX_DIMENSION = 7680


class VideoInputContract(BaseModel):
    """Validated input descriptor for an uploaded video."""
    filename: str
    file_size_bytes: int
    extension: str
    estimated_fps: Optional[float] = None
    estimated_duration_ms: Optional[float] = None

    @field_validator("extension")
    @classmethod
    def extension_supported(cls, v: str) -> str:
        if v.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{v}'. Supported: {SUPPORTED_EXTENSIONS}"
            )
        return v.lower()

    @field_validator("file_size_bytes")
    @classmethod
    def file_size_valid(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("File is empty")
        if v > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
        return v


class AnalysisRequest(BaseModel):
    """API-level analysis request."""
    video_path: str
    target_fps: float = Field(default=12.0, ge=1.0, le=30.0)
    max_elements: int = Field(default=100, ge=1, le=500)
    use_grabcut: bool = True
    scene_detection_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_reid: bool = True
    enable_grouping: bool = True
    enable_layering: bool = True
