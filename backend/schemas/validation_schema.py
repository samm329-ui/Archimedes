"""
validation_schema.py — Validation result schemas.
PRD §23: six validation layers, each producing a record.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from backend.schemas import ValidationStatus


class LayerValidationRecord(BaseModel):
    layer_name: str
    passed: bool
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    failure_reasons: list[str] = Field(default_factory=list)
    notes: str = ""


class FullValidationReport(BaseModel):
    detection_confidence: LayerValidationRecord
    tracking_stability: LayerValidationRecord
    group_consistency: LayerValidationRecord
    motion_confidence: LayerValidationRecord
    render_validation: LayerValidationRecord
    final_gate: LayerValidationRecord
    overall_status: ValidationStatus = ValidationStatus.PENDING
    ssim_score: Optional[float] = None
    similarity_score: Optional[float] = None
    refinement_attempts: int = 0

    def all_passed(self) -> bool:
        return all(
            [
                self.detection_confidence.passed,
                self.tracking_stability.passed,
                self.group_consistency.passed,
                self.motion_confidence.passed,
                self.render_validation.passed,
                self.final_gate.passed,
            ]
        )

    def first_failure(self) -> Optional[str]:
        for layer in [
            self.detection_confidence,
            self.tracking_stability,
            self.group_consistency,
            self.motion_confidence,
            self.render_validation,
            self.final_gate,
        ]:
            if not layer.passed:
                return layer.layer_name
        return None
