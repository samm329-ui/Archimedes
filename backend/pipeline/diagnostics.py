"""
diagnostics.py — Pipeline observability, stage profiling, and diagnostics.

PRD §26: The frontend is an inspection console. Diagnostics must expose
every stage's outputs, confidences, and timing so failures can be debugged.

PRD §24: Every stage must log errors, store fallback decisions, and continue.

This module:
  - Records per-stage timing
  - Aggregates all errors by stage and failure_mode
  - Computes confidence distributions per stage
  - Produces a full diagnostic report attached to the TemplateJSON
  - Never hides uncertainty or suppresses diagnostics
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.schemas import FailureMode, StructuredError


# ── Stage names (canonical) ───────────────────────────────────────────────────

STAGE_INGEST = "ingest"
STAGE_METADATA = "metadata"
STAGE_SCENE_DETECTION = "scene_detection"
STAGE_FRAME_SAMPLING = "frame_sampling"
STAGE_DETECTION = "detection"
STAGE_SEGMENTATION = "segmentation"
STAGE_TRACKING = "tracking"
STAGE_REID = "reid"
STAGE_FEATURE_EXTRACTION = "feature_extraction"
STAGE_GROUPING = "grouping"
STAGE_LAYERING = "layering"
STAGE_MOTION_ANALYSIS = "motion_analysis"
STAGE_CURVE_FITTING = "curve_fitting"
STAGE_HYPOTHESIS_ENGINE = "hypothesis_engine"
STAGE_ROLE_ASSIGNMENT = "role_assignment"
STAGE_SCHEMA_BUILDER = "schema_builder"
STAGE_RENDER_LOOP = "render_loop"
STAGE_VALIDATOR = "validator"

ALL_STAGES = [
    STAGE_INGEST, STAGE_METADATA, STAGE_SCENE_DETECTION,
    STAGE_FRAME_SAMPLING, STAGE_DETECTION, STAGE_SEGMENTATION,
    STAGE_TRACKING, STAGE_REID, STAGE_FEATURE_EXTRACTION,
    STAGE_GROUPING, STAGE_LAYERING, STAGE_MOTION_ANALYSIS,
    STAGE_CURVE_FITTING, STAGE_HYPOTHESIS_ENGINE, STAGE_ROLE_ASSIGNMENT,
    STAGE_SCHEMA_BUILDER, STAGE_RENDER_LOOP, STAGE_VALIDATOR,
]


@dataclass
class StageRecord:
    """Record for a single pipeline stage execution."""
    stage: str
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0
    output_count: int = 0       # elements/scenes/tracks produced
    confidence_values: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False

    def finish(self, output_count: int = 0, notes: list[str] | None = None) -> None:
        self.finished_at = time.monotonic()
        self.duration_ms = (self.finished_at - self.started_at) * 1000.0
        self.output_count = output_count
        if notes:
            self.notes.extend(notes)

    def add_confidence(self, value: float) -> None:
        self.confidence_values.append(max(0.0, min(1.0, value)))

    def mean_confidence(self) -> Optional[float]:
        if not self.confidence_values:
            return None
        return round(sum(self.confidence_values) / len(self.confidence_values), 4)

    def min_confidence(self) -> Optional[float]:
        if not self.confidence_values:
            return None
        return round(min(self.confidence_values), 4)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "output_count": self.output_count,
            "mean_confidence": self.mean_confidence(),
            "min_confidence": self.min_confidence(),
            "skipped": self.skipped,
            "notes": self.notes,
        }


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report for one pipeline run.
    Attached to TemplateJSON.provenance for full traceability.
    """
    video_path: str
    total_errors: int = 0
    total_warnings: int = 0
    stages: dict[str, StageRecord] = field(default_factory=dict)
    errors_by_stage: dict[str, list[str]] = field(default_factory=dict)
    errors_by_mode: dict[str, int] = field(default_factory=dict)
    pipeline_start: float = field(default_factory=time.monotonic)
    pipeline_end: Optional[float] = None
    total_duration_ms: Optional[float] = None
    passed_validation: bool = False
    refinement_count: int = 0
    element_confidence_distribution: dict[str, float] = field(default_factory=dict)

    def start_stage(self, stage: str) -> StageRecord:
        record = StageRecord(stage=stage)
        self.stages[stage] = record
        return record

    def finish_stage(
        self,
        stage: str,
        output_count: int = 0,
        notes: list[str] | None = None,
    ) -> None:
        if stage in self.stages:
            self.stages[stage].finish(output_count, notes)

    def record_errors(self, errors: list[StructuredError]) -> None:
        for err in errors:
            self.total_errors += 1
            stage = err.stage
            if stage not in self.errors_by_stage:
                self.errors_by_stage[stage] = []
            self.errors_by_stage[stage].append(err.message)
            mode = err.failure_mode.value
            self.errors_by_mode[mode] = self.errors_by_mode.get(mode, 0) + 1
            if stage in self.stages:
                self.stages[stage].error_count += 1

    def record_confidences(self, stage: str, values: list[float]) -> None:
        if stage in self.stages:
            for v in values:
                self.stages[stage].add_confidence(v)

    def finalize(self) -> None:
        self.pipeline_end = time.monotonic()
        self.total_duration_ms = (self.pipeline_end - self.pipeline_start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "total_duration_ms": round(self.total_duration_ms, 2) if self.total_duration_ms else None,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "passed_validation": self.passed_validation,
            "refinement_count": self.refinement_count,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "errors_by_stage": self.errors_by_stage,
            "errors_by_mode": self.errors_by_mode,
            "element_confidence_distribution": self.element_confidence_distribution,
        }

    def bottleneck_stage(self) -> Optional[str]:
        """Return the stage with the most errors."""
        if not self.errors_by_stage:
            return None
        return max(self.errors_by_stage, key=lambda s: len(self.errors_by_stage[s]))

    def slowest_stage(self) -> Optional[str]:
        """Return the stage that took the most time."""
        timed = {
            k: v.duration_ms
            for k, v in self.stages.items()
            if v.duration_ms is not None
        }
        if not timed:
            return None
        return max(timed, key=lambda k: timed[k])

    def confidence_summary(self) -> dict[str, Optional[float]]:
        return {
            stage: record.mean_confidence()
            for stage, record in self.stages.items()
            if record.confidence_values
        }
