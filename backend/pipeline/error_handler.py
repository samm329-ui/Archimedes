"""
error_handler.py — Centralized error classification, routing, and recovery.

PRD §24 anti-leak policy:
  - Every stage logs errors
  - Classify every failure mode
  - Preserve partially useful outputs
  - Never overwrite uncertainty with certainty
  - Never discard diagnostics needed for debugging

This module ensures every error is:
  1. Classified by FailureMode
  2. Tagged with stage and recoverability
  3. Routed to the right recovery strategy
  4. Logged with full context
  5. Never silently swallowed
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from backend.schemas import FailureMode, StructuredError

T = TypeVar("T")


# ── Recovery strategies ───────────────────────────────────────────────────────

class RecoveryStrategy:
    """Base class for error recovery strategies."""
    def can_recover(self, error: StructuredError) -> bool:
        return error.recoverable

    def recover(self, error: StructuredError, context: dict) -> Any:
        raise NotImplementedError


class ReturnEmptyList(RecoveryStrategy):
    """Return empty list as fallback."""
    def recover(self, error: StructuredError, context: dict) -> list:
        return []


class ReturnNone(RecoveryStrategy):
    """Return None as fallback."""
    def recover(self, error: StructuredError, context: dict) -> None:
        return None


class ReturnDefault(RecoveryStrategy):
    """Return a provided default value."""
    def __init__(self, default: Any):
        self._default = default

    def recover(self, error: StructuredError, context: dict) -> Any:
        return self._default


# ── Error classifier ──────────────────────────────────────────────────────────

# Maps exception types to FailureMode
EXCEPTION_TO_FAILURE_MODE: dict[type, FailureMode] = {
    FileNotFoundError:    FailureMode.INGEST_FAILED,
    PermissionError:      FailureMode.INGEST_FAILED,
    ValueError:           FailureMode.DETECTION_FAILED,
    ZeroDivisionError:    FailureMode.MOTION_AMBIGUOUS,
    OverflowError:        FailureMode.CURVE_FIT_FAILED,
    RuntimeError:         FailureMode.SCHEMA_BUILD_FAILED,
}

# Maps stage names to default failure modes
STAGE_DEFAULT_FAILURE: dict[str, FailureMode] = {
    "ingest":             FailureMode.INGEST_FAILED,
    "metadata":           FailureMode.INGEST_FAILED,
    "scene_detection":    FailureMode.SCENE_BOUNDARY_UNCERTAIN,
    "frame_sampling":     FailureMode.SCENE_BOUNDARY_UNCERTAIN,
    "detection":          FailureMode.DETECTION_FAILED,
    "segmentation":       FailureMode.MASK_INCOMPLETE,
    "tracking":           FailureMode.TRACK_LOST,
    "reid":               FailureMode.TRACK_LOST,
    "feature_extraction": FailureMode.DETECTION_FAILED,
    "grouping":           FailureMode.SCHEMA_BUILD_FAILED,
    "layering":           FailureMode.LAYER_ORDER_CONFLICT,
    "motion_analysis":    FailureMode.MOTION_AMBIGUOUS,
    "curve_fitting":      FailureMode.CURVE_FIT_FAILED,
    "hypothesis_engine":  FailureMode.MOTION_AMBIGUOUS,
    "role_assignment":    FailureMode.SCHEMA_BUILD_FAILED,
    "schema_builder":     FailureMode.SCHEMA_BUILD_FAILED,
    "render_loop":        FailureMode.RENDER_MISMATCH,
    "validator":          FailureMode.SCHEMA_BUILD_FAILED,
}


def classify_exception(exc: Exception, stage: str) -> FailureMode:
    """Map an exception type and stage to the appropriate FailureMode."""
    # Check exception type first
    for exc_type, mode in EXCEPTION_TO_FAILURE_MODE.items():
        if isinstance(exc, exc_type):
            return mode
    # Fall back to stage default
    return STAGE_DEFAULT_FAILURE.get(stage, FailureMode.SCHEMA_BUILD_FAILED)


def wrap_exception(
    exc: Exception,
    stage: str,
    context: dict[str, Any] | None = None,
    recoverable: bool = True,
) -> StructuredError:
    """
    Convert any Python exception into a StructuredError.
    Always includes the full traceback in details.
    """
    failure_mode = classify_exception(exc, stage)
    tb = traceback.format_exc()
    details = context or {}
    details["traceback"] = tb
    details["exception_type"] = type(exc).__name__

    return StructuredError(
        failure_mode=failure_mode,
        message=f"{type(exc).__name__}: {str(exc)}",
        stage=stage,
        recoverable=recoverable,
        details=details,
    )


@dataclass
class ErrorAccumulator:
    """
    Collects errors across pipeline stages.
    Provides filtering, grouping, and summary operations.
    Never drops errors — append-only.
    """
    _errors: list[StructuredError] = field(default_factory=list)

    def add(self, error: StructuredError) -> None:
        self._errors.append(error)

    def add_all(self, errors: list[StructuredError]) -> None:
        self._errors.extend(errors)

    def wrap_and_add(
        self,
        exc: Exception,
        stage: str,
        context: dict | None = None,
        recoverable: bool = True,
    ) -> StructuredError:
        err = wrap_exception(exc, stage, context, recoverable)
        self.add(err)
        return err

    def all(self) -> list[StructuredError]:
        return list(self._errors)

    def by_stage(self, stage: str) -> list[StructuredError]:
        return [e for e in self._errors if e.stage == stage]

    def by_mode(self, mode: FailureMode) -> list[StructuredError]:
        return [e for e in self._errors if e.failure_mode == mode]

    def recoverable(self) -> list[StructuredError]:
        return [e for e in self._errors if e.recoverable]

    def fatal(self) -> list[StructuredError]:
        return [e for e in self._errors if not e.recoverable]

    def has_fatal(self) -> bool:
        return any(not e.recoverable for e in self._errors)

    def count(self) -> int:
        return len(self._errors)

    def summary(self) -> dict[str, Any]:
        by_mode: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        for e in self._errors:
            by_mode[e.failure_mode.value] = by_mode.get(e.failure_mode.value, 0) + 1
            by_stage[e.stage] = by_stage.get(e.stage, 0) + 1
        return {
            "total": len(self._errors),
            "fatal": len(self.fatal()),
            "recoverable": len(self.recoverable()),
            "by_mode": by_mode,
            "by_stage": by_stage,
        }


def safe_execute(
    fn: Callable[[], T],
    stage: str,
    accumulator: ErrorAccumulator,
    fallback: T,
    context: dict | None = None,
    recoverable: bool = True,
) -> T:
    """
    Execute fn() safely. On any exception:
      - Wrap into StructuredError
      - Add to accumulator
      - Return fallback value

    This is the standard pattern for all pipeline stage calls.
    Never silently swallows — always produces a StructuredError.
    """
    try:
        return fn()
    except Exception as exc:
        accumulator.wrap_and_add(exc, stage, context, recoverable)
        return fallback
