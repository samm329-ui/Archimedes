"""
validator.py — Structural and quality validation of assembled TemplateJSON.

Does NOT perform render-based comparison (that's render_loop.py).
Checks:
  - Schema completeness
  - Confidence thresholds
  - Required field presence
  - Internal consistency

Returns structured ValidationResult with status and reasons.
"""
from __future__ import annotations

from backend.schemas import (
    FailureMode,
    StructuredError,
    TemplateJSON,
    ValidationResult,
    ValidationStatus,
)


# ── Thresholds (explicit) ─────────────────────────────────────────────────────

MIN_ELEMENT_CONFIDENCE = 0.15
MIN_MOTION_CONFIDENCE = 0.10
MAX_ALLOWED_UNKNOWN_ELEMENTS_FRACTION = 0.8
MIN_SCENES_REQUIRED = 1


class ValidationError(ValueError):
    """Raised when validation cannot even begin (bad input)."""
    pass


def validate_template(
    template: TemplateJSON,
) -> tuple[ValidationResult, list[StructuredError]]:
    """
    Validate the assembled template for structural integrity.

    Returns:
        (ValidationResult, list[StructuredError])
    """
    errors: list[StructuredError] = []
    failure_reasons: list[str] = []

    # ── Check 1: Metadata integrity ────────────────────────────────────────
    if template.meta.width <= 0 or template.meta.height <= 0:
        failure_reasons.append(f"Invalid canvas dimensions: {template.meta.width}x{template.meta.height}")

    if template.meta.fps <= 0:
        failure_reasons.append(f"Invalid FPS: {template.meta.fps}")

    if template.meta.frame_count <= 0:
        failure_reasons.append(f"Invalid frame count: {template.meta.frame_count}")

    # ── Check 2: Scene presence ────────────────────────────────────────────
    if len(template.scenes) < MIN_SCENES_REQUIRED:
        failure_reasons.append(f"No scenes detected (minimum {MIN_SCENES_REQUIRED} required)")

    # ── Check 3: Element quality ───────────────────────────────────────────
    total_elements = len(template.elements)
    if total_elements == 0:
        failure_reasons.append("Template contains zero elements")

    low_conf_count = 0
    unknown_count = 0

    for elem in template.elements:
        # Each element must have confidence
        if elem.confidence < MIN_ELEMENT_CONFIDENCE:
            low_conf_count += 1
            errors.append(StructuredError(
                failure_mode=FailureMode.DETECTION_FAILED,
                message=f"Element {elem.id} has very low confidence: {elem.confidence:.3f}",
                stage="validator",
                recoverable=True,
                details={"element_id": elem.id, "confidence": elem.confidence},
            ))

        from backend.schemas import ElementType
        if elem.type == ElementType.UNKNOWN:
            unknown_count += 1

        # Motion must exist (even if static)
        if not elem.motion:
            errors.append(StructuredError(
                failure_mode=FailureMode.MOTION_AMBIGUOUS,
                message=f"Element {elem.id} has no motion hypotheses",
                stage="validator",
                recoverable=True,
                details={"element_id": elem.id},
            ))

        # Layout sanity
        if (
            elem.layout.x_norm < 0 or elem.layout.x_norm > 1.1 or
            elem.layout.y_norm < 0 or elem.layout.y_norm > 1.1
        ):
            failure_reasons.append(
                f"Element {elem.id} layout out of bounds: "
                f"({elem.layout.x_norm:.2f}, {elem.layout.y_norm:.2f})"
            )

        # Provenance must exist
        if not elem.provenance:
            errors.append(StructuredError(
                failure_mode=FailureMode.SCHEMA_BUILD_FAILED,
                message=f"Element {elem.id} missing provenance",
                stage="validator",
                recoverable=False,
                details={"element_id": elem.id},
            ))

    if total_elements > 0:
        unknown_fraction = unknown_count / total_elements
        if unknown_fraction > MAX_ALLOWED_UNKNOWN_ELEMENTS_FRACTION:
            failure_reasons.append(
                f"Too many unknown-type elements: {unknown_count}/{total_elements} "
                f"({unknown_fraction:.0%})"
            )

    # ── Check 4: Schema version ────────────────────────────────────────────
    if not template.schema_version:
        failure_reasons.append("Missing schema_version")

    # ── Check 5: Canvas consistency ────────────────────────────────────────
    canvas_w = template.canvas.get("width", 0)
    canvas_h = template.canvas.get("height", 0)
    if canvas_w != template.meta.width or canvas_h != template.meta.height:
        failure_reasons.append(
            f"Canvas mismatch: canvas={canvas_w}x{canvas_h}, "
            f"meta={template.meta.width}x{template.meta.height}"
        )

    # ── Determine status ───────────────────────────────────────────────────
    if failure_reasons:
        status = ValidationStatus.REJECTED
    elif errors:
        status = ValidationStatus.NEEDS_REFINEMENT
    else:
        status = ValidationStatus.APPROVED

    # Compute basic quality scores
    if total_elements > 0:
        mean_conf = sum(e.confidence for e in template.elements) / total_elements
        text_agreement = float(sum(
            1 for e in template.elements if e.type.value in ("text", "shape")
        ) / total_elements)
    else:
        mean_conf = 0.0
        text_agreement = 0.0

    result = ValidationResult(
        status=status,
        ssim_score=None,           # Set by render_loop
        similarity_score=mean_conf,
        temporal_score=None,        # Set by render_loop
        text_agreement_score=text_agreement,
        failure_reasons=failure_reasons,
        refinement_attempts=template.validation.refinement_attempts,
    )

    return result, errors


def validate_schema_completeness(template_dict: dict) -> list[str]:
    """
    Validate that a template dict has all required top-level keys.
    Returns list of missing fields.
    """
    required_keys = [
        "schema_version", "meta", "canvas", "scenes",
        "elements", "validation", "errors", "provenance",
    ]
    return [k for k in required_keys if k not in template_dict]
