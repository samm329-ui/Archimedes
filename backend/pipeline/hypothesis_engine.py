"""
hypothesis_engine.py — Central hypothesis management system.

PRD §19 + §30 rules:
  - Never finalize early
  - Never collapse hypotheses too soon
  - Every ambiguous field retains alternatives
  - Hypotheses are ranked by confidence, not forced to single answer
  - Provides merge, prune, promote, and demote operations

This module is the single source of truth for all multi-hypothesis
state in the pipeline. It prevents premature commitment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from backend.schemas import (
    EasingType,
    ElementType,
    MotionHypothesis,
    MotionPrimitive,
    TypeCandidate,
    FailureMode,
    StructuredError,
)

T = TypeVar("T")


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_RETAIN_CONFIDENCE = 0.08      # Below this → prune hypothesis
MAX_HYPOTHESES_PER_SLOT = 5       # Cap to prevent combinatorial explosion
PROMOTE_THRESHOLD = 0.75          # Above this → promote to "high confidence"
DEMOTE_THRESHOLD = 0.25           # Below this → demote to "hypothesis only"


class HypothesisState(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"    # confidence >= PROMOTE_THRESHOLD
    ACTIVE = "active"                      # normal state
    HYPOTHESIS_ONLY = "hypothesis_only"   # confidence < DEMOTE_THRESHOLD
    PRUNED = "pruned"                      # below MIN_RETAIN_CONFIDENCE


@dataclass
class Hypothesis(Generic[T]):
    """A single typed hypothesis with confidence and provenance."""
    value: T
    confidence: float
    state: HypothesisState = HypothesisState.ACTIVE
    source: str = ""
    notes: str = ""

    def __post_init__(self):
        self._update_state()

    def _update_state(self):
        if self.confidence >= PROMOTE_THRESHOLD:
            self.state = HypothesisState.HIGH_CONFIDENCE
        elif self.confidence >= DEMOTE_THRESHOLD:
            self.state = HypothesisState.ACTIVE
        elif self.confidence >= MIN_RETAIN_CONFIDENCE:
            self.state = HypothesisState.HYPOTHESIS_ONLY
        else:
            self.state = HypothesisState.PRUNED

    def update_confidence(self, new_confidence: float, reason: str = "") -> None:
        self.confidence = max(0.0, min(1.0, new_confidence))
        if reason:
            self.notes = f"{self.notes}; {reason}" if self.notes else reason
        self._update_state()


@dataclass
class HypothesisSet(Generic[T]):
    """
    A ranked set of hypotheses for one decision slot.
    Maintains ordering, caps count, and enforces PRD rules.
    """
    slot_name: str
    hypotheses: list[Hypothesis[T]] = field(default_factory=list)

    def add(self, value: T, confidence: float, source: str = "") -> None:
        h = Hypothesis(value=value, confidence=confidence, source=source)
        if h.state == HypothesisState.PRUNED:
            return
        self.hypotheses.append(h)
        self._sort_and_cap()

    def _sort_and_cap(self) -> None:
        self.hypotheses = sorted(
            self.hypotheses, key=lambda h: -h.confidence
        )[:MAX_HYPOTHESES_PER_SLOT]

    def best(self) -> Optional[Hypothesis[T]]:
        active = [h for h in self.hypotheses if h.state != HypothesisState.PRUNED]
        return active[0] if active else None

    def best_value(self, default: Optional[T] = None) -> Optional[T]:
        b = self.best()
        return b.value if b else default

    def best_confidence(self) -> float:
        b = self.best()
        return b.confidence if b else 0.0

    def all_active(self) -> list[Hypothesis[T]]:
        return [h for h in self.hypotheses if h.state != HypothesisState.PRUNED]

    def reinforce(self, value: T, bonus: float, source: str = "") -> None:
        """Increase confidence of matching hypothesis, or add new one."""
        for h in self.hypotheses:
            if h.value == value:
                h.update_confidence(
                    min(1.0, h.confidence + bonus),
                    reason=f"reinforced_by:{source}",
                )
                self._sort_and_cap()
                return
        self.add(value, bonus, source=source)

    def penalize(self, value: T, penalty: float, source: str = "") -> None:
        """Reduce confidence of matching hypothesis."""
        for h in self.hypotheses:
            if h.value == value:
                h.update_confidence(
                    max(0.0, h.confidence - penalty),
                    reason=f"penalized_by:{source}",
                )
                self._sort_and_cap()
                return

    def prune_below(self, threshold: float) -> int:
        """Remove hypotheses below threshold. Returns count removed."""
        before = len(self.hypotheses)
        self.hypotheses = [h for h in self.hypotheses if h.confidence >= threshold]
        return before - len(self.hypotheses)

    def is_decided(self) -> bool:
        """True if top hypothesis has high confidence and no close second."""
        active = self.all_active()
        if not active:
            return False
        if active[0].confidence < PROMOTE_THRESHOLD:
            return False
        if len(active) >= 2 and (active[0].confidence - active[1].confidence) < 0.2:
            return False
        return True

    def to_dict(self) -> list[dict]:
        return [
            {
                "value": str(h.value),
                "confidence": h.confidence,
                "state": h.state.value,
                "source": h.source,
            }
            for h in self.all_active()
        ]


class ElementHypothesisManager:
    """
    Manages all hypothesis sets for a single detected/tracked element.
    Keeps type, motion, role, and layer hypotheses in one place.
    """

    def __init__(self, element_id: str):
        self.element_id = element_id
        self.type_set: HypothesisSet[ElementType] = HypothesisSet("type")
        self.motion_primitives: HypothesisSet[MotionPrimitive] = HypothesisSet("motion_primitive")
        self.easing_set: HypothesisSet[EasingType] = HypothesisSet("easing")
        self.role_set: HypothesisSet[str] = HypothesisSet("role")
        self.layer_set: HypothesisSet[int] = HypothesisSet("layer")
        self._evidence_log: list[dict] = []

    def add_type_evidence(
        self, element_type: ElementType, confidence: float, source: str
    ) -> None:
        self.type_set.add(element_type, confidence, source)
        self._log("type", element_type.value, confidence, source)

    def add_motion_evidence(
        self, primitive: MotionPrimitive, confidence: float, source: str
    ) -> None:
        self.motion_primitives.add(primitive, confidence, source)
        self._log("motion_primitive", primitive.value, confidence, source)

    def add_role_evidence(self, role: str, confidence: float, source: str) -> None:
        self.role_set.add(role, confidence, source)
        self._log("role", role, confidence, source)

    def add_layer_evidence(self, layer: int, confidence: float, source: str) -> None:
        self.layer_set.add(layer, confidence, source)
        self._log("layer", str(layer), confidence, source)

    def _log(self, slot: str, value: str, confidence: float, source: str) -> None:
        self._evidence_log.append({
            "slot": slot,
            "value": value,
            "confidence": confidence,
            "source": source,
        })

    def get_best_type(self) -> tuple[ElementType, float]:
        v = self.type_set.best_value(ElementType.UNKNOWN)
        c = self.type_set.best_confidence()
        return v, c

    def get_best_motion(self) -> tuple[MotionPrimitive, float]:
        v = self.motion_primitives.best_value(MotionPrimitive.UNKNOWN)
        c = self.motion_primitives.best_confidence()
        return v, c

    def get_best_layer(self) -> tuple[int, float]:
        v = self.layer_set.best_value(0)
        c = self.layer_set.best_confidence()
        return v or 0, c

    def get_type_candidates(self) -> list[TypeCandidate]:
        return [
            TypeCandidate(type=h.value, confidence=h.confidence)
            for h in self.type_set.all_active()
        ]

    def is_type_decided(self) -> bool:
        return self.type_set.is_decided()

    def is_motion_decided(self) -> bool:
        return self.motion_primitives.is_decided()

    def summary(self) -> dict:
        return {
            "element_id": self.element_id,
            "type": self.type_set.to_dict(),
            "motion_primitive": self.motion_primitives.to_dict(),
            "role": self.role_set.to_dict(),
            "layer": self.layer_set.to_dict(),
            "evidence_count": len(self._evidence_log),
            "type_decided": self.is_type_decided(),
            "motion_decided": self.is_motion_decided(),
        }


def merge_type_hypotheses(
    sets: list[list[TypeCandidate]],
    weights: Optional[list[float]] = None,
) -> list[TypeCandidate]:
    """
    Merge multiple TypeCandidate lists using weighted averaging.
    Used when multiple detection sources vote on the same element.
    """
    if not sets:
        return []

    if weights is None:
        weights = [1.0] * len(sets)

    total_weight = sum(weights)
    if total_weight == 0:
        return []

    merged: dict[str, float] = {}
    for s, w in zip(sets, weights):
        for tc in s:
            key = tc.type.value
            merged[key] = merged.get(key, 0.0) + tc.confidence * (w / total_weight)

    from backend.schemas import ElementType
    results = [
        TypeCandidate(type=ElementType(k), confidence=round(v, 4))
        for k, v in sorted(merged.items(), key=lambda x: -x[1])
        if v >= MIN_RETAIN_CONFIDENCE
    ]
    return results[:MAX_HYPOTHESES_PER_SLOT]


def select_best_motion_hypotheses(
    all_hypotheses: list[MotionHypothesis],
    max_per_primitive: int = 2,
) -> list[MotionHypothesis]:
    """
    From a flat list of motion hypotheses, select the best
    per primitive, respecting the max_per_primitive cap.
    Returns sorted by confidence descending.
    Never discards ALL — keeps at least one even if low confidence.
    """
    if not all_hypotheses:
        return []

    by_primitive: dict[str, list[MotionHypothesis]] = {}
    for h in all_hypotheses:
        key = h.primitive.value
        by_primitive.setdefault(key, []).append(h)

    selected: list[MotionHypothesis] = []
    for prim_hyps in by_primitive.values():
        sorted_hyps = sorted(prim_hyps, key=lambda h: -h.confidence)
        selected.extend(sorted_hyps[:max_per_primitive])

    selected.sort(key=lambda h: -h.confidence)

    # Always keep at least one
    if not selected and all_hypotheses:
        selected = [max(all_hypotheses, key=lambda h: h.confidence)]

    return selected


def validate_hypothesis_coverage(
    manager: ElementHypothesisManager,
) -> list[StructuredError]:
    """
    Check that an element has sufficient hypothesis coverage.
    Returns errors for any slot with zero active hypotheses.
    """
    errors: list[StructuredError] = []

    if not manager.type_set.all_active():
        errors.append(StructuredError(
            failure_mode=FailureMode.DETECTION_FAILED,
            message=f"Element {manager.element_id} has no type hypotheses",
            stage="hypothesis_engine",
            recoverable=True,
            details={"element_id": manager.element_id},
        ))

    if not manager.motion_primitives.all_active():
        errors.append(StructuredError(
            failure_mode=FailureMode.MOTION_AMBIGUOUS,
            message=f"Element {manager.element_id} has no motion hypotheses",
            stage="hypothesis_engine",
            recoverable=True,
            details={"element_id": manager.element_id},
        ))

    return errors
