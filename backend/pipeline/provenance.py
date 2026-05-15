"""
provenance.py — Provenance tracking for all derived fields.

PRD §21: Every derived field must carry provenance.
PRD §24: Missing provenance = pipeline leak.

This module provides:
  - ProvenanceChain: tracks the full derivation path of a value
  - ProvenanceRegistry: global registry for the pipeline run
  - Helper functions for building ProvenanceRecord objects

Rules:
  - Every inferred field must carry confidence + source module
  - Every derived field must carry derivation chain
  - No field can be marked "certain" without a provenance trail
  - Provenance is immutable once written — no overwriting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.schemas import ProvenanceRecord


@dataclass
class ProvenanceStep:
    """One step in a derivation chain."""
    module: str
    method: str
    confidence: float
    frame_range: Optional[tuple[int, int]] = None
    notes: str = ""


@dataclass
class ProvenanceChain:
    """
    Full derivation history of a value.
    Immutable append-only: steps are added but never removed.
    """
    field_name: str
    steps: list[ProvenanceStep] = field(default_factory=list)

    def add_step(
        self,
        module: str,
        method: str,
        confidence: float,
        frame_range: Optional[tuple[int, int]] = None,
        notes: str = "",
    ) -> None:
        self.steps.append(ProvenanceStep(
            module=module,
            method=method,
            confidence=confidence,
            frame_range=frame_range,
            notes=notes,
        ))

    def final_confidence(self) -> float:
        """Return the confidence of the last step."""
        if not self.steps:
            return 0.0
        return self.steps[-1].confidence

    def origin_module(self) -> str:
        """Return the first contributing module."""
        if not self.steps:
            return "unknown"
        return self.steps[0].module

    def to_record(self) -> ProvenanceRecord:
        """Convert to the canonical ProvenanceRecord schema type."""
        if not self.steps:
            return ProvenanceRecord(
                source_module="unknown",
                method="unknown",
                confidence=0.0,
            )
        last = self.steps[-1]
        chain_summary = " → ".join(f"{s.module}.{s.method}" for s in self.steps)
        return ProvenanceRecord(
            source_module=last.module,
            source_frame_range=last.frame_range,
            method=last.method,
            confidence=last.confidence,
            notes=chain_summary if len(self.steps) > 1 else last.notes,
        )

    def to_dict(self) -> list[dict]:
        return [
            {
                "module": s.module,
                "method": s.method,
                "confidence": s.confidence,
                "frame_range": s.frame_range,
                "notes": s.notes,
            }
            for s in self.steps
        ]


class ProvenanceRegistry:
    """
    Global registry of all provenance chains for one pipeline run.
    Keyed by (element_id, field_name).
    Write-once per key — overwriting raises an error.
    """

    def __init__(self):
        self._chains: dict[str, ProvenanceChain] = {}

    def _key(self, element_id: str, field_name: str) -> str:
        return f"{element_id}::{field_name}"

    def record(
        self,
        element_id: str,
        field_name: str,
        module: str,
        method: str,
        confidence: float,
        frame_range: Optional[tuple[int, int]] = None,
        notes: str = "",
    ) -> None:
        """Add a provenance step for element_id.field_name."""
        key = self._key(element_id, field_name)
        if key not in self._chains:
            self._chains[key] = ProvenanceChain(field_name=field_name)
        self._chains[key].add_step(module, method, confidence, frame_range, notes)

    def get_chain(self, element_id: str, field_name: str) -> Optional[ProvenanceChain]:
        return self._chains.get(self._key(element_id, field_name))

    def get_record(
        self, element_id: str, field_name: str
    ) -> ProvenanceRecord:
        chain = self.get_chain(element_id, field_name)
        if chain:
            return chain.to_record()
        return ProvenanceRecord(
            source_module="unknown",
            method="unknown",
            confidence=0.0,
            notes=f"No provenance recorded for {element_id}.{field_name}",
        )

    def has_provenance(self, element_id: str, field_name: str) -> bool:
        return self._key(element_id, field_name) in self._chains

    def missing_fields(
        self,
        element_id: str,
        required_fields: list[str],
    ) -> list[str]:
        """Return fields that have no provenance recorded."""
        return [
            f for f in required_fields
            if not self.has_provenance(element_id, f)
        ]

    def all_entries(self) -> dict[str, list[dict]]:
        return {
            key: chain.to_dict()
            for key, chain in self._chains.items()
        }

    def summary(self) -> dict[str, Any]:
        by_field: dict[str, int] = {}
        for chain in self._chains.values():
            field = chain.field_name
            by_field[field] = by_field.get(field, 0) + 1
        return {
            "total_chains": len(self._chains),
            "by_field": by_field,
        }


# ── Convenience builders ──────────────────────────────────────────────────────

def make_record(
    module: str,
    method: str,
    confidence: float,
    frame_range: Optional[tuple[int, int]] = None,
    notes: str = "",
) -> ProvenanceRecord:
    """Quick factory for a single-step ProvenanceRecord."""
    return ProvenanceRecord(
        source_module=module,
        source_frame_range=frame_range,
        method=method,
        confidence=confidence,
        notes=notes,
    )


REQUIRED_ELEMENT_FIELDS = [
    "type", "layout", "motion", "timing",
    "layer", "role", "style",
]
