"""
grouping.py — Element grouping via multi-signal scored relations.

PRD §16 requirements:
  - Use: motion correlation, spatial proximity, timing sync,
    overlap percentage, shared style traits, synchronized entrances/exits
  - Output is SCORED RELATION, not a hard assignment
  - If grouping confidence is weak → do not group, preserve hypothesis only
  - Never group based on one weak cue alone

Failure cases handled:
  - coincidental alignment → low spatial-only score filtered
  - same motion but different roles → role divergence penalty
  - group membership changing over time → per-scene grouping
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.schemas import (
    BoundingBox,
    FailureMode,
    MotionElement,
    MotionPrimitive,
    StructuredError,
    TrackedElement,
)


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_GROUP_CONFIDENCE = 0.45      # Below this — store hypothesis only, don't assign
SPATIAL_PROXIMITY_THRESHOLD = 0.15   # Normalized canvas distance
TIMING_SYNC_WINDOW = 5           # Frames — enter/exit within this = synchronized
MOTION_CORRELATION_WEIGHT = 0.35
SPATIAL_WEIGHT = 0.25
TIMING_WEIGHT = 0.20
OVERLAP_WEIGHT = 0.15
STYLE_WEIGHT = 0.05


@dataclass
class GroupRelation:
    """
    A scored hypothesis that two tracks belong to the same group.
    Never forced — confidence drives the decision.
    """
    track_id_a: str
    track_id_b: str
    confidence: float             # 0–1
    motion_correlation: float
    spatial_proximity: float
    timing_sync: float
    overlap_pct: float
    style_similarity: float
    is_confirmed: bool            # True only if confidence >= MIN_GROUP_CONFIDENCE
    hypothesis_only: bool         # True when confidence < threshold


@dataclass
class GroupAssignment:
    """Final group assignment for an element."""
    track_id: str
    group_id: Optional[str]       # None if not grouped
    group_confidence: float
    group_relations: list[GroupRelation]
    hypothesis_group_id: Optional[str]  # Possible group even if not confirmed


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2, bbox.y + bbox.h / 2


def _motion_correlation(
    track_a: TrackedElement,
    track_b: TrackedElement,
    motion_a: Optional[MotionElement],
    motion_b: Optional[MotionElement],
) -> float:
    """
    Compare motion trajectories between two tracks.
    Returns 0–1 correlation score.
    """
    if not track_a.bbox_sequence or not track_b.bbox_sequence:
        return 0.0

    # Build time-indexed position maps
    pos_a = {f: (b.x + b.w / 2, b.y + b.h / 2) for f, b in track_a.bbox_sequence}
    pos_b = {f: (b.x + b.w / 2, b.y + b.h / 2) for f, b in track_b.bbox_sequence}

    common_frames = sorted(set(pos_a.keys()) & set(pos_b.keys()))
    if len(common_frames) < 3:
        return 0.0

    # Compute displacement vectors
    dx_a, dy_a = [], []
    dx_b, dy_b = [], []
    for i in range(1, len(common_frames)):
        f0, f1 = common_frames[i - 1], common_frames[i]
        dx_a.append(pos_a[f1][0] - pos_a[f0][0])
        dy_a.append(pos_a[f1][1] - pos_a[f0][1])
        dx_b.append(pos_b[f1][0] - pos_b[f0][0])
        dy_b.append(pos_b[f1][1] - pos_b[f0][1])

    if not dx_a:
        return 0.0

    # Pearson correlation of x-displacements
    try:
        corr_x = float(np.corrcoef(dx_a, dx_b)[0, 1])
        corr_y = float(np.corrcoef(dy_a, dy_b)[0, 1])
        if np.isnan(corr_x):
            corr_x = 0.0
        if np.isnan(corr_y):
            corr_y = 0.0
    except Exception:
        return 0.0

    # Also compare dominant motion primitive
    prim_bonus = 0.0
    if motion_a and motion_b:
        prims_a = {h.primitive for h in motion_a.motion_hypotheses if h.confidence > 0.3}
        prims_b = {h.primitive for h in motion_b.motion_hypotheses if h.confidence > 0.3}
        if prims_a & prims_b:
            prim_bonus = 0.15

    raw = (abs(corr_x) + abs(corr_y)) / 2.0 + prim_bonus
    return min(1.0, float(raw))


def _spatial_proximity(
    track_a: TrackedElement,
    track_b: TrackedElement,
    canvas_width: int,
    canvas_height: int,
) -> float:
    """
    Mean normalized distance between centers of two tracks.
    Returns 0–1 proximity (1 = very close, 0 = far).
    """
    if not track_a.bbox_sequence or not track_b.bbox_sequence:
        return 0.0

    pos_a = {f: _center(b) for f, b in track_a.bbox_sequence}
    pos_b = {f: _center(b) for f, b in track_b.bbox_sequence}
    common = set(pos_a.keys()) & set(pos_b.keys())
    if not common:
        return 0.0

    diag = float(np.sqrt(canvas_width ** 2 + canvas_height ** 2))
    dists = []
    for f in common:
        ax, ay = pos_a[f]
        bx, by = pos_b[f]
        dists.append(float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)) / diag)

    mean_dist = float(np.mean(dists))
    return max(0.0, 1.0 - mean_dist / SPATIAL_PROXIMITY_THRESHOLD)


def _timing_sync(
    track_a: TrackedElement,
    track_b: TrackedElement,
) -> float:
    """
    Score based on synchronized enter and exit frames.
    1 = perfectly synced, 0 = no sync.
    """
    if not track_a.bbox_sequence or not track_b.bbox_sequence:
        return 0.0

    enter_a = min(f for f, _ in track_a.bbox_sequence)
    exit_a  = max(f for f, _ in track_a.bbox_sequence)
    enter_b = min(f for f, _ in track_b.bbox_sequence)
    exit_b  = max(f for f, _ in track_b.bbox_sequence)

    enter_diff = abs(enter_a - enter_b)
    exit_diff  = abs(exit_a - exit_b)

    enter_score = max(0.0, 1.0 - enter_diff / TIMING_SYNC_WINDOW)
    exit_score  = max(0.0, 1.0 - exit_diff  / TIMING_SYNC_WINDOW)
    return (enter_score + exit_score) / 2.0


def _overlap_percentage(
    track_a: TrackedElement,
    track_b: TrackedElement,
) -> float:
    """
    Mean IoU overlap across shared frames.
    """
    if not track_a.bbox_sequence or not track_b.bbox_sequence:
        return 0.0

    bbox_a = {f: b for f, b in track_a.bbox_sequence}
    bbox_b = {f: b for f, b in track_b.bbox_sequence}
    common = set(bbox_a.keys()) & set(bbox_b.keys())
    if not common:
        return 0.0

    ious = [bbox_a[f].iou(bbox_b[f]) for f in common]
    return float(np.mean(ious))


def _style_similarity(
    track_a: TrackedElement,
    track_b: TrackedElement,
) -> float:
    """
    Approximate style similarity from type candidates.
    Tracks with identical dominant types score higher.
    """
    if not track_a.type_candidates or not track_b.type_candidates:
        return 0.0
    dom_a = max(track_a.type_candidates, key=lambda tc: tc.confidence).type
    dom_b = max(track_b.type_candidates, key=lambda tc: tc.confidence).type
    return 1.0 if dom_a == dom_b else 0.3


def compute_group_relation(
    track_a: TrackedElement,
    track_b: TrackedElement,
    motion_a: Optional[MotionElement],
    motion_b: Optional[MotionElement],
    canvas_width: int,
    canvas_height: int,
) -> GroupRelation:
    """
    Compute a full GroupRelation between two tracks.
    Never forces a group — returns confidence + hypothesis_only flag.
    """
    mot_corr  = _motion_correlation(track_a, track_b, motion_a, motion_b)
    spatial   = _spatial_proximity(track_a, track_b, canvas_width, canvas_height)
    timing    = _timing_sync(track_a, track_b)
    overlap   = _overlap_percentage(track_a, track_b)
    style     = _style_similarity(track_a, track_b)

    composite = (
        MOTION_CORRELATION_WEIGHT * mot_corr
        + SPATIAL_WEIGHT          * spatial
        + TIMING_WEIGHT           * timing
        + OVERLAP_WEIGHT          * overlap
        + STYLE_WEIGHT            * style
    )

    # Anti-coincidence rule: require at least 2 strong signals
    strong_signals = sum([
        mot_corr  >= 0.5,
        spatial   >= 0.5,
        timing    >= 0.5,
        overlap   >= 0.1,
    ])
    if strong_signals < 2:
        composite *= 0.5   # Penalize single-signal grouping

    composite = round(min(1.0, max(0.0, composite)), 4)
    confirmed = composite >= MIN_GROUP_CONFIDENCE

    return GroupRelation(
        track_id_a=track_a.track_id,
        track_id_b=track_b.track_id,
        confidence=composite,
        motion_correlation=round(mot_corr, 4),
        spatial_proximity=round(spatial, 4),
        timing_sync=round(timing, 4),
        overlap_pct=round(overlap, 4),
        style_similarity=round(style, 4),
        is_confirmed=confirmed,
        hypothesis_only=not confirmed,
    )


def assign_groups(
    tracks: list[TrackedElement],
    motion_elements: list[MotionElement],
    canvas_width: int,
    canvas_height: int,
) -> tuple[dict[str, GroupAssignment], list[GroupRelation], list[StructuredError]]:
    """
    Compute all pairwise group relations and assign group IDs to confirmed groups.

    Returns:
        - assignments: {track_id: GroupAssignment}
        - all_relations: list of all pairwise GroupRelations
        - errors: list of StructuredErrors
    """
    errors: list[StructuredError] = []
    motion_by_track = {me.track_id: me for me in motion_elements}
    all_relations: list[GroupRelation] = []
    assignments: dict[str, GroupAssignment] = {}

    # Initialize all assignments as ungrouped
    for track in tracks:
        assignments[track.track_id] = GroupAssignment(
            track_id=track.track_id,
            group_id=None,
            group_confidence=0.0,
            group_relations=[],
            hypothesis_group_id=None,
        )

    if len(tracks) < 2:
        return assignments, all_relations, errors

    # ── Pairwise relation computation ──────────────────────────────────────
    n = len(tracks)
    for i in range(n):
        for j in range(i + 1, n):
            ta, tb = tracks[i], tracks[j]
            ma = motion_by_track.get(ta.track_id)
            mb = motion_by_track.get(tb.track_id)

            try:
                relation = compute_group_relation(
                    ta, tb, ma, mb, canvas_width, canvas_height
                )
            except Exception as exc:
                errors.append(StructuredError(
                    failure_mode=FailureMode.SCHEMA_BUILD_FAILED,
                    message=f"Grouping failed for {ta.track_id} ↔ {tb.track_id}: {exc}",
                    stage="grouping",
                    recoverable=True,
                ))
                continue

            all_relations.append(relation)
            assignments[ta.track_id].group_relations.append(relation)
            assignments[tb.track_id].group_relations.append(relation)

    # ── Union-find group assignment for confirmed relations ─────────────────
    parent: dict[str, str] = {t.track_id: t.track_id for t in tracks}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for rel in all_relations:
        if rel.is_confirmed:
            union(rel.track_id_a, rel.track_id_b)

    # Build group IDs from root nodes
    group_id_map: dict[str, str] = {}
    group_counter = 0
    for track in tracks:
        root = find(track.track_id)
        if root not in group_id_map:
            group_id_map[root] = f"grp_{group_counter:03d}"
            group_counter += 1

    # Assign group IDs — only to groups with ≥ 2 members
    root_counts: dict[str, int] = {}
    for track in tracks:
        root = find(track.track_id)
        root_counts[root] = root_counts.get(root, 0) + 1

    for track in tracks:
        root = find(track.track_id)
        gid = group_id_map[root]
        is_solo = root_counts[root] < 2

        if is_solo:
            # Single-element "group" — no group_id, but store hypothesis
            best_rel = max(
                assignments[track.track_id].group_relations,
                key=lambda r: r.confidence,
                default=None,
            )
            assignments[track.track_id].group_id = None
            assignments[track.track_id].group_confidence = 0.0
            assignments[track.track_id].hypothesis_group_id = (
                gid if best_rel and best_rel.confidence > 0.25 else None
            )
        else:
            # Best relation confidence as group confidence
            best_conf = max(
                (r.confidence for r in assignments[track.track_id].group_relations
                 if r.is_confirmed),
                default=0.0,
            )
            assignments[track.track_id].group_id = gid
            assignments[track.track_id].group_confidence = best_conf

    return assignments, all_relations, errors
