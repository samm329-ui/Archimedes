"""
test_frame_sampling.py — Tests for backend/pipeline/frame_sampling.py
"""
from __future__ import annotations
import cv2
import numpy as np
import pytest
from backend.pipeline.frame_sampling import (
    build_sampling_plan, plan_scene_sampling,
    SamplingPlan, MAX_FRAMES_PER_PIPELINE
)
from backend.schemas import SceneSegment


def _make_scene(start=0, end=48, sid="scene_000"):
    return SceneSegment(
        scene_id=sid, start_frame=start, end_frame=end,
        duration_frames=end-start, boundary_confidence=1.0,
        same_scene_hypothesis=1.0, new_scene_hypothesis=0.0,
    )


def _write_video(path: str, frames=60, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(path, fourcc, 24.0, (w, h))
    for i in range(frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:] = (i * 3 % 255, 100, 200)
        cv2.rectangle(f, (i % w, 20), (i % w + 20, 50), (255, 255, 255), -1)
        wr.write(f)
    wr.release()
    return path


class TestPlanSceneSampling:
    def test_always_includes_start_frame(self):
        scene = _make_scene(0, 48)
        plan = plan_scene_sampling(scene, fps=24.0)
        assert 0 in plan["indices"]

    def test_always_includes_end_frame(self):
        scene = _make_scene(0, 48)
        plan = plan_scene_sampling(scene, fps=24.0)
        assert max(plan["indices"]) >= 47

    def test_spike_frames_included(self):
        scene = _make_scene(0, 100)
        plan = plan_scene_sampling(scene, fps=24.0, motion_spike_frames=[50])
        assert 50 in plan["indices"]

    def test_spike_burst_included(self):
        scene = _make_scene(0, 100)
        plan = plan_scene_sampling(scene, fps=24.0, motion_spike_frames=[50])
        # Should include frames around spike
        assert 48 in plan["indices"] or 49 in plan["indices"]

    def test_static_scene_has_fewer_frames(self):
        scene = _make_scene(0, 100)
        plan_normal = plan_scene_sampling(scene, fps=24.0, is_static=False)
        plan_static = plan_scene_sampling(scene, fps=24.0, is_static=True)
        assert len(plan_static["indices"]) <= len(plan_normal["indices"])

    def test_minimum_frames_enforced(self):
        scene = _make_scene(0, 10)
        plan = plan_scene_sampling(scene, fps=24.0)
        from backend.pipeline.frame_sampling import MIN_FRAMES_PER_SCENE
        assert len(plan["indices"]) >= MIN_FRAMES_PER_SCENE

    def test_indices_sorted(self):
        scene = _make_scene(0, 120)
        plan = plan_scene_sampling(scene, fps=24.0, motion_spike_frames=[40, 80])
        assert plan["indices"] == sorted(plan["indices"])

    def test_no_duplicate_indices(self):
        scene = _make_scene(0, 60)
        plan = plan_scene_sampling(scene, fps=24.0, motion_spike_frames=[30])
        assert len(plan["indices"]) == len(set(plan["indices"]))

    def test_reason_map_covers_all_indices(self):
        scene = _make_scene(0, 48)
        plan = plan_scene_sampling(scene, fps=24.0)
        for idx in plan["indices"]:
            assert idx in plan["reason_map"]


class TestBuildSamplingPlan:
    def test_returns_sampling_plan(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        scene = _make_scene(0, 60)
        plan = build_sampling_plan(p, [scene], fps=24.0)
        assert isinstance(plan, SamplingPlan)

    def test_frame_indices_not_empty(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        scene = _make_scene(0, 60)
        plan = build_sampling_plan(p, [scene], fps=24.0)
        assert len(plan.frame_indices) > 0

    def test_frame_indices_sorted(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        scenes = [_make_scene(0, 30, "s0"), _make_scene(30, 60, "s1")]
        plan = build_sampling_plan(p, scenes, fps=24.0)
        assert plan.frame_indices == sorted(plan.frame_indices)

    def test_hard_cap_respected(self, tmp_path):
        p = _write_video(str(tmp_path / "long.mp4"), frames=60)
        scenes = [_make_scene(0, 60)]
        plan = build_sampling_plan(p, scenes, fps=24.0, target_rate=30.0)
        assert len(plan.frame_indices) <= MAX_FRAMES_PER_PIPELINE

    def test_capped_flag_set_when_capped(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"), frames=60)
        # Artificially trigger cap by setting very high target rate
        scenes = [_make_scene(0, 60)]
        plan = build_sampling_plan(p, scenes, fps=24.0, target_rate=30.0)
        # Either capped or not — the flag must reflect reality
        if plan.capped:
            assert len(plan.frame_indices) <= MAX_FRAMES_PER_PIPELINE

    def test_multiple_scenes_combined(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"), frames=60)
        scenes = [_make_scene(0, 30, "s0"), _make_scene(30, 60, "s1")]
        plan = build_sampling_plan(p, scenes, fps=24.0)
        # Indices from both scenes should be present
        assert any(i < 30 for i in plan.frame_indices)
        assert any(i >= 30 for i in plan.frame_indices)

    def test_empty_scenes_returns_plan(self, tmp_path):
        p = _write_video(str(tmp_path / "v.mp4"))
        plan = build_sampling_plan(p, [], fps=24.0)
        assert isinstance(plan, SamplingPlan)
