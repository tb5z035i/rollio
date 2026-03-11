"""Tests for the reusable runtime service layer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from rollio.collect import create_runtime_service
from rollio.config import (
    AsyncPipelineConfig,
    CameraConfig,
    EncoderConfig,
    RollioConfig,
    RobotConfig,
    StorageConfig,
    TeleopPairConfig,
)


def _build_worker_service_config(
    root: Path,
    *,
    project_name: str = "worker_service_collection",
    camera_type: str = "pseudo",
    robot_type: str = "pseudo",
) -> RollioConfig:
    return RollioConfig(
        project_name=project_name,
        fps=10,
        cameras=[
            CameraConfig(
                name="cam_main",
                type=camera_type,
                width=160,
                height=120,
                fps=10,
            )
        ],
        robots=[
            RobotConfig(
                name="leader_arm",
                type=robot_type,
                role="leader",
                num_joints=6,
            ),
            RobotConfig(
                name="follower_arm",
                type=robot_type,
                role="follower",
                num_joints=6,
            ),
        ],
        teleop_pairs=[
            TeleopPairConfig(
                name="pair_main",
                leader="leader_arm",
                follower="follower_arm",
                mapper="joint_direct",
            )
        ],
        storage=StorageConfig(root=str(root), lerobot_version="v2.1"),
        encoder=EncoderConfig(video_codec="mp4v", depth_codec="raw"),
        async_pipeline=AsyncPipelineConfig(
            export_queue_size=2,
            telemetry_hz=40,
            control_hz=80,
        ),
    )


def test_worker_runtime_service_records_under_snapshot_polling(tmp_path: Path) -> None:
    cfg = _build_worker_service_config(tmp_path)
    service = create_runtime_service(
        cfg,
        use_worker=True,
        scheduler_driver="round_robin",
    )
    try:
        service.open()

        deadline = time.monotonic() + 3.0
        latest_snapshot = service.snapshot()
        while latest_snapshot.latest_frames.get("cam_main") is None:
            if time.monotonic() >= deadline:
                raise AssertionError("Worker runtime never produced a preview frame.")
            time.sleep(0.05)
            latest_snapshot = service.snapshot()

        service.start_episode()
        poll_deadline = time.monotonic() + 0.7
        poll_count = 0
        while time.monotonic() < poll_deadline:
            latest_snapshot = service.snapshot()
            poll_count += 1
            json.dumps({"poll": poll_count, "recording": latest_snapshot.recording})
            time.sleep(0.02)

        episode = service.stop_episode()
        export = service.keep_episode(episode)
        assert service.wait_for_episode_export(export.episode_index, timeout=20.0) is True

        dataset_root = tmp_path / "worker_service_collection"
        table = pq.read_table(
            dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
        )
        timestamps = table.column("timestamp").to_pylist()
        effective_hz = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

        assert poll_count >= 10
        assert episode.duration > 0.5
        assert abs(effective_hz - 10.0) < 0.2
        assert latest_snapshot.recording is True
        assert "leader_arm" in latest_snapshot.latest_robot_states
    finally:
        service.close()


def test_worker_runtime_service_bootstrap_entries_support_custom_factories(
    tmp_path: Path,
) -> None:
    cfg = _build_worker_service_config(
        tmp_path,
        project_name="bootstrapped_worker_collection",
        camera_type="worker_test_camera",
        robot_type="worker_test_robot",
    )
    cfg.async_pipeline.worker_bootstrap = ["tests.worker_factory_plugin:register"]
    service = create_runtime_service(
        cfg,
        use_worker=True,
        scheduler_driver="round_robin",
    )
    try:
        service.open()

        deadline = time.monotonic() + 3.0
        snapshot = service.snapshot()
        while snapshot.latest_frames.get("cam_main") is None:
            if time.monotonic() >= deadline:
                raise AssertionError("Bootstrapped worker never produced a preview frame.")
            time.sleep(0.05)
            snapshot = service.snapshot()

        assert "cam_main" in snapshot.latest_frames
        assert "leader_arm" in snapshot.latest_robot_states
        assert snapshot.latest_robot_states["leader_arm"]["position"].shape == (6,)
    finally:
        service.close()
