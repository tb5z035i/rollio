"""Tests for pluggable device factories and extensibility hooks."""
from __future__ import annotations

import time

import pytest

from rollio.collect import (
    AsyncCollectionRuntime,
    ThreadedCameraFrameSource,
    build_camera_from_config,
    build_robot_from_config,
    register_camera_factory,
    register_robot_factory,
)
from rollio.config import CameraConfig, EncoderConfig, RollioConfig, RobotConfig, StorageConfig, TeleopPairConfig
from rollio.robot import PseudoRobotArm
from rollio.sensors import PseudoCamera


def _register_test_factories() -> None:
    register_camera_factory(
        "test_custom_camera",
        lambda cfg: PseudoCamera(
            name=f"{cfg.name}-{cfg.options.get('tag', 'camera')}",
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        ),
        replace=True,
    )
    register_robot_factory(
        "test_custom_robot",
        lambda cfg: PseudoRobotArm(
            name=f"{cfg.name}-{cfg.options.get('tag', 'robot')}",
            n_dof=cfg.num_joints,
            noise_level=float(cfg.options.get("noise_level", 0.0)),
        ),
        replace=True,
    )


def test_custom_factories_build_devices_from_config() -> None:
    _register_test_factories()

    camera = build_camera_from_config(
        CameraConfig(
            name="cam_ext",
            type="test_custom_camera",
            width=160,
            height=120,
            fps=15,
            options={"tag": "preview"},
        )
    )
    robot = build_robot_from_config(
        RobotConfig(
            name="arm_ext",
            type="test_custom_robot",
            role="leader",
            num_joints=6,
            options={"tag": "leader"},
        )
    )

    assert camera.info().name == "cam_ext-preview"
    assert robot.info.name == "arm_ext-leader"


def test_runtime_accepts_registered_backend_types(tmp_path: Path) -> None:
    _register_test_factories()

    cfg = RollioConfig(
        project_name="extensible_runtime",
        fps=10,
        cameras=[
            CameraConfig(
                name="cam_ext",
                type="test_custom_camera",
                width=160,
                height=120,
                fps=10,
            ),
        ],
        robots=[
            RobotConfig(
                name="leader_ext",
                type="test_custom_robot",
                role="leader",
                num_joints=6,
            ),
            RobotConfig(
                name="follower_ext",
                type="test_custom_robot",
                role="follower",
                num_joints=6,
            ),
        ],
        teleop_pairs=[
            TeleopPairConfig(
                name="pair_ext",
                leader="leader_ext",
                follower="follower_ext",
                mapper="joint_direct",
            ),
        ],
        storage=StorageConfig(root=str(tmp_path)),
        encoder=EncoderConfig(video_codec="mp4v", depth_codec="raw"),
    )

    runtime = AsyncCollectionRuntime.from_config(
        cfg,
        scheduler_driver="round_robin",
    )
    try:
        runtime.open()
        time.sleep(0.15)
        metrics = runtime.scheduler_metrics()

        assert runtime._cameras["cam_ext"].info().name == "cam_ext-camera"  # noqa: SLF001
        assert runtime._robots["leader_ext"].info.name == "leader_ext-robot"  # noqa: SLF001
        assert metrics["driver"].task_metrics["teleop-pair_ext"].run_count > 0
    finally:
        runtime.close()


def test_threaded_camera_frame_source_captures_frames() -> None:
    camera = PseudoCamera(name="cam_thread", width=64, height=48, fps=10)
    frame_source = ThreadedCameraFrameSource("cam_thread", camera, max_pending_frames=2)
    try:
        frame_source.open()
        time.sleep(0.2)
        samples = frame_source.drain_samples()
        metrics = frame_source.metrics()

        assert samples
        assert metrics.captured_frames > 0
    finally:
        frame_source.close()


def test_rollio_config_rejects_duplicate_device_names() -> None:
    with pytest.raises(ValueError, match="Camera names must be unique"):
        RollioConfig(
            cameras=[
                CameraConfig(name="dup_cam", type="pseudo"),
                CameraConfig(name="dup_cam", type="pseudo"),
            ],
        )

    with pytest.raises(ValueError, match="Robot names must be unique"):
        RollioConfig(
            robots=[
                RobotConfig(name="dup_arm", type="pseudo", role="leader"),
                RobotConfig(name="dup_arm", type="pseudo", role="follower"),
            ],
        )
