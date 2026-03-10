"""Tests for replay dataset loading and runtime behavior."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rollio.config.schema import CameraConfig, EncoderConfig, RobotConfig, RollioConfig, StorageConfig
from rollio.episode.recorder import EpisodeData
from rollio.episode.writer import LeRobotV21Writer
from rollio.replay import ReplayRuntime, load_replay_episode


def _build_replay_fixture(tmp_path: Path) -> tuple[RollioConfig, Path]:
    project_name = "replay_demo"
    cfg = RollioConfig(
        project_name=project_name,
        fps=10,
        cameras=[
            CameraConfig(
                name="cam_main",
                type="pseudo",
                width=64,
                height=48,
                fps=10,
            ),
        ],
        robots=[
            RobotConfig(name="leader_arm", type="pseudo", role="leader", num_joints=6),
            RobotConfig(name="follower_arm", type="pseudo", role="follower", num_joints=6),
        ],
        storage=StorageConfig(root=str(tmp_path)),
        encoder=EncoderConfig(video_codec="mp4v", depth_codec="raw"),
    )
    writer = LeRobotV21Writer(
        root=tmp_path,
        project_name=project_name,
        fps=10,
        camera_configs={camera.name: camera for camera in cfg.cameras},
        video_codec="mp4v",
        depth_codec="raw",
    )

    frames: list[tuple[float, np.ndarray]] = []
    leader_states: list[tuple[float, dict[str, np.ndarray]]] = []
    follower_states: list[tuple[float, dict[str, np.ndarray]]] = []
    pair_actions: list[tuple[float, np.ndarray]] = []
    for idx in range(5):
        ts = idx * 0.1
        frame = np.full((48, 64, 3), 20 * idx, dtype=np.uint8)
        leader_pos = np.linspace(0.0, 0.25, 6, dtype=np.float32) + idx * 0.01
        follower_pos = np.linspace(0.0, 0.3, 6, dtype=np.float32) + idx * 0.015
        frames.append((ts, frame))
        leader_states.append(
            (
                ts,
                {
                    "position": leader_pos,
                    "velocity": np.zeros(6, dtype=np.float32),
                },
            )
        )
        follower_states.append(
            (
                ts,
                {
                    "position": follower_pos,
                    "velocity": np.zeros(6, dtype=np.float32),
                },
            )
        )
        pair_actions.append((ts, follower_pos))

    episode = EpisodeData(
        episode_index=3,
        fps=10,
        duration=0.4,
        camera_frames={"cam_main": frames},
        robot_states={
            "leader_arm": leader_states,
            "follower_arm": follower_states,
        },
        pair_actions={"pair_0": pair_actions},
        action_layout=[
            {
                "pair_name": "pair_0",
                "leader": "leader_arm",
                "follower": "follower_arm",
                "mode": "joint_direct",
                "start": 0,
                "stop": 6,
                "dim": 6,
            },
        ],
    )
    writer.write(episode)
    episode_path = (
        tmp_path
        / project_name
        / "data"
        / "chunk-000"
        / "episode_000003.parquet"
    )
    return cfg, episode_path


def test_load_replay_episode_resolves_metadata_and_paths(tmp_path: Path) -> None:
    cfg, episode_path = _build_replay_fixture(tmp_path)

    episode = load_replay_episode(cfg, episode_path)

    assert episode.episode_index == 3
    assert episode.row_count == 5
    assert episode.action.shape == (5, 6)
    assert "cam_main" in episode.camera_streams
    assert episode.camera_streams["cam_main"].path.exists() is True
    assert "follower_arm" in episode.recorded_robot_states
    np.testing.assert_allclose(
        episode.state_at_index("follower_arm", 0)["position"],
        np.linspace(0.0, 0.3, 6, dtype=np.float32),
    )


def test_load_replay_episode_rejects_camera_metadata_mismatch(tmp_path: Path) -> None:
    cfg, episode_path = _build_replay_fixture(tmp_path)
    cfg.cameras.append(CameraConfig(name="extra_cam", type="pseudo"))

    try:
        load_replay_episode(cfg, episode_path)
    except ValueError as exc:
        assert "missing from dataset metadata" in str(exc)
    else:
        raise AssertionError("Expected camera metadata mismatch to fail")


def test_replay_runtime_drives_followers_and_returns_to_start(tmp_path: Path) -> None:
    cfg, episode_path = _build_replay_fixture(tmp_path)
    runtime = ReplayRuntime.from_config(
        cfg,
        episode_path,
        return_duration_sec=1.0,
    )
    try:
        runtime.open()
        runtime.start_playback()

        deadline = time.monotonic() + 3.0
        observed_playback = False
        while time.monotonic() < deadline and not runtime.completed:
            runtime.update()
            recorded = runtime.latest_recorded_robot_states()
            live = runtime.latest_live_robot_states()
            if recorded["follower_arm"] and live["follower_arm"]:
                observed_playback = True
            time.sleep(0.02)

        assert observed_playback is True
        assert runtime.completed is True
        assert runtime.playing is False
        assert runtime.returning is False
        assert runtime.latest_live_robot_states()["leader_arm"]
        assert runtime.latest_live_robot_states()["follower_arm"]
        np.testing.assert_allclose(
            runtime.latest_recorded_robot_states()["follower_arm"]["position"],
            np.linspace(0.0, 0.3, 6, dtype=np.float32),
        )

        follower_robot = runtime._robots["follower_arm"]  # noqa: SLF001
        start_target = np.linspace(0.0, 0.3, 6, dtype=np.float64)
        assert np.linalg.norm(follower_robot.get_raw_position() - start_target) < 0.25
    finally:
        runtime.close()
