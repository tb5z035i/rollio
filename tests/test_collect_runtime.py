"""End-to-end tests for the asynchronous collection runtime."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rollio.collect import AsyncCollectionRuntime
from rollio.config import (
    AsyncPipelineConfig,
    CameraConfig,
    EncoderConfig,
    RollioConfig,
    RobotConfig,
    StorageConfig,
    TeleopPairConfig,
    UploadConfig,
)
from rollio.robot import PseudoRobotArm


class LeaderTrajectoryDriver(threading.Thread):
    """Drives a pseudo leader along a smooth scripted trajectory."""

    def __init__(
        self,
        robot: PseudoRobotArm,
        stop_event: threading.Event,
        phase_offset: float,
    ) -> None:
        super().__init__(daemon=True)
        self._robot = robot
        self._stop_event = stop_event
        self._phase_offset = phase_offset

    def run(self) -> None:
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            t = time.monotonic() - t0
            q = np.array([
                0.4 * np.sin(t * 1.1 + self._phase_offset),
                0.35 * np.cos(t * 0.9 + self._phase_offset),
                0.25 * np.sin(t * 1.3 + 0.5 + self._phase_offset),
                0.2 * np.cos(t * 0.7 + 0.3 + self._phase_offset),
                0.15 * np.sin(t * 1.5 + 0.8 + self._phase_offset),
                0.1 * np.cos(t * 1.7 + self._phase_offset),
            ], dtype=np.float64)
            self._robot.set_joint_position(q)
            if self._stop_event.wait(0.01):
                return


def _build_runtime(root: Path, export_delay_sec: float = 1.0) -> AsyncCollectionRuntime:
    cfg = RollioConfig(
        project_name="async_simulated_collection",
        fps=10,
        cameras=[
            CameraConfig(name="cam_left", type="pseudo", width=160, height=120, fps=10),
            CameraConfig(name="cam_right", type="pseudo", width=160, height=120, fps=10),
        ],
        robots=[
            RobotConfig(name="leader_a", type="pseudo", role="leader", num_joints=6),
            RobotConfig(name="leader_b", type="pseudo", role="leader", num_joints=6),
            RobotConfig(name="follower_a", type="pseudo", role="follower", num_joints=6),
            RobotConfig(name="follower_b", type="pseudo", role="follower", num_joints=6),
        ],
        teleop_pairs=[
            TeleopPairConfig(
                name="direct_pair",
                leader="leader_a",
                follower="follower_a",
                mapper="joint_direct",
            ),
            TeleopPairConfig(
                name="pose_pair",
                leader="leader_b",
                follower="follower_b",
                mapper="pose_fk_ik",
            ),
        ],
        storage=StorageConfig(root=str(root), lerobot_version="v2.1"),
        encoder=EncoderConfig(video_codec="mp4v"),
        upload=UploadConfig(enabled=False),
        async_pipeline=AsyncPipelineConfig(
            export_queue_size=4,
            telemetry_hz=40,
            control_hz=100,
        ),
    )
    return AsyncCollectionRuntime.from_config(cfg, export_delay_sec=export_delay_sec)


def _last_position(episode, robot_name: str) -> np.ndarray:
    return episode.data.robot_states[robot_name][-1][1]["position"]


def _last_ee_position(episode, robot_name: str) -> np.ndarray:
    return episode.data.robot_states[robot_name][-1][1]["ee_position"]


def _first_ee_position(episode, robot_name: str) -> np.ndarray:
    return episode.data.robot_states[robot_name][0][1]["ee_position"]


def test_async_runtime_exports_in_background(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, export_delay_sec=1.0)
    driver_stop = threading.Event()
    try:
        runtime.open()
        drivers = [
            LeaderTrajectoryDriver(runtime._robots["leader_a"], driver_stop, 0.0),  # noqa: SLF001
            LeaderTrajectoryDriver(runtime._robots["leader_b"], driver_stop, 1.0),  # noqa: SLF001
        ]
        for driver in drivers:
            driver.start()

        time.sleep(0.3)

        runtime.start_episode()
        time.sleep(0.8)
        episode_0 = runtime.stop_episode()
        export_0 = runtime.keep_episode(episode_0)

        episode_1_start = time.monotonic()
        runtime.start_episode()
        time.sleep(0.8)
        episode_1 = runtime.stop_episode()
        export_1 = runtime.keep_episode(episode_1)

        runtime.wait_for_exports()

        assert export_0.done_event.is_set()
        assert export_1.done_event.is_set()
        assert export_0.error is None
        assert export_1.error is None
        assert export_0.finished_at is not None
        assert episode_1_start < export_0.finished_at

        dataset_root = tmp_path / "async_simulated_collection"
        assert (dataset_root / "data" / "chunk-000" / "episode_000000.parquet").exists()
        assert (dataset_root / "data" / "chunk-000" / "episode_000001.parquet").exists()
        assert (dataset_root / "videos" / "chunk-000" / "cam_left" / "episode_000000.mp4").exists()
        assert (dataset_root / "videos" / "chunk-000" / "cam_right" / "episode_000001.mp4").exists()

        info = (dataset_root / "meta" / "info.json").read_text()
        assert '"fps": 10' in info

        assert episode_0.mapper_modes["direct_pair"] == "joint_direct"
        assert episode_0.mapper_modes["pose_pair"] == "pose_fk_ik"
        assert episode_1.mapper_modes["direct_pair"] == "joint_direct"
        assert episode_1.mapper_modes["pose_pair"] == "pose_fk_ik"

        ep0_table = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000000.parquet")
        ep1_table = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000001.parquet")
        expected_rows_ep0 = int(round(episode_0.duration * 10)) + 1
        expected_rows_ep1 = int(round(episode_1.duration * 10)) + 1
        assert ep0_table.num_rows == expected_rows_ep0
        assert ep1_table.num_rows == expected_rows_ep1
        assert f'"total_frames": {expected_rows_ep0 + expected_rows_ep1}' in info

        for table in (ep0_table, ep1_table):
            ts = table.column("timestamp").to_pylist()
            effective_hz = (len(ts) - 1) / (ts[-1] - ts[0])
            assert abs(effective_hz - 10.0) < 0.2

        leader_a = _last_position(episode_1, "leader_a")
        follower_a = _last_position(episode_1, "follower_a")
        assert np.linalg.norm(leader_a - follower_a) < 0.35

        initial_follower_b_ee = _first_ee_position(episode_1, "follower_b")
        leader_b_ee = _last_ee_position(episode_1, "leader_b")
        follower_b_ee = _last_ee_position(episode_1, "follower_b")
        final_pose_error = np.linalg.norm(leader_b_ee - follower_b_ee)
        assert np.linalg.norm(follower_b_ee - initial_follower_b_ee) > 0.05
        assert final_pose_error < 0.35
    finally:
        driver_stop.set()
        runtime.close()


def test_runtime_from_config_pairs_leaders_and_followers(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, export_delay_sec=0.0)
    try:
        assert len(runtime._teleop_pairs) == 2  # noqa: SLF001
        assert runtime._teleop_pairs[0].mapper_mode == "joint_direct"  # noqa: SLF001
        assert runtime._teleop_pairs[1].mapper_mode == "pose_fk_ik"  # noqa: SLF001
    finally:
        # Runtime is not opened in this test, but close() remains idempotent.
        runtime.close()
