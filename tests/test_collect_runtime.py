"""End-to-end tests for the asynchronous collection runtime."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rollio.collect import AsyncCollectionRuntime, build_robots_from_config, register_robot_factory
from rollio.collect.runtime import TeleopPairBinding, build_teleop_pairs_from_config
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
from rollio.robot import (
    ControlMode,
    FeedbackCapability,
    FreeDriveCommand,
    JointState,
    KinematicsModel,
    Pose,
    PseudoRobotArm,
    RobotArm,
    RobotInfo,
    TargetTrackingCommand,
)


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


class ScalarLeaderDriver(threading.Thread):
    """Drives a 1-DOF pseudo leader along a scalar trajectory."""

    def __init__(self, robot: PseudoRobotArm, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self._robot = robot
        self._stop_event = stop_event

    def run(self) -> None:
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            t = time.monotonic() - t0
            q = np.array([0.05 * np.sin(t * 1.7)], dtype=np.float64)
            self._robot.set_joint_position(q)
            if self._stop_event.wait(0.01):
                return


class _PreviewLinearKinematics(KinematicsModel):
    @property
    def n_dof(self) -> int:
        return 1

    @property
    def frame_names(self) -> list[str]:
        return ["frame"]

    def forward_kinematics(
        self,
        q: np.ndarray,
        frame: str | None = None,
    ) -> Pose:
        del frame
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return Pose(
            position=np.array([float(q[0]), 0.0, 0.0], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: np.ndarray | None = None,
        frame: str | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        del q_init, frame, max_iterations, tolerance
        return np.array([float(target_pose.position[0])], dtype=np.float64), True

    def jacobian(
        self,
        q: np.ndarray,
        frame: str | None = None,
    ) -> np.ndarray:
        del q, frame
        jacobian = np.zeros((6, 1), dtype=np.float64)
        jacobian[0, 0] = 1.0
        return jacobian

    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        del q, qd, qdd
        return np.zeros(1, dtype=np.float64)


class _PreviewSensitiveEEFRobot(RobotArm):
    def __init__(self, name: str, robot_type: str) -> None:
        self._name = name
        self._robot_type = robot_type
        self._kinematics = _PreviewLinearKinematics()
        self._info = RobotInfo(
            name=name,
            robot_type=robot_type,
            n_dof=1,
            feedback_capabilities={
                FeedbackCapability.POSITION,
                FeedbackCapability.VELOCITY,
                FeedbackCapability.EFFORT,
            },
        )
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
        self._sensor_position = 0.0
        self._visible_position = 0.0
        self._pending_position = 0.0
        self._sync_visible = False
        self.hold_commands = 0
        self.free_drive_commands = 0

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def kinematics(self) -> KinematicsModel:
        return self._kinematics

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def direct_map_allowlist(self) -> tuple[str, ...]:
        if self._robot_type == "airbot_e2b":
            return ("airbot_g2",)
        if self._robot_type == "airbot_g2":
            return ("airbot_e2b",)
        return (self._robot_type,)

    @property
    def preview_control_mode(self) -> ControlMode | None:
        if self._robot_type == "airbot_e2b":
            return ControlMode.FREE_DRIVE
        if self._robot_type == "airbot_g2":
            return ControlMode.TARGET_TRACKING
        return None

    @property
    def preview_requires_keepalive(self) -> bool:
        return self._robot_type == "airbot_g2"

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

    def enable(self) -> bool:
        if not self._is_open:
            return False
        self._is_enabled = True
        return True

    def disable(self) -> None:
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

    def read_joint_state(self) -> JointState:
        self._sensor_position += 0.01
        if self._sync_visible:
            if self._robot_type == "airbot_e2b":
                self._visible_position = self._sensor_position
            elif self._robot_type == "airbot_g2":
                self._visible_position = self._pending_position
            self._sync_visible = False
        return JointState(
            timestamp=time.monotonic(),
            position=np.array([self._visible_position], dtype=np.float32),
            velocity=np.array([0.01], dtype=np.float32),
            effort=np.zeros(1, dtype=np.float32),
            is_valid=True,
        )

    def set_control_mode(self, mode: ControlMode) -> bool:
        if not self._is_enabled:
            return False
        self._control_mode = mode
        return True

    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        del cmd
        if self._control_mode != ControlMode.FREE_DRIVE:
            return
        self.free_drive_commands += 1
        self._sync_visible = True

    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        if self._control_mode != ControlMode.TARGET_TRACKING:
            return
        self.hold_commands += 1
        self._pending_position = float(
            np.asarray(cmd.position_target, dtype=np.float64).reshape(-1)[0]
        )
        self._sync_visible = True


def _build_unpaired_preview_runtime(
    root: Path,
    robot: RobotArm,
    *,
    preview_live_feedback: bool,
) -> AsyncCollectionRuntime:
    return AsyncCollectionRuntime(
        cameras={},
        camera_configs={},
        robots={robot.info.name: robot},
        teleop_pairs=[],
        fps=10,
        export_root=root,
        project_name="preview_only_runtime",
        video_codec="mp4v",
        depth_codec="raw",
        export_delay_sec=0.0,
        telemetry_hz=40,
        control_hz=40,
        scheduler_driver="round_robin",
        preview_live_feedback=preview_live_feedback,
    )


def _build_paired_preview_runtime(
    root: Path,
    leader: RobotArm,
    follower: RobotArm,
) -> AsyncCollectionRuntime:
    pair = TeleopPairBinding(
        name="eef_pair",
        leader_name=leader.info.name,
        follower_name=follower.info.name,
        leader=leader,
        follower=follower,
        mapper_mode="joint_direct",
    )
    return AsyncCollectionRuntime(
        cameras={},
        camera_configs={},
        robots={
            leader.info.name: leader,
            follower.info.name: follower,
        },
        teleop_pairs=[pair],
        fps=10,
        export_root=root,
        project_name="preview_pair_runtime",
        video_codec="mp4v",
        depth_codec="raw",
        export_delay_sec=0.0,
        telemetry_hz=40,
        control_hz=40,
        scheduler_driver="round_robin",
        preview_live_feedback=True,
    )


def _build_runtime(
    root: Path,
    export_delay_sec: float = 1.0,
    *,
    scheduler_driver: str = "asyncio",
) -> AsyncCollectionRuntime:
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
        encoder=EncoderConfig(video_codec="mp4v", depth_codec="raw"),
        upload=UploadConfig(enabled=False),
        async_pipeline=AsyncPipelineConfig(
            export_queue_size=4,
            telemetry_hz=40,
            control_hz=100,
        ),
    )
    return AsyncCollectionRuntime.from_config(
        cfg,
        export_delay_sec=export_delay_sec,
        scheduler_driver=scheduler_driver,
    )


def _build_runtime_with_gripper_pair(root: Path) -> AsyncCollectionRuntime:
    cfg = RollioConfig(
        project_name="mixed_entity_collection",
        fps=10,
        cameras=[CameraConfig(name="cam_main", type="pseudo", width=160, height=120, fps=10)],
        robots=[
            RobotConfig(name="leader_arm", type="pseudo", role="leader", num_joints=6),
            RobotConfig(name="follower_arm", type="pseudo", role="follower", num_joints=6),
            RobotConfig(name="leader_gripper", type="pseudo", role="leader", num_joints=1),
            RobotConfig(name="follower_gripper", type="pseudo", role="follower", num_joints=1),
        ],
        teleop_pairs=[
            TeleopPairConfig(
                name="arm_pair",
                leader="leader_arm",
                follower="follower_arm",
                mapper="joint_direct",
            ),
            TeleopPairConfig(
                name="gripper_pair",
                leader="leader_gripper",
                follower="follower_gripper",
                mapper="joint_direct",
            ),
        ],
        storage=StorageConfig(root=str(root), lerobot_version="v2.1"),
        encoder=EncoderConfig(video_codec="mp4v", depth_codec="raw"),
        upload=UploadConfig(enabled=False),
        async_pipeline=AsyncPipelineConfig(
            export_queue_size=2,
            telemetry_hz=40,
            control_hz=100,
        ),
    )
    return AsyncCollectionRuntime.from_config(
        cfg,
        export_delay_sec=0.0,
        scheduler_driver="round_robin",
    )


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
        assert '"codec": "mpeg4"' in info

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


def test_preview_live_feedback_refreshes_unpaired_airbot_e2b(tmp_path: Path) -> None:
    robot = _PreviewSensitiveEEFRobot("eef_e2b", "airbot_e2b")
    runtime = _build_unpaired_preview_runtime(
        tmp_path,
        robot,
        preview_live_feedback=True,
    )
    try:
        runtime.open()
        time.sleep(0.2)

        state = runtime.latest_robot_states()["eef_e2b"]

        assert robot.control_mode == ControlMode.FREE_DRIVE
        assert robot.free_drive_commands > 0
        assert float(state["position"][0]) > 0.0
    finally:
        runtime.close()


def test_preview_live_feedback_refreshes_unpaired_airbot_g2(tmp_path: Path) -> None:
    robot = _PreviewSensitiveEEFRobot("eef_g2", "airbot_g2")
    runtime = _build_unpaired_preview_runtime(
        tmp_path,
        robot,
        preview_live_feedback=True,
    )
    try:
        runtime.open()
        time.sleep(0.2)

        state = runtime.latest_robot_states()["eef_g2"]

        assert robot.control_mode == ControlMode.TARGET_TRACKING
        assert robot.hold_commands > 0
        assert "position" in state
    finally:
        runtime.close()


def test_preview_live_feedback_is_opt_in_for_unpaired_airbot_g2(tmp_path: Path) -> None:
    robot = _PreviewSensitiveEEFRobot("eef_g2", "airbot_g2")
    runtime = _build_unpaired_preview_runtime(
        tmp_path,
        robot,
        preview_live_feedback=False,
    )
    try:
        runtime.open()
        time.sleep(0.2)

        state = runtime.latest_robot_states()["eef_g2"]

        assert robot.control_mode == ControlMode.DISABLED
        assert robot.hold_commands == 0
        assert float(state["position"][0]) == 0.0
    finally:
        runtime.close()


def test_preview_live_feedback_tracks_g2_from_e2b_pair(tmp_path: Path) -> None:
    leader = _PreviewSensitiveEEFRobot("leader_e2b", "airbot_e2b")
    follower = _PreviewSensitiveEEFRobot("follower_g2", "airbot_g2")
    runtime = _build_paired_preview_runtime(tmp_path, leader, follower)
    try:
        runtime.open()
        time.sleep(0.25)

        latest_states = runtime.latest_robot_states()
        leader_state = latest_states["leader_e2b"]
        follower_state = latest_states["follower_g2"]

        assert leader.control_mode == ControlMode.FREE_DRIVE
        assert follower.control_mode == ControlMode.TARGET_TRACKING
        assert leader.free_drive_commands > 0
        assert follower.hold_commands > 0
        assert float(leader_state["position"][0]) > 0.0
        assert abs(
            float(follower_state["position"][0]) - float(leader_state["position"][0])
        ) < 0.02
    finally:
        runtime.close()


def test_scheduler_metrics_are_exposed_for_asyncio_driver(tmp_path: Path) -> None:
    runtime = _build_runtime(
        tmp_path,
        export_delay_sec=0.0,
        scheduler_driver="asyncio",
    )
    driver_stop = threading.Event()
    try:
        runtime.open()
        driver = LeaderTrajectoryDriver(runtime._robots["leader_a"], driver_stop, 0.0)  # noqa: SLF001
        driver.start()
        time.sleep(0.25)

        metrics = runtime.scheduler_metrics()
        driver_metrics = metrics["driver"]
        camera_metrics = metrics["cameras"]

        assert driver_metrics.driver_name == "asyncio"
        assert driver_metrics.task_metrics["teleop-direct_pair"].run_count > 0
        assert driver_metrics.task_metrics["robot-leader_a"].run_count > 0
        assert driver_metrics.task_metrics["camera-cam_left"].run_count > 0
        assert camera_metrics["cam_left"].captured_frames > 0
    finally:
        driver_stop.set()
        runtime.close()


def test_scheduler_metrics_are_exposed_for_round_robin_driver(tmp_path: Path) -> None:
    runtime = _build_runtime(
        tmp_path,
        export_delay_sec=0.0,
        scheduler_driver="round_robin",
    )
    driver_stop = threading.Event()
    try:
        runtime.open()
        driver = LeaderTrajectoryDriver(runtime._robots["leader_a"], driver_stop, 0.0)  # noqa: SLF001
        driver.start()
        time.sleep(0.25)

        metrics = runtime.scheduler_metrics()
        driver_metrics = metrics["driver"]
        camera_metrics = metrics["cameras"]

        assert driver_metrics.driver_name == "round_robin"
        assert driver_metrics.task_metrics["teleop-direct_pair"].run_count > 0
        assert driver_metrics.task_metrics["robot-leader_a"].run_count > 0
        assert driver_metrics.task_metrics["camera-cam_left"].run_count > 0
        assert camera_metrics["cam_left"].captured_frames > 0
    finally:
        driver_stop.set()
        runtime.close()


def test_runtime_exports_per_entity_shapes_and_flat_actions(tmp_path: Path) -> None:
    runtime = _build_runtime_with_gripper_pair(tmp_path)
    driver_stop = threading.Event()
    try:
        runtime.open()
        drivers = [
            LeaderTrajectoryDriver(runtime._robots["leader_arm"], driver_stop, 0.0),  # noqa: SLF001
            ScalarLeaderDriver(runtime._robots["leader_gripper"], driver_stop),  # noqa: SLF001
        ]
        for driver in drivers:
            driver.start()

        time.sleep(0.25)
        runtime.start_episode()
        time.sleep(0.6)
        episode = runtime.stop_episode()
        export = runtime.keep_episode(episode)
        runtime.wait_for_exports()

        assert export.error is None
        assert [entry["pair_name"] for entry in episode.data.action_layout] == [
            "arm_pair",
            "gripper_pair",
        ]
        assert [entry["dim"] for entry in episode.data.action_layout] == [6, 1]

        dataset_root = tmp_path / "mixed_entity_collection"
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        features = info["features"]
        assert features["observation.state.leader_arm.position"]["shape"] == [6]
        assert features["observation.state.leader_gripper.position"]["shape"] == [1]
        assert features["observation.state.follower_gripper.velocity"]["shape"] == [1]
        assert features["action"]["shape"] == [7]
        assert info["action_layout"][-1]["stop"] == 7

        table = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000000.parquet")
        grip_obs = table.column("observation.state.follower_gripper.position").to_pylist()
        actions = table.column("action").to_pylist()

        assert all(len(row) == 1 for row in grip_obs)
        assert all(len(row) == 7 for row in actions)
    finally:
        driver_stop.set()
        runtime.close()


def test_joint_direct_mapper_returns_noop_when_allowlists_do_not_match() -> None:
    register_robot_factory(
        "test_g2_leader",
        lambda cfg: PseudoRobotArm(name=cfg.name, n_dof=cfg.num_joints),
        replace=True,
    )
    register_robot_factory(
        "test_e2b_follower",
        lambda cfg: PseudoRobotArm(name=cfg.name, n_dof=cfg.num_joints),
        replace=True,
    )

    cfg = RollioConfig(
        project_name="noop_pairing",
        robots=[
            RobotConfig(
                name="leader_g2",
                type="test_g2_leader",
                role="leader",
                num_joints=1,
                direct_map_allowlist=["test_g2_leader"],
            ),
            RobotConfig(
                name="follower_e2b",
                type="test_e2b_follower",
                role="follower",
                num_joints=1,
                direct_map_allowlist=["test_e2b_follower"],
            ),
        ],
        teleop_pairs=[
            TeleopPairConfig(
                name="noop_pair",
                leader="leader_g2",
                follower="follower_e2b",
                mapper="joint_direct",
            ),
        ],
    )

    robots = build_robots_from_config(cfg)
    pair = build_teleop_pairs_from_config(cfg, robots)[0]
    command = pair.mapper().map_command(pair.leader, pair.follower)

    assert command.mode == "noop"
    assert command.position_target is None
    assert command.velocity_target is None
