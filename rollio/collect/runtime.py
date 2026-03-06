"""Asynchronous collection runtime built on top of existing device interfaces."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rollio.config.schema import RollioConfig, TeleopPairConfig
from rollio.episode.recorder import EpisodeData
from rollio.episode.writer import LeRobotV21Writer
from rollio.robot import AIRBOTPlay, RobotArm
from rollio.robot.pseudo_robot import PseudoRobotArm
from rollio.sensors import ImageSensor, PseudoCamera, RealSenseCamera, V4L2Camera

from .teleop import MapperMode, TeleopMapper, build_mapper


@dataclass
class ExportRecord:
    """Tracks background export progress for one kept episode."""

    episode_index: int
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    output_path: Path | None = None
    error: str | None = None
    done_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class RecordedEpisode:
    """One completed episode plus runtime metadata."""

    data: EpisodeData
    started_at: float
    stopped_at: float
    mapper_modes: dict[str, str]


@dataclass
class TeleopPairBinding:
    """A resolved tele-operation pair."""

    name: str
    leader_name: str
    follower_name: str
    leader: RobotArm
    follower: RobotArm
    mapper_mode: MapperMode = "auto"
    kp: float = 40.0
    kd: float = 8.0

    def mapper(self) -> TeleopMapper:
        return build_mapper(self.mapper_mode)


class EpisodeAccumulator:
    """Thread-safe accumulation buffer for one in-flight episode."""

    def __init__(
        self,
        episode_index: int,
        fps: int,
        camera_names: list[str],
        robot_names: list[str],
        started_at: float,
        initial_mapper_modes: dict[str, str] | None = None,
    ) -> None:
        self._episode_index = episode_index
        self._fps = fps
        self._started_at = started_at
        self._lock = threading.Lock()
        self._camera_frames: dict[str, list[tuple[float, np.ndarray]]] = {
            name: [] for name in camera_names
        }
        self._robot_states: dict[str, list[tuple[float, dict[str, np.ndarray]]]] = {
            name: [] for name in robot_names
        }
        self._mapper_modes = dict(initial_mapper_modes or {})

    def append_camera(self, name: str, ts: float, frame: np.ndarray) -> None:
        rel_ts = ts - self._started_at
        if rel_ts < 0:
            return
        with self._lock:
            self._camera_frames[name].append((rel_ts, frame.copy()))

    def append_robot(self, name: str, ts: float, state: dict[str, np.ndarray]) -> None:
        rel_ts = ts - self._started_at
        if rel_ts < 0:
            return
        safe_state = {
            key: np.asarray(value).copy()
            for key, value in state.items()
        }
        with self._lock:
            self._robot_states[name].append((rel_ts, safe_state))

    def record_mapper_mode(self, pair_name: str, mapper_mode: str) -> None:
        with self._lock:
            self._mapper_modes[pair_name] = mapper_mode

    def freeze(self, stopped_at: float) -> RecordedEpisode:
        duration = max(0.0, stopped_at - self._started_at)
        with self._lock:
            episode = EpisodeData(
                episode_index=self._episode_index,
                fps=self._fps,
                duration=duration,
                camera_frames={
                    key: list(value) for key, value in self._camera_frames.items()
                },
                robot_states={
                    key: list(value) for key, value in self._robot_states.items()
                },
            )
            mapper_modes = dict(self._mapper_modes)
        return RecordedEpisode(
            data=episode,
            started_at=self._started_at,
            stopped_at=stopped_at,
            mapper_modes=mapper_modes,
        )


class AsyncEpisodeExporter:
    """Background exporter using blocking queues instead of busy waiting."""

    def __init__(
        self,
        root: str | Path,
        project_name: str,
        fps: int,
        queue_size: int = 4,
        export_delay_sec: float = 0.0,
        lerobot_version: str = "v2.1",
    ) -> None:
        if lerobot_version != "v2.1":
            raise NotImplementedError("Only LeRobot v2.1 export is implemented")
        self._writer = LeRobotV21Writer(root=root, project_name=project_name, fps=fps)
        self._queue: queue.Queue[
            tuple[RecordedEpisode | None, ExportRecord | None]
        ] = queue.Queue(maxsize=max(1, queue_size))
        self._records: dict[int, ExportRecord] = {}
        self._export_delay_sec = max(0.0, export_delay_sec)
        self._thread = threading.Thread(
            target=self._run,
            name="rollio-exporter",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    def submit(self, episode: RecordedEpisode) -> ExportRecord:
        record = ExportRecord(
            episode_index=episode.data.episode_index,
            submitted_at=time.monotonic(),
        )
        self._records[record.episode_index] = record
        self._queue.put((episode, record))
        return record

    def wait_for_episode(self, episode_index: int, timeout: float | None = None) -> bool:
        record = self._records.get(episode_index)
        if record is None:
            return False
        return record.done_event.wait(timeout)

    def join(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        if not self._started:
            return
        self._queue.put((None, None))
        self._thread.join(timeout=30.0)
        self._started = False

    def records(self) -> dict[int, ExportRecord]:
        return dict(self._records)

    def _run(self) -> None:
        while True:
            episode, record = self._queue.get()
            try:
                if episode is None or record is None:
                    return
                record.started_at = time.monotonic()
                if self._export_delay_sec:
                    time.sleep(self._export_delay_sec)
                record.output_path = self._writer.write(episode.data)
                record.finished_at = time.monotonic()
            except Exception as exc:  # pragma: no cover - surfaced in tests
                record.error = str(exc)
                record.finished_at = time.monotonic()
            finally:
                if record is not None:
                    record.done_event.set()
                self._queue.task_done()


class PeriodicWorker(threading.Thread):
    """Runs a periodic task using blocking waits."""

    def __init__(self, name: str, interval_sec: float, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self._interval_sec = max(interval_sec, 1e-4)
        self._stop_event = stop_event

    def step(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            self.step()
            next_tick += self._interval_sec
            remaining = next_tick - time.monotonic()
            if remaining <= 0:
                next_tick = time.monotonic()
                continue
            if self._stop_event.wait(remaining):
                return


class CameraCaptureWorker(PeriodicWorker):
    """Captures frames from an ImageSensor."""

    def __init__(
        self,
        camera_name: str,
        camera: ImageSensor,
        runtime: "AsyncCollectionRuntime",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            name=f"camera-{camera_name}",
            interval_sec=1.0 / max(camera.fps, 1),
            stop_event=stop_event,
        )
        self._camera_name = camera_name
        self._camera = camera
        self._runtime = runtime

    def step(self) -> None:
        ts, frame = self._camera.read()
        self._runtime.update_latest_frame(self._camera_name, frame)
        self._runtime.record_camera_frame(self._camera_name, ts, frame)


class RobotTelemetryWorker(PeriodicWorker):
    """Captures robot state into the current episode."""

    def __init__(
        self,
        robot_name: str,
        robot: RobotArm,
        hz: int,
        runtime: "AsyncCollectionRuntime",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            name=f"robot-{robot_name}",
            interval_sec=1.0 / max(hz, 1),
            stop_event=stop_event,
        )
        self._robot_name = robot_name
        self._robot = robot
        self._runtime = runtime

    def step(self) -> None:
        joint_state = self._robot.read_joint_state()
        robot_state: dict[str, np.ndarray] = {}
        if joint_state.position is not None:
            robot_state["position"] = joint_state.position
            pose = self._robot.kinematics.forward_kinematics(joint_state.position)
            robot_state["ee_position"] = pose.position.astype(np.float32)
            robot_state["ee_quaternion"] = pose.quaternion.astype(np.float32)
        if joint_state.velocity is not None:
            robot_state["velocity"] = joint_state.velocity
        if joint_state.effort is not None:
            robot_state["effort"] = joint_state.effort
        self._runtime.update_latest_robot_state(self._robot_name, robot_state)
        self._runtime.record_robot_state(self._robot_name, joint_state.timestamp, robot_state)


class TeleopWorker(PeriodicWorker):
    """Drives one follower from one leader using the selected mapper."""

    def __init__(
        self,
        pair: TeleopPairBinding,
        hz: int,
        runtime: "AsyncCollectionRuntime",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            name=f"teleop-{pair.name}",
            interval_sec=1.0 / max(hz, 1),
            stop_event=stop_event,
        )
        self._pair = pair
        self._runtime = runtime
        self._mapper = pair.mapper()
        self._last_target: np.ndarray | None = None

    def step(self) -> None:
        command = self._mapper.map_command(
            self._pair.leader,
            self._pair.follower,
            previous_target=self._last_target,
        )
        self._last_target = command.position_target
        self._runtime.update_pair_mode(self._pair.name, command.mode)
        self._pair.follower.step_target_tracking(
            position_target=command.position_target,
            velocity_target=command.velocity_target,
            kp=self._pair.kp,
            kd=self._pair.kd,
            add_gravity_compensation=True,
        )


class AsyncCollectionRuntime:
    """Async collection runtime with background export workers."""

    def __init__(
        self,
        cameras: dict[str, ImageSensor],
        robots: dict[str, RobotArm],
        teleop_pairs: list[TeleopPairBinding],
        fps: int,
        export_root: str | Path,
        project_name: str,
        lerobot_version: str = "v2.1",
        export_queue_size: int = 4,
        telemetry_hz: int = 50,
        control_hz: int = 100,
        export_delay_sec: float = 0.0,
    ) -> None:
        self._cameras = cameras
        self._robots = robots
        self._teleop_pairs = teleop_pairs
        self._fps = fps
        self._telemetry_hz = telemetry_hz
        self._control_hz = control_hz
        self._state_lock = threading.Lock()
        self._current_episode: EpisodeAccumulator | None = None
        self._pending_episode: RecordedEpisode | None = None
        self._episode_index = 0
        self._latest_frames: dict[str, np.ndarray | None] = {
            name: None for name in cameras
        }
        self._latest_robot_states: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in robots
        }
        self._latest_pair_modes: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._opened = False
        self._exporter = AsyncEpisodeExporter(
            root=export_root,
            project_name=project_name,
            fps=fps,
            queue_size=export_queue_size,
            export_delay_sec=export_delay_sec,
            lerobot_version=lerobot_version,
        )

    @classmethod
    def from_config(
        cls,
        cfg: RollioConfig,
        export_delay_sec: float = 0.0,
    ) -> "AsyncCollectionRuntime":
        cameras = build_cameras_from_config(cfg)
        robots = build_robots_from_config(cfg)
        teleop_pairs = build_teleop_pairs_from_config(cfg, robots)
        return cls(
            cameras=cameras,
            robots=robots,
            teleop_pairs=teleop_pairs,
            fps=cfg.fps,
            export_root=cfg.storage.root,
            project_name=cfg.project_name,
            lerobot_version=cfg.storage.lerobot_version,
            export_queue_size=cfg.async_pipeline.export_queue_size,
            telemetry_hz=cfg.async_pipeline.telemetry_hz,
            control_hz=cfg.async_pipeline.control_hz,
            export_delay_sec=export_delay_sec,
        )

    @property
    def recording(self) -> bool:
        with self._state_lock:
            return self._current_episode is not None

    def open(self) -> None:
        if self._opened:
            return
        for camera in self._cameras.values():
            camera.open()
        for robot in self._robots.values():
            robot.open()
            robot.enable()
        for pair in self._teleop_pairs:
            pair.leader.enter_free_drive()
            pair.follower.enter_target_tracking()

        self._exporter.start()

        self._workers = [
            CameraCaptureWorker(name, camera, self, self._stop_event)
            for name, camera in self._cameras.items()
        ]
        self._workers.extend(
            RobotTelemetryWorker(name, robot, self._telemetry_hz, self, self._stop_event)
            for name, robot in self._robots.items()
        )
        self._workers.extend(
            TeleopWorker(pair, self._control_hz, self, self._stop_event)
            for pair in self._teleop_pairs
        )

        for worker in self._workers:
            worker.start()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            self._exporter.shutdown()
            return
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._exporter.shutdown()
        for camera in self._cameras.values():
            camera.close()
        for robot in self._robots.values():
            robot.disable()
            robot.close()
        self._opened = False

    def start_episode(self) -> int:
        with self._state_lock:
            if self._current_episode is not None:
                raise RuntimeError("Episode already recording")
            started_at = time.monotonic()
            episode = EpisodeAccumulator(
                episode_index=self._episode_index,
                fps=self._fps,
                camera_names=list(self._cameras.keys()),
                robot_names=list(self._robots.keys()),
                started_at=started_at,
                initial_mapper_modes=self._latest_pair_modes,
            )
            latest_frames = {
                name: frame.copy() if frame is not None else None
                for name, frame in self._latest_frames.items()
            }
            latest_robot_states = {
                name: {
                    key: np.asarray(value).copy()
                    for key, value in state.items()
                }
                for name, state in self._latest_robot_states.items()
            }
            self._current_episode = episode
            self._episode_index += 1
            episode_index = episode._episode_index  # noqa: SLF001

        for name, frame in latest_frames.items():
            if frame is not None:
                episode.append_camera(name, started_at, frame)
        for name, state in latest_robot_states.items():
            if state:
                episode.append_robot(name, started_at, state)
        return episode_index

    def stop_episode(self) -> RecordedEpisode:
        with self._state_lock:
            if self._current_episode is None:
                raise RuntimeError("No episode is recording")
            episode = self._current_episode
            self._current_episode = None
            stopped_at = time.monotonic()
        recorded = episode.freeze(stopped_at)
        with self._state_lock:
            self._pending_episode = recorded
        return recorded

    def keep_episode(self, episode: RecordedEpisode | None = None) -> ExportRecord:
        with self._state_lock:
            if episode is None:
                episode = self._pending_episode
            if episode is None:
                raise RuntimeError("No pending episode to keep")
            if self._pending_episode is episode:
                self._pending_episode = None
        return self._exporter.submit(episode)

    def discard_episode(self, episode: RecordedEpisode | None = None) -> None:
        with self._state_lock:
            if episode is None or self._pending_episode is episode:
                self._pending_episode = None

    def wait_for_exports(self) -> None:
        self._exporter.join()

    def wait_for_episode_export(self, episode_index: int, timeout: float | None = None) -> bool:
        return self._exporter.wait_for_episode(episode_index, timeout)

    def export_records(self) -> dict[int, ExportRecord]:
        return self._exporter.records()

    def latest_frames(self) -> dict[str, np.ndarray | None]:
        with self._state_lock:
            return dict(self._latest_frames)

    def latest_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        with self._state_lock:
            return {name: dict(state) for name, state in self._latest_robot_states.items()}

    def update_latest_frame(self, name: str, frame: np.ndarray) -> None:
        with self._state_lock:
            self._latest_frames[name] = frame

    def update_latest_robot_state(self, name: str, state: dict[str, np.ndarray]) -> None:
        with self._state_lock:
            self._latest_robot_states[name] = {
                key: np.asarray(value).copy() for key, value in state.items()
            }

    def update_pair_mode(self, pair_name: str, mapper_mode: str) -> None:
        with self._state_lock:
            self._latest_pair_modes[pair_name] = mapper_mode
            episode = self._current_episode
        if episode is not None:
            episode.record_mapper_mode(pair_name, mapper_mode)

    def record_camera_frame(self, name: str, ts: float, frame: np.ndarray) -> None:
        with self._state_lock:
            episode = self._current_episode
        if episode is not None:
            episode.append_camera(name, ts, frame)

    def record_robot_state(self, name: str, ts: float, state: dict[str, np.ndarray]) -> None:
        with self._state_lock:
            episode = self._current_episode
        if episode is not None:
            episode.append_robot(name, ts, state)


def build_cameras_from_config(cfg: RollioConfig) -> dict[str, ImageSensor]:
    """Instantiate cameras from the existing setup-stage config."""

    cameras: dict[str, ImageSensor] = {}
    for cam_cfg in cfg.cameras:
        if cam_cfg.type == "pseudo":
            cameras[cam_cfg.name] = PseudoCamera(
                name=cam_cfg.name,
                width=cam_cfg.width,
                height=cam_cfg.height,
                fps=cam_cfg.fps,
            )
            continue
        if cam_cfg.type == "v4l2":
            cameras[cam_cfg.name] = V4L2Camera(
                name=cam_cfg.name,
                device=cam_cfg.device,
                width=cam_cfg.width,
                height=cam_cfg.height,
                fps=cam_cfg.fps,
                pixel_format=cam_cfg.pixel_format,
            )
            continue
        if cam_cfg.type == "realsense":
            channel = cam_cfg.channel or "color"
            kwargs: dict[str, Any] = {
                "name": cam_cfg.name,
                "device": str(cam_cfg.device).split(":")[0],
                "enable_color": channel == "color",
                "enable_depth": channel == "depth",
                "enable_infrared": channel == "infrared",
                "preview_channel": channel,
            }
            if channel == "color":
                kwargs.update(width=cam_cfg.width, height=cam_cfg.height, fps=cam_cfg.fps)
            elif channel == "depth":
                kwargs.update(
                    width=cam_cfg.width,
                    height=cam_cfg.height,
                    fps=cam_cfg.fps,
                    depth_width=cam_cfg.width,
                    depth_height=cam_cfg.height,
                    depth_fps=cam_cfg.fps,
                    depth_format=cam_cfg.pixel_format,
                )
            else:
                kwargs.update(
                    width=cam_cfg.width,
                    height=cam_cfg.height,
                    fps=cam_cfg.fps,
                    ir_width=cam_cfg.width,
                    ir_height=cam_cfg.height,
                    ir_fps=cam_cfg.fps,
                    ir_format=cam_cfg.pixel_format,
                )
            cameras[cam_cfg.name] = RealSenseCamera(**kwargs)
            continue
        raise NotImplementedError(f"Unsupported camera type: {cam_cfg.type}")
    return cameras


def build_robots_from_config(cfg: RollioConfig) -> dict[str, RobotArm]:
    """Instantiate robot arms from the existing setup-stage config."""

    robots: dict[str, RobotArm] = {}
    for robot_cfg in cfg.robots:
        if robot_cfg.type == "pseudo":
            robots[robot_cfg.name] = PseudoRobotArm(
                name=robot_cfg.name,
                n_dof=robot_cfg.num_joints,
                noise_level=0.0,
            )
            continue
        if robot_cfg.type == "airbot_play":
            if AIRBOTPlay is None:
                raise ImportError("AIRBOTPlay support is not available in this environment")
            robots[robot_cfg.name] = AIRBOTPlay(can_interface=robot_cfg.device or "can0")
            continue
        raise NotImplementedError(f"Unsupported robot type: {robot_cfg.type}")
    return robots


def build_teleop_pairs_from_config(
    cfg: RollioConfig,
    robots: dict[str, RobotArm],
) -> list[TeleopPairBinding]:
    """Resolve tele-operation pairs from configuration."""

    pair_cfgs: list[TeleopPairConfig]
    if cfg.teleop_pairs:
        pair_cfgs = cfg.teleop_pairs
    else:
        leaders = [robot.name for robot in cfg.robots if robot.role == "leader"]
        followers = [robot.name for robot in cfg.robots if robot.role == "follower"]
        pair_cfgs = [
            TeleopPairConfig(
                name=f"pair_{idx}",
                leader=leader_name,
                follower=follower_name,
                mapper="auto",
            )
            for idx, (leader_name, follower_name) in enumerate(zip(leaders, followers))
        ]

    pairs: list[TeleopPairBinding] = []
    for pair_cfg in pair_cfgs:
        pairs.append(
            TeleopPairBinding(
                name=pair_cfg.name,
                leader_name=pair_cfg.leader,
                follower_name=pair_cfg.follower,
                leader=robots[pair_cfg.leader],
                follower=robots[pair_cfg.follower],
                mapper_mode=pair_cfg.mapper,
            )
        )
    return pairs
