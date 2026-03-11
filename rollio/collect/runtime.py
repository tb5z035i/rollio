"""Asynchronous collection runtime built on top of existing device interfaces."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rollio.config import suggest_teleop_pairs
from rollio.config.schema import CameraConfig, RollioConfig, TeleopPairConfig
from rollio.defaults import DEFAULT_CONTROL_HZ
from rollio.episode.recorder import EpisodeData
from rollio.episode.writer import LeRobotV21Writer
from rollio.robot import ControlMode, JointState, RobotArm
from rollio.sensors import ImageSensor

from .camera_bridge import FrameSourceMetrics, ThreadedCameraFrameSource
from .devices import build_cameras_from_config, build_robots_from_config
from .scheduler import DriverMetrics, ScheduledTask, build_scheduler_driver
from .teleop import (
    MapperMode,
    ResolvedMapperMode,
    TeleopMapper,
    build_mapper,
    supports_joint_direct_runtime,
    supports_pose_fkik_runtime,
)


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

    @property
    def episode_index(self) -> int:
        return self.data.episode_index

    @property
    def duration(self) -> float:
        return self.data.duration


@dataclass(frozen=True)
class TimingTrace:
    """Compact timing summary suitable for TUI diagnostics."""

    intervals_ms: tuple[float, ...] = ()
    target_interval_ms: float | None = None
    last_gap_ms: float | None = None
    max_gap_ms: float | None = None
    age_ms: float | None = None


@dataclass(frozen=True)
class RuntimeTimingDiagnostics:
    """Short-term timing histories used by TUI debug visualizations."""

    scheduler_loop: TimingTrace = field(default_factory=TimingTrace)
    telemetry_runs: TimingTrace = field(default_factory=TimingTrace)
    control_runs: TimingTrace = field(default_factory=TimingTrace)
    valid_robot_samples: dict[str, TimingTrace] = field(default_factory=dict)


class _TimingHistory:
    """Bounded monotonic timestamp history for recent interval analysis."""

    def __init__(self, maxlen: int = 64) -> None:
        self._timestamps: deque[float] = deque(maxlen=maxlen)
        self._intervals_ms: deque[float] = deque(maxlen=maxlen)

    def observe(self, timestamp: float | None = None) -> None:
        ts = time.monotonic() if timestamp is None else float(timestamp)
        if self._timestamps:
            self._intervals_ms.append(max(0.0, (ts - self._timestamps[-1]) * 1000.0))
        self._timestamps.append(ts)

    def reset(self) -> None:
        self._timestamps.clear()
        self._intervals_ms.clear()

    def snapshot(
        self,
        *,
        target_interval_ms: float | None = None,
        now: float | None = None,
    ) -> TimingTrace:
        intervals = tuple(self._intervals_ms)
        last_timestamp = self._timestamps[-1] if self._timestamps else None
        current_time = time.monotonic() if now is None else float(now)
        return TimingTrace(
            intervals_ms=intervals,
            target_interval_ms=target_interval_ms,
            last_gap_ms=intervals[-1] if intervals else None,
            max_gap_ms=max(intervals) if intervals else None,
            age_ms=(
                max(0.0, (current_time - last_timestamp) * 1000.0)
                if last_timestamp is not None
                else None
            ),
        )


def _target_interval_ms(rate_hz: int) -> float | None:
    hz = max(int(rate_hz), 0)
    if hz <= 0:
        return None
    return 1000.0 / float(hz)


def _observe_runtime_hook(runtime: object, name: str, *args: object) -> None:
    hook = getattr(runtime, name, None)
    if callable(hook):
        hook(*args)


@dataclass
class TeleopPairBinding:
    """A resolved tele-operation pair."""

    name: str
    leader_name: str
    follower_name: str
    leader: RobotArm
    follower: RobotArm
    mapper_mode: MapperMode = "auto"
    kp: float | np.ndarray | None = None
    kd: float | np.ndarray | None = None

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
        action_layout: list[dict[str, int | str]] | None = None,
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
        self._action_layout = [dict(entry) for entry in (action_layout or [])]
        self._pair_actions: dict[str, list[tuple[float, np.ndarray]]] = {
            str(entry["pair_name"]): [] for entry in self._action_layout
        }

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
        safe_state = {key: np.asarray(value).copy() for key, value in state.items()}
        with self._lock:
            self._robot_states[name].append((rel_ts, safe_state))

    def record_mapper_mode(self, pair_name: str, mapper_mode: str) -> None:
        with self._lock:
            self._mapper_modes[pair_name] = mapper_mode

    def append_pair_action(self, pair_name: str, ts: float, target: np.ndarray) -> None:
        rel_ts = ts - self._started_at
        if rel_ts < 0:
            return
        if pair_name not in self._pair_actions:
            return
        safe_target = np.asarray(target, dtype=np.float32).copy()
        with self._lock:
            self._pair_actions[pair_name].append((rel_ts, safe_target))

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
                pair_actions={
                    key: list(value) for key, value in self._pair_actions.items()
                },
                action_layout=[dict(entry) for entry in self._action_layout],
            )
            mapper_modes = dict(self._mapper_modes)
        return RecordedEpisode(
            data=episode,
            started_at=self._started_at,
            stopped_at=stopped_at,
            mapper_modes=mapper_modes,
        )


@dataclass(frozen=True)
class _ExportWorkerConfig:
    root: str
    project_name: str
    fps: int
    camera_configs: dict[str, dict[str, object]]
    video_codec: str
    depth_codec: str
    export_delay_sec: float = 0.0


def _export_worker_main(
    config: _ExportWorkerConfig,
    request_queue: object,
    result_queue: object,
) -> None:
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        writer = LeRobotV21Writer(
            root=config.root,
            project_name=config.project_name,
            fps=config.fps,
            camera_configs={
                name: CameraConfig.model_validate(payload)
                for name, payload in config.camera_configs.items()
            },
            video_codec=config.video_codec,
            depth_codec=config.depth_codec,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        result_queue.put(("fatal", -1, time.monotonic(), None, str(exc)))
        result_queue.put(None)
        return

    try:
        while True:
            item = request_queue.get()
            if item is None:
                request_queue.task_done()
                return
            episode_index = -1
            try:
                episode_index, episode = item
                result_queue.put(
                    ("started", episode_index, time.monotonic(), None, None)
                )
                if config.export_delay_sec:
                    time.sleep(config.export_delay_sec)
                output_path = writer.write(episode.data)
                result_queue.put(
                    (
                        "finished",
                        episode_index,
                        time.monotonic(),
                        str(output_path),
                        None,
                    ),
                )
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                result_queue.put(
                    ("finished", episode_index, time.monotonic(), None, str(exc)),
                )
            finally:
                request_queue.task_done()
    finally:
        result_queue.put(None)


class AsyncEpisodeExporter:
    """Background exporter isolated in a dedicated worker process."""

    def __init__(
        self,
        root: str | Path,
        project_name: str,
        fps: int,
        camera_configs: dict[str, CameraConfig],
        video_codec: str,
        depth_codec: str,
        queue_size: int = 4,
        pending_queue_size: int = 8,
        export_delay_sec: float = 0.0,
        lerobot_version: str = "v2.1",
    ) -> None:
        if lerobot_version != "v2.1":
            raise NotImplementedError("Only LeRobot v2.1 export is implemented")
        self._ctx = mp.get_context("spawn")
        self._worker_config = _ExportWorkerConfig(
            root=str(root),
            project_name=project_name,
            fps=fps,
            camera_configs={
                name: config.model_dump(mode="python")
                for name, config in camera_configs.items()
            },
            video_codec=video_codec,
            depth_codec=depth_codec,
            export_delay_sec=max(0.0, export_delay_sec),
        )
        self._queue_size = max(1, queue_size)
        self._pending_queue_size = max(1, pending_queue_size)
        self._records: dict[int, ExportRecord] = {}
        self._records_lock = threading.Lock()
        self._pending_queue: queue.Queue[tuple[int, RecordedEpisode] | None] | None = (
            None
        )
        self._request_queue: object | None = None
        self._result_queue: object | None = None
        self._process: mp.Process | None = None
        self._submit_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._pending_queue = queue.Queue(maxsize=self._pending_queue_size)
        self._request_queue = self._ctx.JoinableQueue(maxsize=self._queue_size)
        self._result_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_export_worker_main,
            args=(self._worker_config, self._request_queue, self._result_queue),
            name="rollio-exporter",
        )
        self._process.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_results,
            name="rollio-export-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        self._submit_thread = threading.Thread(
            target=self._submit_pending_requests,
            name="rollio-export-submit",
            daemon=True,
        )
        self._submit_thread.start()
        self._started = True

    def submit(self, episode: RecordedEpisode) -> ExportRecord:
        if not self._started or self._pending_queue is None:
            raise RuntimeError("Export worker is not running")
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError("Export worker stopped unexpectedly")
        record = ExportRecord(
            episode_index=episode.data.episode_index,
            submitted_at=time.monotonic(),
        )
        with self._records_lock:
            self._records[record.episode_index] = record
        self._pending_queue.put((record.episode_index, episode))
        return record

    def wait_for_episode(
        self, episode_index: int, timeout: float | None = None
    ) -> bool:
        with self._records_lock:
            record = self._records.get(episode_index)
        if record is None:
            return False
        return record.done_event.wait(timeout)

    def join(self) -> None:
        while True:
            with self._records_lock:
                pending = [
                    record
                    for record in self._records.values()
                    if not record.done_event.is_set()
                ]
            if not pending:
                return
            if self._process is not None and not self._process.is_alive():
                self._mark_unfinished_records_failed(
                    "Export worker exited unexpectedly",
                )
                return
            pending[0].done_event.wait(0.1)

    def shutdown(self) -> None:
        if not self._started:
            return
        self.join()
        if self._pending_queue is not None:
            self._pending_queue.put(None)
        if self._submit_thread is not None:
            self._submit_thread.join(timeout=5.0)
        if self._process is not None:
            self._process.join(timeout=30.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
        if self._request_queue is not None:
            self._request_queue.close()
            self._request_queue.join_thread()
        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue.join_thread()
        self._pending_queue = None
        self._request_queue = None
        self._result_queue = None
        self._process = None
        self._submit_thread = None
        self._monitor_thread = None
        self._started = False

    def records(self) -> dict[int, ExportRecord]:
        with self._records_lock:
            return dict(self._records)

    def _monitor_results(self) -> None:
        while True:
            if self._result_queue is None:
                return
            try:
                result = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    self._mark_unfinished_records_failed(
                        "Export worker exited unexpectedly",
                    )
                    return
                continue
            if result is None:
                return
            kind, episode_index, timestamp, output_path, error = result
            if kind == "fatal":
                self._mark_unfinished_records_failed(
                    error or "Export worker failed to start"
                )
                continue
            record = self._record_for_episode(int(episode_index))
            if record is None:
                continue
            if kind == "started":
                record.started_at = float(timestamp)
                continue
            if kind == "finished":
                if record.started_at is None:
                    record.started_at = float(timestamp)
                record.finished_at = float(timestamp)
                record.output_path = Path(output_path) if output_path else None
                record.error = error
                record.done_event.set()

    def _submit_pending_requests(self) -> None:
        while True:
            if self._pending_queue is None:
                return
            item = self._pending_queue.get()
            try:
                if item is None:
                    if self._request_queue is not None:
                        self._request_queue.put(None)
                    return
                episode_index, episode = item
                if self._request_queue is None:
                    raise RuntimeError("Export worker request queue is unavailable")
                if self._process is not None and not self._process.is_alive():
                    raise RuntimeError("Export worker stopped unexpectedly")
                self._request_queue.put((episode_index, episode))
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record = self._record_for_episode(int(episode_index))
                if record is not None:
                    finished_at = time.monotonic()
                    if record.started_at is None:
                        record.started_at = finished_at
                    record.finished_at = finished_at
                    record.error = str(exc)
                    record.done_event.set()
            finally:
                self._pending_queue.task_done()

    def _record_for_episode(self, episode_index: int) -> ExportRecord | None:
        with self._records_lock:
            return self._records.get(episode_index)

    def _mark_unfinished_records_failed(self, error: str) -> None:
        finished_at = time.monotonic()
        with self._records_lock:
            pending = [
                record
                for record in self._records.values()
                if not record.done_event.is_set()
            ]
        for record in pending:
            if record.started_at is None:
                record.started_at = finished_at
            record.finished_at = finished_at
            record.error = error
            record.done_event.set()


def _joint_state_to_robot_state(
    robot: RobotArm, joint_state: JointState
) -> dict[str, np.ndarray]:
    robot_state: dict[str, np.ndarray] = {}
    if joint_state.position is not None:
        robot_state["position"] = joint_state.position
        try:
            pose = robot.kinematics.forward_kinematics(joint_state.position)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError):
            pose = None
        if pose is not None:
            robot_state["ee_position"] = pose.position.astype(np.float32)
            robot_state["ee_quaternion"] = pose.quaternion.astype(np.float32)
    if joint_state.velocity is not None:
        robot_state["velocity"] = joint_state.velocity
    if joint_state.effort is not None:
        robot_state["effort"] = joint_state.effort
    control_loop_metrics = getattr(robot, "control_loop_metrics", None)
    if callable(control_loop_metrics):
        try:
            metrics = control_loop_metrics()
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError):
            metrics = None
        if metrics is not None:
            robot_state["control_loop_target_interval_ms"] = np.array(
                [float(metrics.target_interval_ms)],
                dtype=np.float32,
            )
            if metrics.last_interval_ms is not None:
                robot_state["control_loop_interval_ms"] = np.array(
                    [float(metrics.last_interval_ms)],
                    dtype=np.float32,
                )
            if metrics.avg_interval_ms is not None:
                robot_state["control_loop_avg_interval_ms"] = np.array(
                    [float(metrics.avg_interval_ms)],
                    dtype=np.float32,
                )
    return robot_state


def _resolve_pair_mode(pair: TeleopPairBinding) -> ResolvedMapperMode:
    if pair.mapper_mode == "joint_direct":
        return (
            "joint_direct"
            if supports_joint_direct_runtime(pair.leader, pair.follower)
            else "noop"
        )
    if pair.mapper_mode == "pose_fk_ik":
        return (
            "pose_fk_ik"
            if supports_pose_fkik_runtime(pair.leader, pair.follower)
            else "noop"
        )
    if supports_joint_direct_runtime(pair.leader, pair.follower):
        return "joint_direct"
    if supports_pose_fkik_runtime(pair.leader, pair.follower):
        return "pose_fk_ik"
    return "noop"


def _build_action_layout(
    pairs: list[TeleopPairBinding],
) -> list[dict[str, int | str]]:
    layout: list[dict[str, int | str]] = []
    start = 0
    for pair in pairs:
        resolved_mode = _resolve_pair_mode(pair)
        if resolved_mode == "noop":
            continue
        dim = pair.follower.n_dof
        layout.append(
            {
                "pair_name": pair.name,
                "leader": pair.leader_name,
                "follower": pair.follower_name,
                "mode": resolved_mode,
                "start": start,
                "stop": start + dim,
                "dim": dim,
            }
        )
        start += dim
    return layout


def _preview_control_mode(robot: RobotArm) -> ControlMode | None:
    return robot.preview_control_mode


def _requires_preview_keepalive(robot: RobotArm) -> bool:
    return robot.preview_requires_keepalive


class CameraIngestTask:
    """Pull frames from a threaded camera source into runtime caches."""

    def __init__(
        self,
        camera_name: str,
        frame_source: ThreadedCameraFrameSource,
        runtime: "AsyncCollectionRuntime",
    ) -> None:
        self._camera_name = camera_name
        self._frame_source = frame_source
        self._runtime = runtime

    def scheduled_task(self) -> ScheduledTask:
        return ScheduledTask(
            name=f"camera-{self._camera_name}",
            interval_sec=1.0 / max(self._frame_source.fps, 1),
            step=self.step,
        )

    def step(self) -> None:
        samples = self._frame_source.drain_samples()
        if not samples:
            return
        latest = samples[-1]
        self._runtime.update_latest_frame(self._camera_name, latest.frame)
        for sample in samples:
            self._runtime.record_camera_frame(
                self._camera_name,
                sample.timestamp,
                sample.frame,
            )


class RobotTelemetryTask:
    """Capture robot state into runtime caches and the active episode."""

    def __init__(
        self,
        robot_name: str,
        robot: RobotArm,
        hz: int,
        runtime: "AsyncCollectionRuntime",
        *,
        preview_keepalive: bool = False,
    ) -> None:
        self._robot_name = robot_name
        self._robot = robot
        self._interval_sec = 1.0 / max(hz, 1)
        self._runtime = runtime
        self._preview_keepalive = preview_keepalive

    def scheduled_task(self) -> ScheduledTask:
        return ScheduledTask(
            name=f"robot-{self._robot_name}",
            interval_sec=self._interval_sec,
            step=self.step,
        )

    def step(self) -> None:
        if (
            self._runtime._preview_live_feedback  # noqa: SLF001
            and self._robot_name in self._runtime._teleop_leader_names  # noqa: SLF001
        ):
            # In preview, paired leaders are already sampled by the teleop task.
            # Re-reading them here only adds scheduler load and leader-side latency.
            return
        _observe_runtime_hook(self._runtime, "observe_telemetry_run")
        if (
            self._runtime._preview_live_feedback  # noqa: SLF001
            and self._robot.control_mode == ControlMode.FREE_DRIVE
            and self._robot_name
            not in self._runtime._teleop_leader_names  # noqa: SLF001
        ):
            self._robot.step_free_drive()
        joint_state = self._robot.read_joint_state()
        if joint_state.is_valid and joint_state.position is not None:
            _observe_runtime_hook(
                self._runtime,
                "observe_valid_robot_sample",
                self._robot_name,
                joint_state.timestamp,
            )
        if (
            self._preview_keepalive
            and joint_state.is_valid
            and joint_state.position is not None
        ):
            velocity_target = (
                np.asarray(joint_state.velocity, dtype=np.float64)
                if joint_state.velocity is not None
                else np.zeros_like(joint_state.position, dtype=np.float64)
            )
            self._robot.step_target_tracking(
                position_target=np.asarray(joint_state.position, dtype=np.float64),
                velocity_target=velocity_target,
                add_gravity_compensation=False,
            )
        robot_state = _joint_state_to_robot_state(self._robot, joint_state)
        self._runtime.update_latest_robot_state(self._robot_name, robot_state)
        self._runtime.record_robot_state(
            self._robot_name,
            joint_state.timestamp,
            robot_state,
        )


class TeleopTask:
    """Drive one follower from one leader using the selected mapper."""

    def __init__(
        self,
        pair: TeleopPairBinding,
        hz: int,
        runtime: "AsyncCollectionRuntime",
    ) -> None:
        self._pair = pair
        self._interval_sec = 1.0 / max(hz, 1)
        self._runtime = runtime
        self._mapper = pair.mapper()
        self._last_target: np.ndarray | None = None

    def scheduled_task(self) -> ScheduledTask:
        return ScheduledTask(
            name=f"teleop-{self._pair.name}",
            interval_sec=self._interval_sec,
            step=self.step,
        )

    def step(self) -> None:
        # AIRBOT backends replay the latest intent in their own device threads,
        # but we still refresh the newest leader/follower targets here.
        _observe_runtime_hook(self._runtime, "observe_control_run")
        self._pair.leader.step_free_drive()
        command = self._mapper.map_command(
            self._pair.leader,
            self._pair.follower,
            previous_target=self._last_target,
        )
        if (
            command.leader_joint_state is not None
            and command.leader_joint_state.is_valid
            and command.leader_joint_state.position is not None
        ):
            _observe_runtime_hook(
                self._runtime,
                "observe_valid_robot_sample",
                self._pair.leader_name,
                command.leader_joint_state.timestamp,
            )
        if (
            self._runtime._preview_live_feedback  # noqa: SLF001
            and command.leader_joint_state is not None
        ):
            leader_state = _joint_state_to_robot_state(
                self._pair.leader,
                command.leader_joint_state,
            )
            self._runtime.update_latest_robot_state(
                self._pair.leader_name,
                leader_state,
            )
        if command.position_target is None or command.velocity_target is None:
            self._runtime.update_pair_mode(self._pair.name, command.mode)
            return
        self._last_target = command.position_target
        tracking_kwargs = {
            "position_target": command.position_target,
            "velocity_target": command.velocity_target,
            "add_gravity_compensation": True,
        }
        if self._pair.kp is not None:
            tracking_kwargs["kp"] = self._pair.kp
        if self._pair.kd is not None:
            tracking_kwargs["kd"] = self._pair.kd
        self._pair.follower.step_target_tracking(**tracking_kwargs)
        self._runtime.update_pair_mode(self._pair.name, command.mode)
        self._runtime.record_pair_action(
            self._pair.name,
            time.monotonic(),
            command.position_target,
        )


class AsyncCollectionRuntime:
    """Collection runtime with a shared scheduler and camera bridges."""

    def __init__(
        self,
        cameras: dict[str, ImageSensor],
        camera_configs: dict[str, CameraConfig],
        robots: dict[str, RobotArm],
        teleop_pairs: list[TeleopPairBinding],
        fps: int,
        export_root: str | Path,
        project_name: str,
        video_codec: str,
        depth_codec: str,
        lerobot_version: str = "v2.1",
        export_queue_size: int = 4,
        max_pending_episodes: int = 8,
        telemetry_hz: int = DEFAULT_CONTROL_HZ,
        control_hz: int = DEFAULT_CONTROL_HZ,
        export_delay_sec: float = 0.0,
        scheduler_driver: str = "asyncio",
        camera_queue_size: int = 4,
        preview_live_feedback: bool = False,
    ) -> None:
        self._cameras = cameras
        self._camera_configs = camera_configs
        self._robots = robots
        self._teleop_pairs = teleop_pairs
        self._fps = fps
        self._video_codec = video_codec
        self._depth_codec = depth_codec
        self._telemetry_hz = telemetry_hz
        self._control_hz = control_hz
        self._scheduler_driver_name = scheduler_driver
        self._camera_queue_size = max(1, camera_queue_size)
        self._preview_live_feedback = preview_live_feedback
        self._teleop_leader_names = {pair.leader_name for pair in teleop_pairs}
        self._state_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._current_episode: EpisodeAccumulator | None = None
        self._pending_episode: RecordedEpisode | None = None
        self._episode_index = 0
        self._action_layout = _build_action_layout(teleop_pairs)
        self._latest_frames: dict[str, np.ndarray | None] = {
            name: None for name in cameras
        }
        self._latest_robot_states: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in robots
        }
        self._latest_pair_modes: dict[str, str] = {}
        self._control_run_history = _TimingHistory()
        self._telemetry_run_history = _TimingHistory()
        self._valid_robot_sample_histories = {
            name: _TimingHistory() for name in robots
        }
        self._preview_keepalive_names: set[str] = set()
        self._frame_sources: dict[str, ThreadedCameraFrameSource] = {}
        self._scheduler = None
        self._opened = False
        self._exporter = AsyncEpisodeExporter(
            root=export_root,
            project_name=project_name,
            fps=fps,
            camera_configs=camera_configs,
            video_codec=video_codec,
            depth_codec=depth_codec,
            queue_size=export_queue_size,
            pending_queue_size=max_pending_episodes,
            export_delay_sec=export_delay_sec,
            lerobot_version=lerobot_version,
        )

    @classmethod
    def from_config(
        cls,
        cfg: RollioConfig,
        export_delay_sec: float = 0.0,
        *,
        scheduler_driver: str = "asyncio",
        preview_live_feedback: bool = False,
    ) -> "AsyncCollectionRuntime":
        cameras = build_cameras_from_config(cfg)
        robots = build_robots_from_config(cfg)
        teleop_pairs = build_teleop_pairs_from_config(cfg, robots)
        camera_queue_size = max(4, cfg.fps)
        if not cfg.async_pipeline.allow_drop_preview_frames:
            camera_queue_size = max(camera_queue_size, cfg.fps * 2)
        return cls(
            cameras=cameras,
            camera_configs={camera_cfg.name: camera_cfg for camera_cfg in cfg.cameras},
            robots=robots,
            teleop_pairs=teleop_pairs,
            fps=cfg.fps,
            export_root=cfg.storage.root,
            project_name=cfg.project_name,
            video_codec=cfg.encoder.video_codec,
            depth_codec=cfg.encoder.depth_codec,
            lerobot_version=cfg.storage.lerobot_version,
            export_queue_size=cfg.async_pipeline.export_queue_size,
            max_pending_episodes=cfg.async_pipeline.max_pending_episodes,
            telemetry_hz=cfg.async_pipeline.telemetry_hz,
            control_hz=cfg.async_pipeline.control_hz,
            export_delay_sec=export_delay_sec,
            scheduler_driver=scheduler_driver,
            camera_queue_size=camera_queue_size,
            preview_live_feedback=preview_live_feedback,
        )

    @property
    def recording(self) -> bool:
        with self._state_lock:
            return self._current_episode is not None

    @property
    def elapsed(self) -> float:
        with self._state_lock:
            episode = self._current_episode
            if episode is None:
                return 0.0
            started_at = episode._started_at  # noqa: SLF001
        return max(0.0, time.monotonic() - started_at)

    @property
    def video_codec(self) -> str:
        return self._video_codec

    @property
    def depth_codec(self) -> str:
        return self._depth_codec

    @property
    def scheduler_driver(self) -> str:
        return self._scheduler_driver_name

    @property
    def telemetry_hz(self) -> int:
        return self._telemetry_hz

    @property
    def control_hz(self) -> int:
        return self._control_hz

    def open(self) -> None:
        if self._opened:
            return

        self._reset_timing_histories()
        for robot in self._robots.values():
            robot.open()
            robot.enable()
        self._preview_keepalive_names.clear()
        paired_robot_names = {pair.leader_name for pair in self._teleop_pairs} | {
            pair.follower_name for pair in self._teleop_pairs
        }
        for pair in self._teleop_pairs:
            pair.leader.enter_free_drive()
            pair.follower.enter_target_tracking()
        if self._preview_live_feedback:
            for name, robot in self._robots.items():
                if name in paired_robot_names:
                    continue
                preview_mode = _preview_control_mode(robot)
                if preview_mode == ControlMode.FREE_DRIVE:
                    entered = robot.enter_free_drive()
                elif preview_mode == ControlMode.TARGET_TRACKING:
                    entered = robot.enter_target_tracking()
                else:
                    entered = False
                if entered and _requires_preview_keepalive(robot):
                    self._preview_keepalive_names.add(name)

        self._frame_sources = {
            name: ThreadedCameraFrameSource(
                name,
                camera,
                max_pending_frames=self._camera_queue_size,
            )
            for name, camera in self._cameras.items()
        }
        for frame_source in self._frame_sources.values():
            frame_source.open()

        self._exporter.start()
        tasks = self._build_scheduled_tasks()
        self._scheduler = build_scheduler_driver(self._scheduler_driver_name, tasks)
        self._scheduler.start()
        self._opened = True

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None

        for frame_source in self._frame_sources.values():
            frame_source.close()
        self._frame_sources = {}

        self._exporter.shutdown()

        for robot in self._robots.values():
            robot.disable()
            robot.close()

        self._preview_keepalive_names.clear()
        self._opened = False

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        """Request all runtime robots to return to zero when supported."""
        results: dict[str, bool] = {}
        for name, robot in self._robots.items():
            try:
                results[name] = bool(robot.move_to_zero(timeout=timeout))
            except (OSError, RuntimeError, ValueError, TypeError):
                results[name] = False
        return results

    def _build_scheduled_tasks(self) -> list[ScheduledTask]:
        tasks: list[ScheduledTask] = []
        tasks.extend(
            TeleopTask(pair, self._control_hz, self).scheduled_task()
            for pair in self._teleop_pairs
        )
        tasks.extend(
            RobotTelemetryTask(
                name,
                robot,
                self._telemetry_hz,
                self,
                preview_keepalive=name in self._preview_keepalive_names,
            ).scheduled_task()
            for name, robot in self._robots.items()
        )
        tasks.extend(
            CameraIngestTask(name, frame_source, self).scheduled_task()
            for name, frame_source in self._frame_sources.items()
        )
        return tasks

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
                action_layout=self._action_layout,
            )
            latest_frames = {
                name: frame.copy() if frame is not None else None
                for name, frame in self._latest_frames.items()
            }
            latest_robot_states = {
                name: {key: np.asarray(value).copy() for key, value in state.items()}
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

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool:
        return self._exporter.wait_for_episode(episode_index, timeout)

    def export_records(self) -> dict[int, ExportRecord]:
        return self._exporter.records()

    def export_status(self) -> tuple[int, int]:
        records = self._exporter.records().values()
        pending = sum(1 for record in records if not record.done_event.is_set())
        completed = sum(
            1
            for record in records
            if record.done_event.is_set() and record.error is None
        )
        return pending, completed

    def latest_frames(self) -> dict[str, np.ndarray | None]:
        with self._state_lock:
            return dict(self._latest_frames)

    def latest_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        with self._state_lock:
            return {
                name: dict(state) for name, state in self._latest_robot_states.items()
            }

    def latest_pair_modes(self) -> dict[str, str]:
        with self._state_lock:
            return dict(self._latest_pair_modes)

    def action_layout(self) -> list[dict[str, int | str]]:
        with self._state_lock:
            return [dict(entry) for entry in self._action_layout]

    def scheduler_metrics(
        self,
    ) -> dict[str, DriverMetrics | dict[str, FrameSourceMetrics]]:
        driver_metrics = (
            self._scheduler.metrics()
            if self._scheduler is not None
            else DriverMetrics(driver_name=self._scheduler_driver_name, task_metrics={})
        )
        camera_metrics = {
            name: frame_source.metrics()
            for name, frame_source in self._frame_sources.items()
        }
        return {
            "driver": driver_metrics,
            "cameras": camera_metrics,
        }

    def timing_diagnostics(self) -> RuntimeTimingDiagnostics:
        driver_metrics = (
            self._scheduler.metrics()
            if self._scheduler is not None
            else DriverMetrics(driver_name=self._scheduler_driver_name, task_metrics={})
        )
        scheduler_intervals = tuple(
            getattr(driver_metrics, "recent_loop_intervals_ms", ())
        )
        target_intervals_ms = tuple(
            value
            for value in (
                _target_interval_ms(self._control_hz),
                _target_interval_ms(self._telemetry_hz),
            )
            if value is not None
        )
        now = time.monotonic()
        with self._timing_lock:
            control_runs = self._control_run_history.snapshot(
                target_interval_ms=_target_interval_ms(self._control_hz),
                now=now,
            )
            telemetry_runs = self._telemetry_run_history.snapshot(
                target_interval_ms=_target_interval_ms(self._telemetry_hz),
                now=now,
            )
            valid_robot_samples = {
                name: history.snapshot(
                    target_interval_ms=self._robot_target_interval_ms(name),
                    now=now,
                )
                for name, history in self._valid_robot_sample_histories.items()
            }
        return RuntimeTimingDiagnostics(
            scheduler_loop=TimingTrace(
                intervals_ms=scheduler_intervals,
                target_interval_ms=min(target_intervals_ms) if target_intervals_ms else None,
                last_gap_ms=scheduler_intervals[-1] if scheduler_intervals else None,
                max_gap_ms=max(scheduler_intervals) if scheduler_intervals else None,
                age_ms=getattr(driver_metrics, "last_loop_age_ms", None),
            ),
            telemetry_runs=telemetry_runs,
            control_runs=control_runs,
            valid_robot_samples=valid_robot_samples,
        )

    def observe_control_run(self, timestamp: float | None = None) -> None:
        with self._timing_lock:
            self._control_run_history.observe(timestamp)

    def observe_telemetry_run(self, timestamp: float | None = None) -> None:
        with self._timing_lock:
            self._telemetry_run_history.observe(timestamp)

    def observe_valid_robot_sample(
        self,
        robot_name: str,
        timestamp: float | None = None,
    ) -> None:
        with self._timing_lock:
            history = self._valid_robot_sample_histories.get(robot_name)
            if history is None:
                history = _TimingHistory()
                self._valid_robot_sample_histories[robot_name] = history
            history.observe(timestamp)

    def update_latest_frame(self, name: str, frame: np.ndarray) -> None:
        with self._state_lock:
            self._latest_frames[name] = frame

    def update_latest_robot_state(
        self, name: str, state: dict[str, np.ndarray]
    ) -> None:
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

    def record_robot_state(
        self, name: str, ts: float, state: dict[str, np.ndarray]
    ) -> None:
        with self._state_lock:
            episode = self._current_episode
        if episode is not None:
            episode.append_robot(name, ts, state)

    def record_pair_action(self, pair_name: str, ts: float, target: np.ndarray) -> None:
        with self._state_lock:
            episode = self._current_episode
        if episode is not None:
            episode.append_pair_action(pair_name, ts, target)

    def _reset_timing_histories(self) -> None:
        with self._timing_lock:
            self._control_run_history.reset()
            self._telemetry_run_history.reset()
            for history in self._valid_robot_sample_histories.values():
                history.reset()

    def _robot_target_interval_ms(self, robot_name: str) -> float | None:
        if self._preview_live_feedback and robot_name in self._teleop_leader_names:
            return _target_interval_ms(self._control_hz)
        return _target_interval_ms(self._telemetry_hz)


def build_teleop_pairs_from_config(
    cfg: RollioConfig,
    robots: dict[str, RobotArm],
) -> list[TeleopPairBinding]:
    """Resolve tele-operation pairs from configuration."""

    pair_cfgs: list[TeleopPairConfig]
    if cfg.teleop_pairs:
        pair_cfgs = cfg.teleop_pairs
    else:
        pair_cfgs = suggest_teleop_pairs(cfg.robots)

    pairs: list[TeleopPairBinding] = []
    for pair_cfg in pair_cfgs:
        leader = robots[pair_cfg.leader]
        follower = robots[pair_cfg.follower]
        if leader is follower:
            raise ValueError(
                f"Tele-op pair '{pair_cfg.name}' resolves leader and follower "
                "to the same runtime robot instance"
            )
        pairs.append(
            TeleopPairBinding(
                name=pair_cfg.name,
                leader_name=pair_cfg.leader,
                follower_name=pair_cfg.follower,
                leader=leader,
                follower=follower,
                mapper_mode=pair_cfg.mapper,
            )
        )
    return pairs
