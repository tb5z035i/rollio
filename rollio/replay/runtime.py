"""Runtime support for deterministic episode replay."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rollio.collect.devices import build_robots_from_config
from rollio.config.schema import RollioConfig
from rollio.robot import JointState, RobotArm
from rollio.replay.dataset import ReplayEpisode, load_replay_episode


def _joint_state_to_robot_state(
    robot: RobotArm,
    joint_state: JointState,
) -> dict[str, np.ndarray]:
    robot_state: dict[str, np.ndarray] = {}
    if joint_state.position is not None:
        robot_state["position"] = np.asarray(joint_state.position, dtype=np.float32)
        try:
            pose = robot.kinematics.forward_kinematics(joint_state.position)
        except Exception:
            pose = None
        if pose is not None:
            robot_state["ee_position"] = pose.position.astype(np.float32)
            robot_state["ee_quaternion"] = pose.quaternion.astype(np.float32)
    if joint_state.velocity is not None:
        robot_state["velocity"] = np.asarray(joint_state.velocity, dtype=np.float32)
    if joint_state.effort is not None:
        robot_state["effort"] = np.asarray(joint_state.effort, dtype=np.float32)
    control_loop_metrics = getattr(robot, "control_loop_metrics", None)
    if callable(control_loop_metrics):
        try:
            metrics = control_loop_metrics()
        except Exception:
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


class _VideoReplayReader:
    """Lightweight random/sequential frame reader for one episode video."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture: cv2.VideoCapture | None = None
        self._last_index = -1
        self._last_frame: np.ndarray | None = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def open(self) -> None:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open replay video: {self._path}")
        self._capture = capture
        self._last_index = -1
        self._last_frame = None
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._frame_count = max(frame_count, 0)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._last_index = -1
        self._last_frame = None
        self._frame_count = 0

    def reset(self) -> None:
        if self._capture is None:
            raise RuntimeError("Replay video reader is not open")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._last_index = -1
        self._last_frame = None

    def frame_at(self, index: int) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Replay video reader is not open")
        if self._last_frame is not None and index == self._last_index:
            return self._last_frame

        target_index = max(0, index)
        if self._frame_count > 0:
            target_index = min(target_index, self._frame_count - 1)

        if target_index != self._last_index + 1:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)

        ok, frame = self._capture.read()
        if not ok or frame is None:
            if self._last_frame is not None:
                return self._last_frame
            raise RuntimeError(
                f"Failed to decode replay frame {target_index} from {self._path}"
            )

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        self._last_index = target_index
        self._last_frame = frame
        return frame


@dataclass(frozen=True)
class ReplayControlTarget:
    """Target slice for one follower robot."""

    follower: str
    target: np.ndarray


class ReplayRuntime:
    """Owns replay playback, live robot reads, and follower commands."""

    def __init__(
        self,
        cfg: RollioConfig,
        episode: ReplayEpisode,
        *,
        return_duration_sec: float = 2.0,
    ) -> None:
        self._cfg = cfg
        self._episode = episode
        self._robots = build_robots_from_config(cfg)
        self._video_readers = {
            name: _VideoReplayReader(stream.path)
            for name, stream in episode.camera_streams.items()
        }
        self._controlled_followers = {
            str(entry["follower"]): self._robots[str(entry["follower"])]
            for entry in episode.action_layout
        }
        self._latest_frames: dict[str, np.ndarray | None] = {
            name: None for name in self._video_readers
        }
        self._latest_recorded_robot_states: dict[str, dict[str, np.ndarray]] = {
            robot.name: {} for robot in cfg.robots
        }
        self._latest_live_robot_states: dict[str, dict[str, np.ndarray]] = {
            robot.name: {} for robot in cfg.robots
        }
        self._opened = False
        self._playing = False
        self._paused = False
        self._returning = False
        self._completed = False
        self._start_monotonic = 0.0
        self._pause_started_at = 0.0
        self._paused_total_sec = 0.0
        self._current_index = 0
        self._return_started_at = 0.0
        self._return_duration_sec = max(0.5, float(return_duration_sec))
        self._last_error: str | None = None

    @classmethod
    def from_config(
        cls,
        cfg: RollioConfig,
        episode_path: str | Path,
        *,
        return_duration_sec: float = 2.0,
    ) -> "ReplayRuntime":
        episode = load_replay_episode(cfg, episode_path)
        return cls(cfg, episode, return_duration_sec=return_duration_sec)

    @property
    def episode(self) -> ReplayEpisode:
        return self._episode

    @property
    def fps(self) -> int:
        return self._episode.fps

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def returning(self) -> bool:
        return self._returning

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def elapsed(self) -> float:
        if not self._playing:
            return float(self._episode.timestamps[self._current_index])
        if self._paused:
            base = self._pause_started_at - self._start_monotonic - self._paused_total_sec
        else:
            base = time.monotonic() - self._start_monotonic - self._paused_total_sec
        return max(0.0, min(base, self._episode.duration))

    @property
    def state(self) -> str:
        if self._returning:
            return "RETURNING"
        if self._playing and self._paused:
            return "PAUSED"
        if self._playing:
            return "PLAYING"
        if self._completed:
            return "DONE"
        return "IDLE"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def open(self) -> None:
        if self._opened:
            return
        opened_names: list[str] = []
        try:
            for name, robot in self._robots.items():
                robot.open()
                opened_names.append(name)
            for name, robot in self._controlled_followers.items():
                if not robot.enable():
                    raise RuntimeError(f"Failed to enable replay follower robot: {name}")
                if not robot.enter_target_tracking():
                    raise RuntimeError(
                        f"Failed to enter target-tracking mode for replay follower: {name}"
                    )
            for reader in self._video_readers.values():
                reader.open()
            self._set_display_index(0)
            self._refresh_live_states()
            self._opened = True
        except Exception:
            for reader in self._video_readers.values():
                reader.close()
            for name in reversed(opened_names):
                robot = self._robots[name]
                try:
                    robot.disable()
                except Exception:
                    pass
                try:
                    robot.close()
                except Exception:
                    pass
            raise

    def close(self) -> None:
        for reader in self._video_readers.values():
            reader.close()
        for robot in self._robots.values():
            try:
                robot.disable()
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass
        self._opened = False

    def start_playback(self) -> None:
        if not self._opened:
            raise RuntimeError("Replay runtime is not open")
        for reader in self._video_readers.values():
            reader.reset()
        self._playing = True
        self._paused = False
        self._returning = False
        self._completed = False
        self._paused_total_sec = 0.0
        self._start_monotonic = time.monotonic()
        self._pause_started_at = 0.0
        self._current_index = 0
        self._last_error = None
        self._set_display_index(0)
        self._apply_index_targets(0)
        self._refresh_live_states()

    def pause_playback(self) -> None:
        if self._playing and not self._paused:
            self._paused = True
            self._pause_started_at = time.monotonic()

    def resume_playback(self) -> None:
        if self._playing and self._paused:
            self._paused_total_sec += time.monotonic() - self._pause_started_at
            self._paused = False
            self._pause_started_at = 0.0

    def stop_playback(self) -> None:
        self._playing = False
        self._paused = False
        self._returning = False
        self._paused_total_sec = 0.0
        self._pause_started_at = 0.0
        self._completed = False
        self._current_index = 0
        self._set_display_index(0)
        self._refresh_live_states()

    def update(self) -> None:
        if not self._opened:
            raise RuntimeError("Replay runtime is not open")
        try:
            if self._returning:
                self._step_return_to_start()
            elif self._playing:
                self._step_playback()
            self._refresh_live_states()
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def latest_frames(self) -> dict[str, np.ndarray | None]:
        return dict(self._latest_frames)

    def latest_recorded_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                key: np.asarray(value).copy()
                for key, value in state.items()
            }
            for name, state in self._latest_recorded_robot_states.items()
        }

    def latest_live_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                key: np.asarray(value).copy()
                for key, value in state.items()
            }
            for name, state in self._latest_live_robot_states.items()
        }

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, robot in self._robots.items():
            try:
                results[name] = bool(robot.move_to_zero(timeout=timeout))
            except Exception:
                results[name] = False
        return results

    def _step_playback(self) -> None:
        if self._paused:
            return
        elapsed = self.elapsed
        index = min(int(elapsed * self._episode.fps), self._episode.row_count - 1)
        self._current_index = index
        self._set_display_index(index)
        self._apply_index_targets(index)
        if elapsed >= self._episode.duration:
            self._playing = False
            self._paused = False
            self._begin_return_to_start()

    def _begin_return_to_start(self) -> None:
        self._returning = True
        self._completed = False
        self._return_started_at = time.monotonic()
        self._set_display_index(0)

    def _step_return_to_start(self) -> None:
        for entry in self._episode.action_layout:
            follower_name = str(entry["follower"])
            follower = self._controlled_followers[follower_name]
            target = self._episode.action_slice(entry, 0)
            follower.step_target_tracking(
                position_target=np.asarray(target, dtype=np.float64),
                velocity_target=np.zeros_like(target, dtype=np.float64),
                add_gravity_compensation=True,
            )
        self._set_display_index(0)
        if time.monotonic() - self._return_started_at >= self._return_duration_sec:
            self._returning = False
            self._completed = True

    def _set_display_index(self, index: int) -> None:
        safe_index = max(0, min(index, self._episode.row_count - 1))
        self._current_index = safe_index
        for name, reader in self._video_readers.items():
            self._latest_frames[name] = reader.frame_at(safe_index)
        for robot in self._cfg.robots:
            self._latest_recorded_robot_states[robot.name] = self._episode.state_at_index(
                robot.name,
                safe_index,
            )

    def _apply_index_targets(self, index: int) -> None:
        for entry in self._episode.action_layout:
            follower_name = str(entry["follower"])
            follower = self._controlled_followers[follower_name]
            target = self._episode.action_slice(entry, index)
            follower.step_target_tracking(
                position_target=np.asarray(target, dtype=np.float64),
                velocity_target=np.zeros_like(target, dtype=np.float64),
                add_gravity_compensation=True,
            )

    def _refresh_live_states(self) -> None:
        for name, robot in self._robots.items():
            try:
                joint_state = robot.read_joint_state()
                self._latest_live_robot_states[name] = _joint_state_to_robot_state(
                    robot,
                    joint_state,
                )
            except Exception:
                self._latest_live_robot_states[name] = {}


__all__ = ["ReplayControlTarget", "ReplayRuntime"]
