"""Episode recorder — orchestrates multi-sensor recording with timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rollio.sensors.base import ImageSensor, RobotSensor
from rollio.utils.time import EpisodeClock


@dataclass
class EpisodeData:
    """Collected data for one completed episode."""

    episode_index: int
    fps: int
    duration: float  # seconds

    # Per-camera: list of (relative_timestamp, bgr_frame)
    camera_frames: dict[str, list[tuple[float, np.ndarray]]] = field(
        default_factory=dict
    )

    # Per-robot: list of (relative_timestamp, state_dict)
    robot_states: dict[str, list[tuple[float, dict[str, np.ndarray]]]] = field(
        default_factory=dict
    )

    # Per-teleop-pair: list of (relative_timestamp, target_vector)
    pair_actions: dict[str, list[tuple[float, np.ndarray]]] = field(
        default_factory=dict
    )

    # Ordered action slices for the flattened action vector.
    action_layout: list[dict[str, int | str]] = field(default_factory=list)


class EpisodeRecorder:
    """Records one episode at a time from multiple sensors.

    Sensors are polled synchronously at the target FPS in the main thread.
    The caller drives the lifecycle: ``start()`` → poll in a loop via
    ``tick()`` → ``stop()``.
    """

    def __init__(
        self,
        cameras: dict[str, ImageSensor],
        robots: dict[str, RobotSensor],
        fps: int = 30,
    ) -> None:
        self._cameras = cameras
        self._robots = robots
        self._fps = fps
        self._clock = EpisodeClock()
        self._episode_idx = 0

        # Accumulation buffers  (filled during recording)
        self._cam_buf: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._rob_buf: dict[str, list[tuple[float, dict[str, np.ndarray]]]] = {}
        self._recording = False

    # ── public API ─────────────────────────────────────────────────

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        return self._clock.elapsed()

    @property
    def episode_index(self) -> int:
        return self._episode_idx

    def start(self) -> None:
        """Begin recording a new episode."""
        self._cam_buf = {name: [] for name in self._cameras}
        self._rob_buf = {name: [] for name in self._robots}
        self._clock.start()
        self._recording = True

    def tick(self) -> dict[str, np.ndarray | None]:
        """Sample all sensors once.  Returns latest camera frames (for TUI)."""
        if not self._recording:
            return {}

        t_base = self._clock.start_time
        latest_frames: dict[str, np.ndarray | None] = {}

        for name, cam in self._cameras.items():
            ts, frame = cam.read()
            rel_ts = ts - t_base
            self._cam_buf[name].append((rel_ts, frame))
            latest_frames[name] = frame

        for name, rob in self._robots.items():
            ts, state = rob.read()
            rel_ts = ts - t_base
            self._rob_buf[name].append((rel_ts, state))

        return latest_frames

    def stop(self) -> EpisodeData:
        """Stop recording and return the collected episode data."""
        duration = self._clock.stop()
        self._recording = False

        data = EpisodeData(
            episode_index=self._episode_idx,
            fps=self._fps,
            duration=duration,
            camera_frames=dict(self._cam_buf),
            robot_states=dict(self._rob_buf),
        )
        self._episode_idx += 1
        return data

    def peek_sensors(
        self,
    ) -> tuple[dict[str, np.ndarray | None], dict[str, dict[str, np.ndarray] | None]]:
        """Read sensors once without recording (for live preview)."""
        frames: dict[str, np.ndarray | None] = {}
        states: dict[str, dict[str, np.ndarray] | None] = {}
        for name, cam in self._cameras.items():
            _, frame = cam.read()
            frames[name] = frame
        for name, rob in self._robots.items():
            _, state = rob.read()
            states[name] = state
        return frames, states
