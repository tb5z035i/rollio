"""Tests for the replay TUI helpers."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np

from rollio.config.schema import CameraConfig, RollioConfig
from rollio.tui import replay


class _FakeTerm:
    cols = 90
    rows = 28

    def __init__(self, keys: list[str]) -> None:
        self._keys = iter(keys)

    def __enter__(self) -> "_FakeTerm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def key(self) -> str | None:
        return next(self._keys, None)


class _FakeStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


class _FakeReplayRuntime:
    def __init__(self) -> None:
        self.episode = SimpleNamespace(
            episode_relative_path="data/chunk-000/episode_000003.parquet",
            episode_index=3,
            duration=0.4,
            action_layout=[{"follower": "follower_arm"}],
        )
        self.fps = 10
        self.playing = False
        self.paused = False
        self.returning = False
        self.completed = False
        self.elapsed = 0.0
        self.open_called = False
        self.close_called = False
        self.return_zero_calls: list[float] = []
        self.start_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.update_calls = 0
        self._frames = {"cam_main": np.zeros((8, 8, 3), dtype=np.uint8)}
        self._recorded = {
            "follower_arm": {"position": np.array([0.1, 0.2], dtype=np.float32)},
        }
        self._live = {
            "follower_arm": {"position": np.array([0.12, 0.18], dtype=np.float32)},
        }

    @property
    def state(self) -> str:
        if self.returning:
            return "RETURNING"
        if self.playing and self.paused:
            return "PAUSED"
        if self.playing:
            return "PLAYING"
        if self.completed:
            return "DONE"
        return "IDLE"

    def open(self) -> None:
        self.open_called = True

    def close(self) -> None:
        self.close_called = True

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        self.return_zero_calls.append(timeout)
        return {}

    def start_playback(self) -> None:
        self.start_calls += 1
        self.playing = True
        self.paused = False
        self.completed = False

    def pause_playback(self) -> None:
        self.pause_calls += 1
        self.paused = True

    def resume_playback(self) -> None:
        self.resume_calls += 1
        self.paused = False

    def update(self) -> None:
        self.update_calls += 1
        if self.playing and not self.paused:
            self.elapsed += 0.1

    def latest_frames(self) -> dict[str, np.ndarray]:
        return self._frames

    def latest_recorded_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        return self._recorded

    def latest_live_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        return self._live


def _run_replay_once(
    monkeypatch,
    runtime: _FakeReplayRuntime,
    keys: list[str],
    cfg: RollioConfig,
) -> str:
    fake_stdout = _FakeStdout()
    monkeypatch.setattr(
        replay.ReplayRuntime,
        "from_config",
        staticmethod(lambda _cfg, _episode_path: runtime),
    )
    monkeypatch.setattr(replay, "_Term", lambda: _FakeTerm(keys))
    monkeypatch.setattr(replay.sys, "stdout", fake_stdout)
    monkeypatch.setattr(replay.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay, "calc_render_size", lambda *args, **kwargs: (4, 1))
    monkeypatch.setattr(replay, "render_frame", lambda *args, **kwargs: b"frame")
    monkeypatch.setattr(replay, "render_depth", lambda *args, **kwargs: b"depth")
    monkeypatch.setattr(replay, "blit_frame", lambda *args, **kwargs: b"")
    replay.run_replay(cfg, "episode_000003.parquet")
    return fake_stdout.buffer.getvalue().decode("utf-8", errors="ignore")


def test_run_replay_uses_start_key_for_start_pause_resume(monkeypatch) -> None:
    runtime = _FakeReplayRuntime()
    cfg = RollioConfig(
        fps=10,
        cameras=[CameraConfig(name="cam_main", type="pseudo")],
        robots=[],
    )

    _run_replay_once(monkeypatch, runtime, [" ", " ", " ", "q"], cfg)

    assert runtime.open_called is True
    assert runtime.start_calls == 1
    assert runtime.pause_calls == 1
    assert runtime.resume_calls == 1
    assert runtime.close_called is True
    assert runtime.return_zero_calls == [5.0]


def test_replay_help_panel_mentions_toggle_and_rec_live() -> None:
    runtime = _FakeReplayRuntime()
    cfg = RollioConfig(
        fps=10,
        cameras=[],
        robots=[],
    )

    lines = replay._help_panel_lines(cfg, runtime, panel_w=36, panel_h=18)
    joined = "\n".join(lines)

    assert "start/pause/continue" in joined
    assert "all robots opened" in joined
    assert "rec + live state shown" in joined


def test_replay_robot_panel_shows_recorded_and_live(monkeypatch) -> None:
    del monkeypatch
    lines = replay._dual_robot_panel_lines(
        {"arm_a": {"position": np.array([0.1], dtype=np.float32)}},
        {"arm_a": {"position": np.array([0.2], dtype=np.float32)}},
        {"arm_a": "pseudo"},
        panel_w=36,
        panel_h=12,
    )

    joined = "\n".join(lines)
    assert "REC" in joined
    assert "LIVE" in joined
