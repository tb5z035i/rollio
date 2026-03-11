"""Tests for non-interactive collection TUI helpers."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np

from rollio.config.schema import CameraConfig, ControlConfig, RollioConfig
from rollio.defaults import DEFAULT_CONTROL_INTERVAL_MS
from rollio.tui import app


class _FakeTerm:
    cols = 80
    rows = 24

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


class _FakeRuntime:
    def __init__(
        self,
        *,
        frames: dict[str, np.ndarray] | None = None,
        robot_states: dict[str, dict[str, np.ndarray]] | None = None,
        pending_episode: object | None = None,
    ) -> None:
        self.recording = False
        self.elapsed = 0.0
        self.video_codec = "mpeg4"
        self.depth_codec = "rawvideo"
        self._frames = frames or {}
        self._robot_states = robot_states or {}
        self._pending_episode = pending_episode or SimpleNamespace(
            episode_index=0,
            duration=0.2,
        )
        self.keep_calls: list[object] = []
        self.discard_calls: list[object] = []
        self.return_zero_calls: list[float] = []
        self.close_called = False
        self.call_order: list[str] = []

    def open(self) -> None:
        self.call_order.append("open")
        return None

    def close(self) -> None:
        self.close_called = True
        self.call_order.append("close")

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        self.return_zero_calls.append(timeout)
        self.call_order.append("return_zero")
        return {}

    def export_status(self) -> tuple[int, int]:
        return 0, 0

    def latest_frames(self) -> dict[str, np.ndarray]:
        return self._frames

    def latest_robot_states(self) -> dict[str, dict[str, np.ndarray]]:
        return self._robot_states

    def start_episode(self) -> None:
        self.recording = True

    def stop_episode(self) -> object:
        self.recording = False
        return self._pending_episode

    def keep_episode(self, episode: object) -> object:
        self.keep_calls.append(episode)
        return SimpleNamespace(done_event=SimpleNamespace(is_set=lambda: False))

    def discard_episode(self, episode: object) -> None:
        self.discard_calls.append(episode)


def _run_collection_once(
    monkeypatch,
    fake_runtime: _FakeRuntime,
    keys: list[str],
    cfg: RollioConfig,
) -> str:
    fake_stdout = _FakeStdout()
    monkeypatch.setattr(
        app.AsyncCollectionRuntime,
        "from_config",
        staticmethod(lambda _cfg: fake_runtime),
    )
    monkeypatch.setattr(app, "_Term", lambda: _FakeTerm(keys))
    monkeypatch.setattr(app.sys, "stdout", fake_stdout)
    monkeypatch.setattr(app.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "calc_render_size", lambda *args, **kwargs: (4, 1))
    monkeypatch.setattr(app, "render_frame", lambda *args, **kwargs: b"frame")
    monkeypatch.setattr(app, "blit_frame", lambda *args, **kwargs: b"")
    app.run_collection(cfg)
    assert fake_runtime.close_called is True
    return fake_stdout.buffer.getvalue().decode("utf-8", errors="ignore")


def test_status_lines_include_queue_and_codecs() -> None:
    runtime = SimpleNamespace(
        recording=False,
        elapsed=0.0,
        video_codec="libx264",
        depth_codec="ffv1",
    )

    line1, line2 = app._status_lines(
        runtime,
        pending_episode=None,
        episodes_kept=4,
        pending_exports=2,
        completed_exports=7,
        actual_fps=29.7,
        render_mode="16",
    )

    assert "State: IDLE" in line1
    assert "Export queue: 2 pending / 7 done" in line1
    assert "FPS: 29.7" in line1
    assert "Render: 16-clr" in line2
    assert "RGB codec: libx264" in line2
    assert "Depth codec: ffv1" in line2


def test_help_panel_shows_key_help_and_codecs() -> None:
    cfg = SimpleNamespace(
        mode="teleop",
        controls=SimpleNamespace(start_stop=" ", keep="k", discard="d"),
    )
    runtime = SimpleNamespace(video_codec="mpeg4", depth_codec="rawvideo")

    lines = app._help_panel_lines(cfg, runtime, panel_w=32, panel_h=16)
    joined = "\n".join(lines)

    assert "RGB codec:" in joined
    assert "mpeg4" in joined
    assert "Depth codec:" in joined
    assert "rawvideo" in joined
    assert "SPACE" in joined
    assert "start / stop" in joined
    assert "ENTER/k" in joined
    assert "BACKSPACE/d" in joined


def test_robot_panel_uses_eef_mm_range() -> None:
    lines = app._robot_panel_lines(
        {
            "leader_e2b": {
                "position": np.array([0.035], dtype=np.float32),
            },
        },
        {"leader_e2b": "airbot_e2b"},
        panel_w=32,
        panel_h=8,
    )

    assert "35.0mm" in "\n".join(lines)


def test_robot_panel_shows_control_interval_bar() -> None:
    lines = app._robot_panel_lines(
        {
            "leader_arm": {
                "position": np.array([0.1], dtype=np.float32),
                "control_loop_interval_ms": np.array([4.2], dtype=np.float32),
                "control_loop_target_interval_ms": np.array(
                    [DEFAULT_CONTROL_INTERVAL_MS],
                    dtype=np.float32,
                ),
            },
        },
        {"leader_arm": "airbot_play"},
        panel_w=36,
        panel_h=10,
    )

    joined = "\n".join(lines)
    assert "ctrl" in joined
    assert "4.2ms" in joined


def test_run_collection_renders_camera_name_below_preview(monkeypatch) -> None:
    fake_runtime = _FakeRuntime(
        frames={"cam_main": np.zeros((8, 8, 3), dtype=np.uint8)},
    )
    cfg = RollioConfig(
        fps=10,
        mode="teleop",
        cameras=[CameraConfig(name="cam_main", type="pseudo")],
        robots=[],
    )

    rendered = _run_collection_once(monkeypatch, fake_runtime, ["x", "q"], cfg)

    assert "[1] cam_main" in rendered


def test_run_collection_uses_enter_to_keep_reviewed(monkeypatch) -> None:
    pending_episode = SimpleNamespace(episode_index=3, duration=0.4)
    fake_runtime = _FakeRuntime(pending_episode=pending_episode)
    cfg = RollioConfig(
        fps=10,
        mode="teleop",
        cameras=[],
        robots=[],
        controls=ControlConfig(start_stop=" ", keep="k", discard="d"),
    )

    _run_collection_once(monkeypatch, fake_runtime, [" ", " ", "\n", "q"], cfg)

    assert fake_runtime.keep_calls == [pending_episode]
    assert fake_runtime.discard_calls == []


def test_run_collection_uses_backspace_to_discard_reviewed(monkeypatch) -> None:
    pending_episode = SimpleNamespace(episode_index=5, duration=0.6)
    fake_runtime = _FakeRuntime(pending_episode=pending_episode)
    cfg = RollioConfig(
        fps=10,
        mode="teleop",
        cameras=[],
        robots=[],
        controls=ControlConfig(start_stop=" ", keep="k", discard="d"),
    )

    _run_collection_once(monkeypatch, fake_runtime, [" ", " ", "\x7f", "q"], cfg)

    assert fake_runtime.keep_calls == []
    assert fake_runtime.discard_calls == [pending_episode]


def test_run_collection_returns_robots_to_zero_before_close(monkeypatch) -> None:
    fake_runtime = _FakeRuntime()
    cfg = RollioConfig(
        fps=10,
        mode="teleop",
        cameras=[],
        robots=[],
    )

    _run_collection_once(monkeypatch, fake_runtime, ["q"], cfg)

    assert fake_runtime.return_zero_calls == [5.0]
    assert fake_runtime.call_order[-2:] == ["return_zero", "close"]
