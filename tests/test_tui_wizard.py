"""Tests for setup wizard project settings flow."""
from __future__ import annotations

import io

import numpy as np

from rollio.defaults import DEFAULT_CONTROL_HZ, DEFAULT_CONTROL_INTERVAL_MS
from rollio.config.schema import RobotConfig
from rollio.episode.codecs import (
    available_depth_codec_options,
    available_rgb_codec_options,
)
from rollio.sensors.scanner import DetectedDevice
from rollio.tui import wizard


class _FakeTerm:
    cols = 80
    rows = 24

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = iter(keys or ["x"])

    def __enter__(self) -> "_FakeTerm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read_key_blocking(self, timeout: float = 0.05) -> str | None:
        return next(self._keys, "x")

    def read_key(self) -> str | None:
        return next(self._keys, None)


def test_term_decodes_arrow_escape_sequences(monkeypatch) -> None:
    class _FakeStdin:
        def __init__(self, chars: list[str]) -> None:
            self._chars = chars

        def fileno(self) -> int:
            return 0

        def read(self, size: int) -> str:
            assert size == 1
            return self._chars.pop(0)

    fake_stdin = _FakeStdin(list("\x1b[1;2A"))

    def fake_select(readers, _writers, _errors, _timeout):
        if readers and fake_stdin._chars:
            return ([fake_stdin], [], [])
        return ([], [], [])

    monkeypatch.setattr(wizard.sys, "stdin", fake_stdin)
    monkeypatch.setattr(wizard.select, "select", fake_select)

    term = wizard._Term()

    assert term.read_key_blocking(0.0) == "UP"


def test_pick_option_supports_arrow_navigation() -> None:
    result = wizard._pick_option(
        _FakeTerm(["DOWN", "\n"]),
        io.BytesIO(),
        title="TEST PICKER",
        options=["first", "second", "third"],
        current_idx=0,
    )

    assert result == 1


def test_airbot_led_block_renders_blink_states() -> None:
    blink_on = wizard._airbot_led_block(blink_on=True, width=8)
    blink_off = wizard._airbot_led_block(blink_on=False, width=8)

    assert "\x1b[48;5;208m" in blink_on
    assert "\x1b[48;5;236m" in blink_off


def test_airbot_led_block_defaults_to_4hz_cycle(monkeypatch) -> None:
    monkeypatch.setattr(wizard.time, "monotonic", lambda: 0.00)
    assert "\x1b[48;5;208m" in wizard._airbot_led_block(width=8)

    monkeypatch.setattr(wizard.time, "monotonic", lambda: 0.13)
    assert "\x1b[48;5;236m" in wizard._airbot_led_block(width=8)

    monkeypatch.setattr(wizard.time, "monotonic", lambda: 0.26)
    assert "\x1b[48;5;208m" in wizard._airbot_led_block(width=8)


def test_format_joint_preview_uses_70mm_range_for_airbot_eefs() -> None:
    text, frac = wizard._format_joint_preview("airbot_e2b", 0.035)

    assert text.strip() == "35.0mm"
    assert frac == 0.5


def test_format_control_interval_preview_uses_target_ratio() -> None:
    text, frac = wizard._format_control_interval_preview(
        DEFAULT_CONTROL_INTERVAL_MS * 2.0,
        DEFAULT_CONTROL_INTERVAL_MS,
    )

    assert text.strip() == f"{DEFAULT_CONTROL_INTERVAL_MS * 2.0:0.1f}ms"
    assert frac == 0.5


def test_screen_settings_warns_before_codecs_when_teleop_has_no_robots(
    monkeypatch,
) -> None:
    prompts = iter(["demo", "~/rollio_data"])
    pick_calls: list[str] = []

    monkeypatch.setattr(
        wizard,
        "_prompt_line",
        lambda *args, **kwargs: next(prompts),
    )

    def fake_pick_option(*args, **kwargs):
        pick_calls.append(kwargs["title"])
        if len(pick_calls) == 1:
            return 0
        raise AssertionError("Codec selection should not be reached")

    monkeypatch.setattr(wizard, "_pick_option", fake_pick_option)

    out = io.BytesIO()
    result = wizard._screen_settings(
        _FakeTerm(["x"]),
        out,
        [],
        step=3,
        total_steps=5,
    )

    assert result is None
    assert pick_calls == ["COLLECTION MODE"]
    assert "No robots were configured in the previous step." in out.getvalue().decode(
        "utf-8",
        errors="ignore",
    )


def test_screen_settings_allows_valid_teleop_to_continue_to_codecs(
    monkeypatch,
) -> None:
    prompts = iter(["demo", "~/rollio_data"])
    picks = iter([0, 0, 0])

    monkeypatch.setattr(
        wizard,
        "_prompt_line",
        lambda *args, **kwargs: next(prompts),
    )
    monkeypatch.setattr(
        wizard,
        "_pick_option",
        lambda *args, **kwargs: next(picks),
    )

    result = wizard._screen_settings(
        _FakeTerm(),
        io.BytesIO(),
        [
            RobotConfig(name="leader_arm", role="leader"),
            RobotConfig(name="follower_arm", role="follower"),
        ],
        step=3,
        total_steps=5,
    )

    assert result == (
        "demo",
        "~/rollio_data",
        "teleop",
        list(available_rgb_codec_options())[0].name,
        list(available_depth_codec_options())[0].name,
    )


def test_screen_summary_uses_shared_preview_runtime(monkeypatch) -> None:
    created: list[tuple[object, str, bool]] = []

    class _FakePreviewRuntime:
        def __init__(self) -> None:
            self.open_called = False
            self.close_called = False
            self.return_zero_called = False
            self.scheduler_driver = "asyncio"
            self.telemetry_hz = DEFAULT_CONTROL_HZ
            self.control_hz = DEFAULT_CONTROL_HZ

        def open(self) -> None:
            self.open_called = True

        def close(self) -> None:
            self.close_called = True

        def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
            del timeout
            self.return_zero_called = True
            return {}

        def latest_frames(self) -> dict[str, object]:
            return {}

        def latest_robot_states(self) -> dict[str, dict[str, object]]:
            return {}

        def scheduler_metrics(self) -> dict[str, object]:
            class _TaskMetrics:
                def __init__(self) -> None:
                    self.run_count = 10
                    self.overrun_count = 0
                    self.avg_step_ms = 1.0

            class _DriverMetrics:
                def __init__(self) -> None:
                    self.task_metrics = {
                        "robot-arm0": _TaskMetrics(),
                        "teleop-pair_0": _TaskMetrics(),
                    }

            return {"driver": _DriverMetrics()}

    preview_runtime = _FakePreviewRuntime()

    def fake_from_config(
        cfg,
        scheduler_driver="asyncio",
        preview_live_feedback=False,
    ):
        created.append((cfg, scheduler_driver, preview_live_feedback))
        return preview_runtime

    monkeypatch.setattr(
        wizard.AsyncCollectionRuntime,
        "from_config",
        staticmethod(fake_from_config),
    )

    result = wizard._screen_summary(
        _FakeTerm(["\n"]),
        io.BytesIO(),
        [],
        [],
        [],
        [],
        "demo",
        "~/rollio_data",
        "rollio_config.yaml",
        mode="teleop",
        video_codec="mp4v",
        depth_codec="raw",
        teleop_pairs=[],
        step=5,
        total_steps=5,
    )

    assert result is True
    assert created
    assert created[0][1] == "asyncio"
    assert created[0][2] is True
    assert preview_runtime.open_called is True
    assert preview_runtime.return_zero_called is True
    assert preview_runtime.close_called is True


def test_match_robot_devices_distinguishes_same_can_by_type() -> None:
    configs = [
        RobotConfig(
            name="leader_arm",
            type="airbot_play",
            role="leader",
            num_joints=6,
            device="can0",
        ),
        RobotConfig(
            name="follower_g2",
            type="airbot_g2",
            role="follower",
            num_joints=1,
            device="can0",
        ),
    ]
    devices = [
        DetectedDevice(
            kind="robot",
            dtype="airbot_play",
            device_id="can0",
            label="AIRBOT Play (can0)",
            properties={"can_interface": "can0", "num_joints": 6},
        ),
        DetectedDevice(
            kind="robot",
            dtype="airbot_g2",
            device_id="can0",
            label="AIRBOT G2 (can0)",
            properties={"can_interface": "can0", "num_joints": 1},
        ),
    ]

    matched = wizard._match_robot_devices(configs, devices)

    assert matched[0][1] is devices[0]
    assert matched[1][1] is devices[1]


def test_screen_robots_persists_airbot_e2b_as_separate_entity(monkeypatch) -> None:
    prompts = iter(["eef_demo", "l"])
    monkeypatch.setattr(wizard, "_prompt_line", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(wizard, "_get_airbot_robot", lambda *args, **kwargs: None)
    monkeypatch.setattr(wizard.time, "sleep", lambda *args, **kwargs: None)

    configs = wizard._screen_robots(
        _FakeTerm(["\n"]),
        io.BytesIO(),
        [
            DetectedDevice(
                kind="robot",
                dtype="airbot_e2b",
                device_id="can0",
                label="AIRBOT E2B (can0)",
                properties={"can_interface": "can0", "num_joints": 1},
            )
        ],
    )

    assert configs is not None
    assert configs == [
        RobotConfig(
            name="eef_demo",
            type="airbot_e2b",
            role="leader",
            num_joints=1,
            device="can0",
            direct_map_allowlist=["airbot_g2"],
        )
    ]


def test_screen_robots_shows_airbot_play_joint_positions(monkeypatch) -> None:
    prompts = iter(["leader_arm", "l", "m"])
    out = io.BytesIO()

    class _FakeAirbotPlay:
        def __init__(self) -> None:
            self._is_open = True
            self.identify_started = 0
            self.identify_steps = 0
            self.identify_stopped = 0

        def identify_start(self, with_gravity_comp: bool = True) -> bool:
            assert with_gravity_comp is True
            self.identify_started += 1
            return True

        def identify_step(self) -> None:
            self.identify_steps += 1

        def read_joint_state(self):
            class _State:
                is_valid = True
                position = np.array([0.12, -0.34], dtype=np.float32)

            return _State()

        def identify_stop(self) -> bool:
            self.identify_stopped += 1
            return True

    fake_robot = _FakeAirbotPlay()
    monkeypatch.setattr(wizard, "_prompt_line", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(wizard, "_get_airbot_robot", lambda *args, **kwargs: fake_robot)
    monkeypatch.setattr(wizard.time, "sleep", lambda *args, **kwargs: None)

    configs = wizard._screen_robots(
        _FakeTerm(["\n"]),
        out,
        [
            DetectedDevice(
                kind="robot",
                dtype="airbot_play",
                device_id="can0",
                label="AIRBOT Play (can0)",
                properties={"can_interface": "can0", "num_joints": 2},
            )
        ],
    )

    assert configs is not None
    assert fake_robot.identify_started == 1
    assert fake_robot.identify_steps >= 1
    assert fake_robot.identify_stopped == 1
    assert configs == [
        RobotConfig(
            name="leader_arm",
            type="airbot_play",
            role="leader",
            num_joints=2,
            device="can0",
            options={"target_tracking_mode": "mit"},
            direct_map_allowlist=["airbot_play"],
        )
    ]
    rendered = out.getvalue().decode("utf-8", errors="ignore")
    assert "gravity compensation" in rendered
    assert "j0" in rendered
    assert "+0.12" in rendered


def test_screen_robots_persists_airbot_play_pvt_tracking_mode(monkeypatch) -> None:
    prompts = iter(["follower_arm", "f", "p"])
    monkeypatch.setattr(wizard, "_prompt_line", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(wizard, "_get_airbot_robot", lambda *args, **kwargs: None)
    monkeypatch.setattr(wizard.time, "sleep", lambda *args, **kwargs: None)

    configs = wizard._screen_robots(
        _FakeTerm(["\n"]),
        io.BytesIO(),
        [
            DetectedDevice(
                kind="robot",
                dtype="airbot_play",
                device_id="can1",
                label="AIRBOT Play (can1)",
                properties={"can_interface": "can1", "num_joints": 6},
            )
        ],
    )

    assert configs == [
        RobotConfig(
            name="follower_arm",
            type="airbot_play",
            role="follower",
            num_joints=6,
            device="can1",
            options={"target_tracking_mode": "pvt"},
            direct_map_allowlist=["airbot_play"],
        )
    ]


def test_screen_robots_steps_g2_identification_preview(monkeypatch) -> None:
    prompts = iter(["gripper_demo", "f"])
    out = io.BytesIO()

    class _FakeG2Robot:
        def __init__(self) -> None:
            self._is_open = True
            self.identify_started = 0
            self.identify_steps = 0
            self.identify_stopped = 0

        def identify_start(self) -> bool:
            self.identify_started += 1
            return True

        def identify_step(self) -> None:
            self.identify_steps += 1

        def read_joint_state(self):
            class _State:
                is_valid = True
                position = np.array([0.035], dtype=np.float32)

            return _State()

        def latest_command_debug(self) -> tuple[str, str]:
            return ("PVT", "pos=[ 0.0350] vel=[200.0000] current_threshold=[200.0000]")

        def identify_stop(self) -> bool:
            self.identify_stopped += 1
            return True

    fake_robot = _FakeG2Robot()
    monkeypatch.setattr(wizard, "_prompt_line", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(wizard, "_get_airbot_robot", lambda *args, **kwargs: fake_robot)
    monkeypatch.setattr(wizard.time, "sleep", lambda *args, **kwargs: None)

    configs = wizard._screen_robots(
        _FakeTerm(["\n"]),
        out,
        [
            DetectedDevice(
                kind="robot",
                dtype="airbot_g2",
                device_id="can0",
                label="AIRBOT G2 (can0)",
                properties={"can_interface": "can0", "num_joints": 1},
            )
        ],
    )

    assert configs is not None
    assert fake_robot.identify_started == 1
    assert fake_robot.identify_steps >= 1
    assert fake_robot.identify_stopped == 1
    rendered = out.getvalue().decode("utf-8", errors="ignore")
    assert "G2 oscillation" in rendered
    assert "Cmd: PVT" in rendered
    assert "current_threshold=[200.0000]" in rendered


def test_screen_cameras_sizes_preview_from_actual_frame(monkeypatch) -> None:
    calc_calls: list[tuple[int, int, int, int]] = []

    class _FakeCamera:
        def read(self) -> tuple[float, np.ndarray]:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            return 0.0, frame

        def close(self) -> None:
            return None

    def fake_calc_render_size(
        cam_w: int,
        cam_h: int,
        avail_w: int,
        avail_h: int,
    ) -> tuple[int, int]:
        calc_calls.append((cam_w, cam_h, avail_w, avail_h))
        return 20, 10

    monkeypatch.setattr(wizard, "_make_camera", lambda *args, **kwargs: _FakeCamera())
    monkeypatch.setattr(wizard, "calc_render_size", fake_calc_render_size)
    monkeypatch.setattr(wizard, "render_frame", lambda *args, **kwargs: b"frame")
    monkeypatch.setattr(wizard, "blit_frame", lambda *args, **kwargs: b"")

    result = wizard._screen_cameras(
        _FakeTerm(["s"]),
        io.BytesIO(),
        [
            DetectedDevice(
                kind="camera",
                dtype="v4l2",
                device_id=0,
                label="USB Camera",
                properties={},
                width=640,
                height=480,
                fps=30,
                pixel_format="MJPG",
            )
        ],
    )

    assert result == []
    assert len(calc_calls) == 1
    assert calc_calls[0][:2] == (1280, 720)


def test_screen_cameras_returns_none_when_quit_pressed(monkeypatch) -> None:
    class _FakeCamera:
        def __init__(self) -> None:
            self.close_called = False

        def read(self) -> tuple[float, None]:
            return 0.0, None

        def close(self) -> None:
            self.close_called = True

    fake_camera = _FakeCamera()

    monkeypatch.setattr(wizard, "_make_camera", lambda *args, **kwargs: fake_camera)
    monkeypatch.setattr(wizard, "calc_render_size", lambda *args, **kwargs: (20, 10))

    result = wizard._screen_cameras(
        _FakeTerm(["q"]),
        io.BytesIO(),
        [
            DetectedDevice(
                kind="camera",
                dtype="v4l2",
                device_id=0,
                label="USB Camera",
                properties={},
                width=640,
                height=480,
                fps=30,
                pixel_format="MJPG",
            )
        ],
    )

    assert result is None
    assert fake_camera.close_called is True


def test_run_wizard_aborts_when_camera_screen_cancels(monkeypatch) -> None:
    class _FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO()
            self._text = io.StringIO()

        def write(self, value: str) -> int:
            return self._text.write(value)

        def flush(self) -> None:
            return None

    monkeypatch.setattr(wizard, "scan_cameras", lambda *args, **kwargs: [])
    monkeypatch.setattr(wizard, "scan_robots", lambda *args, **kwargs: [])
    monkeypatch.setattr(wizard.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(wizard, "_Term", _FakeTerm)
    monkeypatch.setattr(wizard.sys, "stdout", _FakeStdout())
    monkeypatch.setattr(wizard, "_screen_cameras", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wizard,
        "_screen_robots",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("wizard should stop after camera cancel")
        ),
    )

    result = wizard.run_wizard("rollio_config.yaml")

    assert result is None
