"""Tests for rollio CLI argument handling."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import pytest

from rollio import cli
from rollio.config.schema import RollioConfig


class _FakeConfig:
    def __init__(self) -> None:
        self.project_name = "demo"
        self.cameras = []
        self.robots = []
        self.storage = types.SimpleNamespace(root="~/rollio_data")
        self.saved_to: Path | None = None

    def save(self, path: str | Path) -> None:
        self.saved_to = Path(path)


def test_setup_cli_passes_simulated_device_counts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[tuple[str, int, int]] = []
    fake_cfg = _FakeConfig()

    def fake_run_wizard(
        output_path: str,
        *,
        simulated_cameras: int = 0,
        simulated_arms: int = 0,
    ):
        calls.append((output_path, simulated_cameras, simulated_arms))
        return fake_cfg

    monkeypatch.setitem(
        sys.modules,
        "rollio.tui.wizard",
        types.SimpleNamespace(run_wizard=fake_run_wizard),
    )
    monkeypatch.setattr(cli, "_acquire_setup_lock", contextlib.nullcontext)

    output_path = tmp_path / "custom_rollio.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollio",
            "setup",
            "--output",
            str(output_path),
            "--sim-cameras",
            "2",
            "--sim-arms",
            "3",
        ],
    )

    cli.main()

    assert calls == [(str(output_path), 2, 3)]
    assert fake_cfg.saved_to == output_path
    stdout = capsys.readouterr().out
    assert "Config saved" in stdout
    assert "Cameras:  []" in stdout
    assert "Robots:   []" in stdout


def test_setup_cli_defaults_to_no_simulated_devices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, int]] = []
    fake_cfg = _FakeConfig()

    def fake_run_wizard(
        output_path: str,
        *,
        simulated_cameras: int = 0,
        simulated_arms: int = 0,
    ):
        calls.append((output_path, simulated_cameras, simulated_arms))
        return fake_cfg

    monkeypatch.setitem(
        sys.modules,
        "rollio.tui.wizard",
        types.SimpleNamespace(run_wizard=fake_run_wizard),
    )
    monkeypatch.setattr(cli, "_acquire_setup_lock", contextlib.nullcontext)

    output_path = tmp_path / "default_rollio.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollio", "setup", "--output", str(output_path)],
    )

    cli.main()

    assert calls == [(str(output_path), 0, 0)]


def test_setup_cli_rejects_second_running_setup(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("run_wizard should not be called when setup lock is held")

    class _FailingSetupLock:
        def __enter__(self):
            raise cli.SetupAlreadyRunningError(
                "Another rollio setup is already running. Lock holder pid: 1234."
            )

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(cli, "_acquire_setup_lock", lambda: _FailingSetupLock())
    monkeypatch.setitem(
        sys.modules,
        "rollio.tui.wizard",
        types.SimpleNamespace(run_wizard=should_not_run),
    )

    output_path = tmp_path / "locked_rollio.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollio", "setup", "--output", str(output_path)],
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    stderr = capsys.readouterr().err
    assert "Another rollio setup is already running." in stderr
    assert "1234" in stderr


def test_replay_cli_dispatches_episode_path(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    fake_cfg = RollioConfig(
        project_name="demo",
        cameras=[],
        robots=[],
    )
    config_path = tmp_path / "rollio_config.yaml"
    config_path.write_text("project_name: demo\n")
    episode_path = tmp_path / "episode_000123.parquet"
    episode_path.write_text("placeholder")

    calls: list[tuple[RollioConfig, Path]] = []

    def fake_run_replay(cfg: RollioConfig, episode: str | Path) -> None:
        calls.append((cfg, Path(episode)))

    monkeypatch.setattr(RollioConfig, "load", staticmethod(lambda _path: fake_cfg))
    monkeypatch.setitem(
        sys.modules,
        "rollio.tui.replay",
        types.SimpleNamespace(run_replay=fake_run_replay),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollio",
            "replay",
            str(episode_path),
            "--config",
            str(config_path),
        ],
    )

    cli.main()

    assert calls == [(fake_cfg, episode_path)]
    stdout = capsys.readouterr().out
    assert "Loaded config" in stdout
    assert "Replay episode:" in stdout
    assert "Starting replay TUI" in stdout


def test_replay_cli_rejects_missing_config(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    missing_config = tmp_path / "missing_rollio.yaml"
    episode_path = tmp_path / "episode_000001.parquet"
    episode_path.write_text("placeholder")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollio",
            "replay",
            str(episode_path),
            "--config",
            str(missing_config),
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    stdout = capsys.readouterr().out
    assert "Config not found" in stdout
