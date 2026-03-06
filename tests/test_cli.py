"""Tests for rollio CLI argument handling."""
from __future__ import annotations

import sys
import types
from pathlib import Path

from rollio import cli


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

    output_path = tmp_path / "default_rollio.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollio", "setup", "--output", str(output_path)],
    )

    cli.main()

    assert calls == [(str(output_path), 0, 0)]
