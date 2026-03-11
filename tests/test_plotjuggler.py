"""Tests for PlotJuggler UDP payload helpers."""

from __future__ import annotations

import json

from rollio.plotjuggler import (
    DEFAULT_PLOTJUGGLER_PORT,
    _build_plotjuggler_message,
    _encode_plotjuggler_message,
    _get_plotjuggler_port,
)


def test_build_plotjuggler_message_nests_robot_joint_series() -> None:
    message = _build_plotjuggler_message(
        "leader_arm",
        12.5,
        (0.1, -0.2, 0.3),
    )

    assert message == {
        "timestamp": 12.5,
        "leader_arm": {
            "j0": 0.1,
            "j1": -0.2,
            "j2": 0.3,
        },
    }


def test_encode_plotjuggler_message_returns_json_bytes() -> None:
    encoded = _encode_plotjuggler_message(
        "leader_gripper",
        3.25,
        (0.04,),
    )

    decoded = json.loads(encoded.decode("utf-8"))
    assert decoded == {
        "timestamp": 3.25,
        "leader_gripper": {
            "j0": 0.04,
        },
    }


def test_get_plotjuggler_port_uses_env_when_valid(monkeypatch) -> None:
    monkeypatch.setenv("PLOTJUGGLER_PORT", "9988")

    assert _get_plotjuggler_port() == 9988


def test_get_plotjuggler_port_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("PLOTJUGGLER_PORT", "not-a-port")

    assert _get_plotjuggler_port() == DEFAULT_PLOTJUGGLER_PORT
