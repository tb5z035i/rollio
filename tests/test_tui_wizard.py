"""Tests for setup wizard project settings flow."""
from __future__ import annotations

import io

from rollio.config.schema import RobotConfig
from rollio.episode.codecs import (
    available_depth_codec_options,
    available_rgb_codec_options,
)
from rollio.tui import wizard


class _FakeTerm:
    cols = 80
    rows = 24

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = iter(keys or ["x"])

    def read_key_blocking(self, timeout: float = 0.05) -> str | None:
        return next(self._keys, "x")


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
