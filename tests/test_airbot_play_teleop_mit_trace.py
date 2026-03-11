from __future__ import annotations

import numpy as np
import pytest

from rollio.tests.airbot_play_teleop_mit_trace import (
    TraceSample,
    _encode_plotjuggler_message,
    _build_plotjuggler_message,
    _clamp_g2_target,
    _parse_axis_values,
)


def test_parse_axis_values_expands_scalar() -> None:
    parsed = _parse_axis_values("5.0", expected=6, name="kp")

    np.testing.assert_array_equal(parsed, np.full(6, 5.0))


def test_parse_axis_values_accepts_full_vector() -> None:
    parsed = _parse_axis_values("1,2,3,4,5,6", expected=6, name="kd")

    np.testing.assert_array_equal(parsed, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64))


def test_parse_axis_values_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="6 comma-separated values"):
        _parse_axis_values("1,2,3", expected=6, name="kp")


def test_clamp_g2_target_respects_supported_range() -> None:
    assert _clamp_g2_target(-1.0) == 0.0
    assert _clamp_g2_target(0.03) == 0.03
    assert _clamp_g2_target(0.08) == 0.072


def test_build_plotjuggler_message_flattens_named_series() -> None:
    sample = TraceSample(
        timestamp_sec=123.5,
        leader_position=np.array([1, 2, 3, 4, 5, 6, 0.01], dtype=np.float64),
        mapped_target_position=np.array([10, 20, 30, 40, 50, 60, 0.02], dtype=np.float64),
        follower_position=np.array([11, 19, 29, 41, 49, 61, 0.03], dtype=np.float64),
    )

    message = _build_plotjuggler_message(sample)

    assert message["timestamp"] == 123.5
    assert message["leader/joint1"] == 1.0
    assert message["target/joint6"] == 60.0
    assert message["follower/joint3"] == 29.0
    assert message["leader/e2b"] == 0.01
    assert message["target/g2"] == 0.02
    assert message["follower/g2"] == 0.03
    assert message["error/joint2"] == -1.0
    assert message["error/g2"] == 0.01
    assert message["error/arm_rms"] > 0.0


def test_encode_plotjuggler_message_returns_json_bytes() -> None:
    sample = TraceSample(
        timestamp_sec=1.25,
        leader_position=np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float64),
        mapped_target_position=np.array([1, 1, 1, 1, 1, 1, 0.02], dtype=np.float64),
        follower_position=np.array([2, 2, 2, 2, 2, 2, 0.03], dtype=np.float64),
    )

    encoded = _encode_plotjuggler_message(sample)

    assert isinstance(encoded, bytes)
    assert b" " not in encoded
    assert b"timestamp" in encoded
