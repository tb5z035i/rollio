from __future__ import annotations

import numpy as np
import pytest

import rollio.tests.airbot_play_teleop_mit_trace as mit_trace
from rollio.tests.airbot_play_teleop_mit_trace import (
    PLOTJUGGLER_HOST,
    TraceSample,
    _build_arg_parser,
    _clamp_g2_target,
    _normalize_plotjuggler_host,
    _parse_axis_values,
)


def test_parse_axis_values_expands_scalar() -> None:
    parsed = _parse_axis_values("5.0", expected=6, name="kp")

    np.testing.assert_array_equal(parsed, np.full(6, 5.0))


def test_parse_axis_values_accepts_full_vector() -> None:
    parsed = _parse_axis_values("1,2,3,4,5,6", expected=6, name="kd")

    np.testing.assert_array_equal(
        parsed, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    )


def test_parse_axis_values_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="6 comma-separated values"):
        _parse_axis_values("1,2,3", expected=6, name="kp")


def test_clamp_g2_target_respects_supported_range() -> None:
    assert _clamp_g2_target(-1.0) == 0.0
    assert _clamp_g2_target(0.03) == 0.03
    assert _clamp_g2_target(0.08) == 0.072


def test_arg_parser_no_longer_accepts_g2_velocity_flag() -> None:
    parser = _build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--g2-velocity", "25"])


def test_normalize_plotjuggler_host_accepts_localhost_alias() -> None:
    assert _normalize_plotjuggler_host("localhost") == PLOTJUGGLER_HOST
    assert _normalize_plotjuggler_host(PLOTJUGGLER_HOST) == PLOTJUGGLER_HOST


def test_normalize_plotjuggler_host_rejects_remote_host() -> None:
    with pytest.raises(ValueError, match="only supports local UDP"):
        _normalize_plotjuggler_host("192.168.1.5")


def test_publish_plotjuggler_sample_uses_internal_rollio_publisher(
    monkeypatch,
) -> None:
    sample = TraceSample(
        timestamp_sec=123.5,
        leader_position=np.array([1, 2, 3, 4, 5, 6, 0.01], dtype=np.float64),
        mapped_target_position=np.array(
            [10, 20, 30, 40, 50, 60, 0.02], dtype=np.float64
        ),
        follower_position=np.array([11, 19, 29, 41, 49, 61, 0.03], dtype=np.float64),
    )
    published: list[tuple[str, float, tuple[float, ...]]] = []

    monkeypatch.setattr(
        mit_trace,
        "publish_joint_state",
        lambda name, timestamp, position: published.append((name, timestamp, position)),
    )

    mit_trace._publish_plotjuggler_sample(sample)  # pylint: disable=protected-access

    assert [name for name, _, _ in published] == [
        "leader_arm",
        "target_arm",
        "follower_arm",
        "error_arm",
        "leader_e2b",
        "target_g2",
        "follower_g2",
        "error_g2",
        "error_arm_rms",
    ]
    assert all(timestamp == 123.5 for _, timestamp, _ in published)
    published_by_name = {name: position for name, _, position in published}
    assert published_by_name["leader_arm"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert published_by_name["target_arm"] == (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    assert published_by_name["follower_arm"] == (11.0, 19.0, 29.0, 41.0, 49.0, 61.0)
    assert published_by_name["error_arm"] == (1.0, -1.0, -1.0, 1.0, -1.0, 1.0)
    assert published_by_name["leader_e2b"] == (0.01,)
    assert published_by_name["target_g2"] == (0.02,)
    assert published_by_name["follower_g2"] == (0.03,)
    assert published_by_name["error_g2"] == pytest.approx((0.01,))
    assert published_by_name["error_arm_rms"] == pytest.approx((float(np.sqrt(1.0)),))
