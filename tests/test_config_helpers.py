"""Tests for pairing and codec-related config helpers."""
from __future__ import annotations

import pytest

from rollio.config import (
    CameraConfig,
    EncoderConfig,
    RobotConfig,
    RollioConfig,
    TeleopPairConfig,
    default_mapper_for_pair,
    suggest_teleop_pairs,
    validate_teleop_pairs,
)
from rollio.episode.codecs import get_depth_codec_option, get_rgb_codec_option


def _robots() -> list[RobotConfig]:
    return [
        RobotConfig(name="leader_a", type="pseudo", role="leader", num_joints=6),
        RobotConfig(name="leader_b", type="airbot_play", role="leader", num_joints=6),
        RobotConfig(name="follower_a", type="pseudo", role="follower", num_joints=6),
        RobotConfig(name="follower_b", type="airbot_play", role="follower", num_joints=6),
    ]


def test_default_mapper_prefers_direct_for_compatible_pairs() -> None:
    leader = RobotConfig(name="leader", type="pseudo", role="leader", num_joints=6)
    follower = RobotConfig(name="follower", type="pseudo", role="follower", num_joints=6)

    assert default_mapper_for_pair(leader, follower) == "joint_direct"


def test_default_mapper_uses_fkik_for_mixed_robot_types() -> None:
    leader = RobotConfig(name="leader", type="pseudo", role="leader", num_joints=6)
    follower = RobotConfig(name="follower", type="airbot_play", role="follower", num_joints=6)

    assert default_mapper_for_pair(leader, follower) == "pose_fk_ik"


def test_suggest_teleop_pairs_builds_explicit_default_pairs() -> None:
    pairs = suggest_teleop_pairs(_robots())

    assert [(pair.leader, pair.follower, pair.mapper) for pair in pairs] == [
        ("leader_a", "follower_a", "joint_direct"),
        ("leader_b", "follower_b", "joint_direct"),
    ]


def test_validate_teleop_pairs_rejects_duplicate_followers() -> None:
    with pytest.raises(ValueError, match="follower"):
        validate_teleop_pairs(
            _robots(),
            [
                TeleopPairConfig(
                    name="pair_0",
                    leader="leader_a",
                    follower="follower_a",
                    mapper="joint_direct",
                ),
                TeleopPairConfig(
                    name="pair_1",
                    leader="leader_b",
                    follower="follower_a",
                    mapper="pose_fk_ik",
                ),
            ],
        )


def test_encoder_config_normalizes_legacy_aliases() -> None:
    cfg = EncoderConfig(video_codec="mp4v", depth_codec="raw")

    assert cfg.video_codec == "mpeg4"
    assert cfg.depth_codec == "rawvideo"


def test_rollio_config_validates_explicit_pair_references() -> None:
    with pytest.raises(ValueError, match="Unknown follower robot"):
        RollioConfig(
            project_name="demo",
            mode="teleop",
            cameras=[CameraConfig(name="cam0", type="pseudo")],
            robots=_robots(),
            teleop_pairs=[
                TeleopPairConfig(
                    name="pair_0",
                    leader="leader_a",
                    follower="missing_follower",
                    mapper="joint_direct",
                ),
            ],
        )


def test_codec_option_lookup_supports_aliases() -> None:
    assert get_rgb_codec_option("mp4v").name == "mpeg4"
    assert get_depth_codec_option("raw").name == "rawvideo"
