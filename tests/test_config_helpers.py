"""Tests for pairing and codec-related config helpers."""

from __future__ import annotations

import subprocess

import pytest

import rollio.episode.codecs as codecs
from rollio.config import (
    CameraConfig,
    EncoderConfig,
    RobotConfig,
    RollioConfig,
    TeleopPairConfig,
    default_mapper_for_pair,
    suggest_teleop_pairs,
    supports_joint_direct_mapping,
    validate_teleop_pairs,
)
from rollio.defaults import DEFAULT_CONTROL_HZ
from rollio.episode.codecs import get_depth_codec_option, get_rgb_codec_option


def _robots() -> list[RobotConfig]:
    return [
        RobotConfig(name="leader_a", type="pseudo", role="leader", num_joints=6),
        RobotConfig(name="leader_b", type="airbot_play", role="leader", num_joints=6),
        RobotConfig(name="follower_a", type="pseudo", role="follower", num_joints=6),
        RobotConfig(
            name="follower_b", type="airbot_play", role="follower", num_joints=6
        ),
    ]


def test_default_mapper_prefers_direct_for_compatible_pairs() -> None:
    leader = RobotConfig(name="leader", type="pseudo", role="leader", num_joints=6)
    follower = RobotConfig(
        name="follower", type="pseudo", role="follower", num_joints=6
    )

    assert default_mapper_for_pair(leader, follower) == "joint_direct"


def test_default_mapper_uses_fkik_for_mixed_robot_types() -> None:
    leader = RobotConfig(name="leader", type="pseudo", role="leader", num_joints=6)
    follower = RobotConfig(
        name="follower", type="airbot_play", role="follower", num_joints=6
    )

    assert default_mapper_for_pair(leader, follower) == "pose_fk_ik"


def test_suggest_teleop_pairs_builds_explicit_default_pairs() -> None:
    pairs = suggest_teleop_pairs(_robots())

    assert [(pair.leader, pair.follower, pair.mapper) for pair in pairs] == [
        ("leader_a", "follower_a", "joint_direct"),
        ("leader_b", "follower_b", "joint_direct"),
    ]


def test_airbot_eef_allowlists_encode_supported_direction() -> None:
    leader_e2b = RobotConfig(
        name="leader_e2b",
        type="airbot_e2b",
        role="leader",
        num_joints=1,
    )
    follower_g2 = RobotConfig(
        name="follower_g2",
        type="airbot_g2",
        role="follower",
        num_joints=1,
    )
    leader_g2 = RobotConfig(
        name="leader_g2",
        type="airbot_g2",
        role="leader",
        num_joints=1,
    )
    follower_e2b = RobotConfig(
        name="follower_e2b",
        type="airbot_e2b",
        role="follower",
        num_joints=1,
    )

    assert leader_e2b.direct_map_allowlist == ["airbot_g2"]
    assert follower_g2.direct_map_allowlist == ["airbot_e2b"]
    assert leader_g2.direct_map_allowlist == []
    assert follower_e2b.direct_map_allowlist == []
    assert supports_joint_direct_mapping(leader_e2b, follower_g2) is True
    assert supports_joint_direct_mapping(leader_g2, follower_e2b) is False


def test_suggest_teleop_pairs_uses_supported_airbot_eef_direction() -> None:
    robots = [
        RobotConfig(name="leader_e2b", type="airbot_e2b", role="leader", num_joints=1),
        RobotConfig(name="leader_g2", type="airbot_g2", role="leader", num_joints=1),
        RobotConfig(
            name="follower_g2", type="airbot_g2", role="follower", num_joints=1
        ),
        RobotConfig(
            name="follower_e2b", type="airbot_e2b", role="follower", num_joints=1
        ),
    ]

    pairs = suggest_teleop_pairs(robots)

    assert [(pair.leader, pair.follower, pair.mapper) for pair in pairs] == [
        ("leader_e2b", "follower_g2", "joint_direct"),
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


def test_rollio_config_defaults_async_pipeline_to_default_control_hz() -> None:
    cfg = RollioConfig(project_name="demo")

    assert cfg.async_pipeline.telemetry_hz == DEFAULT_CONTROL_HZ
    assert cfg.async_pipeline.control_hz == DEFAULT_CONTROL_HZ


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


def test_rollio_config_rejects_duplicate_robot_type_device_pairs() -> None:
    with pytest.raises(ValueError, match="type/device combinations must be unique"):
        RollioConfig(
            project_name="demo",
            robots=[
                RobotConfig(
                    name="leader_arm",
                    type="airbot_play",
                    role="leader",
                    num_joints=6,
                    device="can0",
                ),
                RobotConfig(
                    name="follower_arm",
                    type="airbot_play",
                    role="follower",
                    num_joints=6,
                    device="can0",
                ),
            ],
        )


def test_airbot_play_target_tracking_mode_normalizes_in_robot_options() -> None:
    cfg = RobotConfig(
        name="arm_cfg",
        type="airbot_play",
        role="follower",
        num_joints=6,
        options={"target_tracking_mode": "PVT"},
    )

    assert cfg.options["target_tracking_mode"] == "pvt"


def test_airbot_play_target_tracking_mode_rejects_invalid_option() -> None:
    with pytest.raises(ValueError, match="target_tracking_mode"):
        RobotConfig(
            name="arm_cfg",
            type="airbot_play",
            role="follower",
            num_joints=6,
            options={"target_tracking_mode": "invalid"},
        )


def test_airbot_g2_target_tracking_mode_normalizes_in_robot_options() -> None:
    cfg = RobotConfig(
        name="g2_cfg",
        type="airbot_g2",
        role="follower",
        num_joints=1,
        options={"target_tracking_mode": "PVT"},
    )

    assert cfg.options["target_tracking_mode"] == "pvt"


def test_airbot_g2_target_tracking_mode_rejects_invalid_option() -> None:
    with pytest.raises(ValueError, match="target_tracking_mode"):
        RobotConfig(
            name="g2_cfg",
            type="airbot_g2",
            role="follower",
            num_joints=1,
            options={"target_tracking_mode": "invalid"},
        )


def test_codec_option_lookup_supports_aliases() -> None:
    assert get_rgb_codec_option("mp4v").name == "mpeg4"
    assert get_depth_codec_option("raw").name == "rawvideo"


def test_available_rgb_codecs_keep_nvenc_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    encoder_output = "\n".join(
        [
            "Encoders:",
            " V..... h264_nvenc           NVIDIA NVENC H.264 encoder",
            " V..... libx264              libx264 H.264 / AVC",
            " V..... mpeg4                MPEG-4 part 2",
        ]
    )

    def fake_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        commands.append(command)
        if command == ["ffmpeg", "-hide_banner", "-encoders"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=encoder_output, stderr=""
            )
        if command[0] == "ffmpeg":
            if command[command.index("-c:v") + 1] == "h264_nvenc":
                assert codecs.RGB_PROBE_SOURCE in command
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected ffmpeg command: {command}")

    monkeypatch.setattr(codecs.subprocess, "run", fake_run)
    codecs.discover_ffmpeg_encoders.cache_clear()
    codecs.available_rgb_codec_options.cache_clear()
    try:
        assert [option.name for option in codecs.available_rgb_codec_options()] == [
            "h264_nvenc",
            "libx264",
            "mpeg4",
        ]
    finally:
        codecs.discover_ffmpeg_encoders.cache_clear()
        codecs.available_rgb_codec_options.cache_clear()

    assert any(codecs.RGB_PROBE_SOURCE in command for command in commands)
