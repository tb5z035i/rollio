"""Configuration schema for Rollio — validated with Pydantic."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from rollio.episode.codecs import (
    get_depth_codec_option,
    get_rgb_codec_option,
)
from rollio.robot import robot_class_for_type


# ─── Sub-models ────────────────────────────────────────────────────────


def default_direct_map_allowlist(
    robot_type: str,
    role: Literal["leader", "follower"] | None = None,
) -> list[str]:
    """Return the default direct-mapping allowlist for one robot type."""
    normalized_type = str(robot_type).strip()
    robot_cls = robot_class_for_type(normalized_type)
    if robot_cls is not None:
        return list(robot_cls.default_direct_map_allowlist(normalized_type, role))
    if normalized_type:
        return [normalized_type]
    return []

class CameraChannelConfig(BaseModel):
    """Configuration for a single camera channel/stream."""
    name: str = "color"            # "color", "depth", "infrared", etc.
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "rgb24"    # "rgb24", "MJPG", "YUYV", "z16" (depth), etc.
    enabled: bool = True


class CameraConfig(BaseModel):
    name: str = "cam0"
    type: str = "pseudo"
    device: int | str = 0          # device index or path (for realsense: "serial:channel")
    width: int = 640               # primary channel width (for single-channel compat)
    height: int = 480              # primary channel height
    fps: int = 30                  # primary channel fps
    pixel_format: str = "rgb24"    # primary channel format
    id_path: str = ""              # udev ID_PATH for stable device identification
    channel: str = "color"         # for realsense: "color", "depth", or "infrared"
    options: dict[str, Any] = Field(default_factory=dict)  # backend-specific options
    channels: list[CameraChannelConfig] = Field(
        default_factory=list)      # multi-channel config (empty = single channel mode)

    @field_validator("type", mode="before")
    @classmethod
    def _validate_camera_type(cls, value: str) -> str:
        sensor_type = str(value).strip()
        if not sensor_type:
            raise ValueError("Camera type must be a non-empty string")
        return sensor_type


class RobotConfig(BaseModel):
    name: str = "arm0"
    type: str = "pseudo"
    role: Literal["leader", "follower"] = "follower"
    num_joints: int = 6
    device: str = ""               # CAN bus, serial port, etc.
    direct_map_allowlist: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)  # backend-specific options

    @field_validator("type", mode="before")
    @classmethod
    def _validate_robot_type(cls, value: str) -> str:
        robot_type = str(value).strip()
        if not robot_type:
            raise ValueError("Robot type must be a non-empty string")
        return robot_type

    @model_validator(mode="after")
    def _populate_direct_map_allowlist(self) -> "RobotConfig":
        if self.direct_map_allowlist:
            normalized = [
                str(item).strip()
                for item in self.direct_map_allowlist
                if str(item).strip()
            ]
            self.direct_map_allowlist = list(dict.fromkeys(normalized))
        else:
            self.direct_map_allowlist = default_direct_map_allowlist(
                self.type,
                self.role,
            )
        return self


class StorageConfig(BaseModel):
    root: str = "~/rollio_data"
    lerobot_version: Literal["v2.1", "v3.0"] = "v2.1"


class EncoderConfig(BaseModel):
    video_codec: str = "libx264"
    depth_codec: str = "ffv1"
    background_workers: int = 1

    @field_validator("video_codec", mode="before")
    @classmethod
    def _validate_video_codec(cls, value: str) -> str:
        option = get_rgb_codec_option(str(value))
        return option.name

    @field_validator("depth_codec", mode="before")
    @classmethod
    def _validate_depth_codec(cls, value: str) -> str:
        option = get_depth_codec_option(str(value))
        return option.name


class UploadConfig(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    api_key_env: str = ""


class AsyncPipelineConfig(BaseModel):
    export_queue_size: int = 4
    telemetry_hz: int = 250
    control_hz: int = 250
    max_pending_episodes: int = 8
    allow_drop_preview_frames: bool = True


class TeleopPairConfig(BaseModel):
    name: str
    leader: str
    follower: str
    mapper: Literal["auto", "joint_direct", "pose_fk_ik"] = "auto"


class ControlConfig(BaseModel):
    start_stop: str = " "          # space bar
    keep: str = "k"
    discard: str = "d"


# ─── Top-level config ─────────────────────────────────────────────────

class RollioConfig(BaseModel):
    project_name: str = "default"
    description: str = ""
    fps: int = 30
    mode: Literal["teleop", "intervention"] = "teleop"

    cameras: list[CameraConfig] = Field(
        default_factory=lambda: [CameraConfig()])
    robots: list[RobotConfig] = Field(
        default_factory=lambda: [RobotConfig()])
    teleop_pairs: list[TeleopPairConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    async_pipeline: AsyncPipelineConfig = Field(default_factory=AsyncPipelineConfig)
    controls: ControlConfig = Field(default_factory=ControlConfig)

    @model_validator(mode="after")
    def _validate_explicit_pairs(self) -> "RollioConfig":
        duplicate_camera_names = [
            name for name, count in Counter(camera.name for camera in self.cameras).items()
            if count > 1
        ]
        if duplicate_camera_names:
            raise ValueError(
                "Camera names must be unique: "
                + ", ".join(sorted(duplicate_camera_names))
            )

        duplicate_robot_names = [
            name for name, count in Counter(robot.name for robot in self.robots).items()
            if count > 1
        ]
        if duplicate_robot_names:
            raise ValueError(
                "Robot names must be unique: "
                + ", ".join(sorted(duplicate_robot_names))
            )

        duplicate_robot_device_keys = [
            f"{robot_type}@{device}"
            for (robot_type, device), count in Counter(
                (robot.type, robot.device.strip())
                for robot in self.robots
                if robot.device.strip()
            ).items()
            if count > 1
        ]
        if duplicate_robot_device_keys:
            raise ValueError(
                "Robot type/device combinations must be unique: "
                + ", ".join(sorted(duplicate_robot_device_keys))
            )

        if self.teleop_pairs:
            from rollio.config.pairing import validate_teleop_pairs
            validate_teleop_pairs(self.robots, self.teleop_pairs)
        return self

    # ── I/O helpers ────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(
            self.model_dump(), default_flow_style=False, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "RollioConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> "RollioConfig":
        return cls()
