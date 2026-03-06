"""Configuration schema for Rollio — validated with Pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from rollio.episode.codecs import (
    get_depth_codec_option,
    get_rgb_codec_option,
)


# ─── Sub-models ────────────────────────────────────────────────────────

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
    type: Literal["pseudo", "v4l2", "realsense"] = "pseudo"
    device: int | str = 0          # device index or path (for realsense: "serial:channel")
    width: int = 640               # primary channel width (for single-channel compat)
    height: int = 480              # primary channel height
    fps: int = 30                  # primary channel fps
    pixel_format: str = "rgb24"    # primary channel format
    id_path: str = ""              # udev ID_PATH for stable device identification
    channel: str = "color"         # for realsense: "color", "depth", or "infrared"
    channels: list[CameraChannelConfig] = Field(
        default_factory=list)      # multi-channel config (empty = single channel mode)


class RobotConfig(BaseModel):
    name: str = "arm0"
    type: Literal["pseudo", "airbot_play", "nero"] = "pseudo"
    role: Literal["leader", "follower"] = "follower"
    num_joints: int = 6
    device: str = ""               # CAN bus, serial port, etc.


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
    telemetry_hz: int = 50
    control_hz: int = 100
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
