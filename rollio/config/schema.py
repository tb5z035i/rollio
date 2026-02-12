"""Configuration schema for Rollio — validated with Pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


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
    storage: StorageConfig = Field(default_factory=StorageConfig)
    controls: ControlConfig = Field(default_factory=ControlConfig)

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
