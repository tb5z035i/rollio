"""Abstract base classes for all sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rollio.sensors.scanner import DetectedDevice


@dataclass
class SensorInfo:
    """Metadata describing a sensor."""

    name: str
    sensor_type: str  # "camera", "robot"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraMode:
    """A supported camera resolution/framerate combination."""

    width: int
    height: int
    fps: int

    def __str__(self) -> str:
        return f"{self.width}×{self.height}@{self.fps}fps"


@dataclass
class CameraFormat:
    """A supported pixel format with its available modes."""

    fourcc: str  # "YUYV", "MJPG", etc.
    description: str  # Human-readable name
    modes: list[CameraMode] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.fourcc} ({self.description})"


@dataclass
class CameraSettings:
    """Current camera configuration/settings."""

    width: int
    height: int
    fps: int
    pixel_format: str


@dataclass
class CameraChannel:
    """Description of a camera channel/stream."""

    name: str  # "color", "depth", "infrared", etc.
    default_width: int = 640
    default_height: int = 480
    default_fps: int = 30
    pixel_format: str = "rgb24"  # native format
    description: str = ""  # human-readable description


class ImageSensor(ABC):
    """Abstract interface for cameras.

    Subclasses should implement class methods for device scanning:
    - scan(): Detect available devices of this type
    - probe_formats(): Probe formats without instantiating
    """

    # ── Class-level type identifier ───────────────────────────────────

    SENSOR_TYPE: str = "unknown"  # Override in subclasses: "pseudo", "v4l2", etc.

    # ── Factory / scanning class methods ──────────────────────────────

    @classmethod
    def scan(cls) -> list["DetectedDevice"]:
        """Scan for available devices of this sensor type.

        Returns a list of DetectedDevice objects.
        Subclasses should override this method.
        """
        return []

    @classmethod
    def probe_formats(cls, device_id: int | str) -> list[CameraFormat]:
        """Probe available formats for a device without instantiating.

        Subclasses should override for hardware enumeration.
        """
        return []

    @classmethod
    def get_channels(cls) -> list[CameraChannel]:
        """Return available channels for this camera type.

        Most cameras have a single 'color' channel.
        RealSense-type cameras may have 'color', 'depth', 'infrared', etc.
        """
        return [CameraChannel(name="color", description="RGB Color")]

    # ── Instance methods ──────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[float, np.ndarray]:
        """Return (timestamp_sec, bgr_frame)."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def info(self) -> SensorInfo: ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def fps(self) -> int: ...

    # ── Parameter probing interface (optional, with defaults) ─────────

    def list_formats(self) -> list[CameraFormat]:
        """Return available pixel formats and their modes.

        Default implementation returns a single pseudo format.
        Subclasses should override for real hardware enumeration.
        """
        return [
            CameraFormat(
                fourcc="RGB",
                description="RGB24",
                modes=[CameraMode(self.width, self.height, self.fps)],
            )
        ]

    def get_config(self) -> CameraSettings:
        """Return current camera configuration."""
        return CameraSettings(
            width=self.width, height=self.height, fps=self.fps, pixel_format="RGB"
        )

    def apply_config(
        self, width: int, height: int, fps: int, pixel_format: str
    ) -> bool:
        """Apply new camera configuration.

        Returns True if successful, False otherwise.
        Default implementation does nothing (config is fixed).
        """
        return False

    def supports_config_change(self) -> bool:
        """Return True if this camera supports runtime config changes."""
        return False


class RobotSensor(ABC):
    """Abstract interface for robot proprioception (legacy).

    Note: For full robot control with kinematics, control modes, etc.,
    use the rollio.robot module instead. This class remains for backward
    compatibility with simple proprioception-only use cases.
    """

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[float, dict[str, np.ndarray]]:
        """Return (timestamp_sec, {"position": arr, "velocity": arr, ...})."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def info(self) -> SensorInfo: ...

    @property
    @abstractmethod
    def num_joints(self) -> int: ...
