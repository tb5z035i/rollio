"""Hardware scanner — detect cameras and robots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rollio.sensors.base import CameraChannel, CameraFormat

if TYPE_CHECKING:
    from rollio.sensors.base import ImageSensor


@dataclass
class DetectedDevice:
    """A detected hardware device."""
    kind: Literal["camera", "robot"]
    dtype: str                       # "pseudo", "v4l2", "realsense_color", etc.
    device_id: int | str             # index or path
    label: str                       # human-readable description
    properties: dict                 # extra properties (num_joints, etc.)
    formats: list[CameraFormat] = field(default_factory=list)  # for cameras
    id_path: str = ""                # udev ID_PATH for stable identification
    channels: list[CameraChannel] = field(default_factory=list)  # multi-channel
    # Common camera properties (extracted from properties for convenience)
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "RGB"        # MJPG, YUYV, z16, y8, y16, bgr8, etc.


# ─── Camera type registry ─────────────────────────────────────────────

# List of camera sensor classes to scan. Import lazily to avoid circular deps.
def _get_camera_classes() -> list[type["ImageSensor"]]:
    """Return all registered camera sensor classes."""
    from rollio.sensors.pseudo_camera import PseudoCamera
    from rollio.sensors.realsense_camera import RealSenseCamera
    from rollio.sensors.v4l2_camera import V4L2Camera

    return [PseudoCamera, V4L2Camera, RealSenseCamera]


def scan_cameras() -> list[DetectedDevice]:
    """Scan for available cameras using registered sensor classes.

    Each camera sensor class implements its own scan() method.
    This function aggregates results from all registered types.
    """
    found: list[DetectedDevice] = []

    for camera_cls in _get_camera_classes():
        try:
            devices = camera_cls.scan()
            found.extend(devices)
        except Exception:
            pass

    return found


def scan_robots() -> list[DetectedDevice]:
    """Scan for available robots."""
    found: list[DetectedDevice] = []

    # Always offer a pseudo robot
    found.append(DetectedDevice(
        kind="robot", dtype="pseudo", device_id=0,
        label="Pseudo Robot (6-DOF sine wave)",
        properties={"num_joints": 6}))

    # Future: scan CAN bus for AIRBOT, USB serial for NERO, etc.

    return found
