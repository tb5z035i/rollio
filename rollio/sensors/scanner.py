"""Hardware scanner — detect cameras and robots."""
from __future__ import annotations

from copy import deepcopy
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


def _build_pseudo_camera_devices(count: int) -> list[DetectedDevice]:
    """Create one DetectedDevice entry per requested pseudo camera."""
    from rollio.sensors.pseudo_camera import PseudoCamera

    if count <= 0:
        return []

    base_devices = PseudoCamera.scan()
    if not base_devices:
        return []

    base = base_devices[0]
    devices: list[DetectedDevice] = []
    for idx in range(count):
        devices.append(DetectedDevice(
            kind=base.kind,
            dtype=base.dtype,
            device_id=idx,
            label=f"Pseudo Camera {idx + 1} (test pattern)",
            properties=deepcopy(base.properties),
            formats=deepcopy(base.formats),
            id_path=base.id_path,
            channels=deepcopy(base.channels),
            width=base.width,
            height=base.height,
            fps=base.fps,
            pixel_format=base.pixel_format,
        ))
    return devices


def scan_cameras(
    *,
    include_simulated: bool = False,
    simulated_count: int = 0,
) -> list[DetectedDevice]:
    """Scan for available cameras using registered sensor classes.

    Each camera sensor class implements its own scan() method.
    This function aggregates results from all registered types.
    """
    found: list[DetectedDevice] = []

    for camera_cls in _get_camera_classes():
        if camera_cls.SENSOR_TYPE == "pseudo":
            continue
        try:
            devices = camera_cls.scan()
            found.extend(devices)
        except Exception:
            pass

    if include_simulated:
        found.extend(_build_pseudo_camera_devices(simulated_count))

    return found


def _build_pseudo_robot_devices(count: int) -> list[DetectedDevice]:
    """Create one DetectedDevice entry per requested pseudo robot."""
    if count <= 0:
        return []

    devices: list[DetectedDevice] = []
    for idx in range(count):
        devices.append(DetectedDevice(
            kind="robot",
            dtype="pseudo",
            device_id=idx,
            label=f"Pseudo Robot {idx + 1} (6-DOF simulation)",
            properties={"num_joints": 6, "simulated": True},
        ))
    return devices


def scan_robots(
    *,
    include_simulated: bool = False,
    simulated_count: int = 0,
) -> list[DetectedDevice]:
    """Scan for available robots.
    
    Scans for AIRBOT Play robots via CAN bus. Simulated pseudo robots are
    only included when explicitly requested.
    
    Note: For full robot control capabilities, use rollio.robot module directly.
    """
    found: list[DetectedDevice] = []

    # Scan for AIRBOT Play robots
    try:
        from rollio.robot import scan_robots as robot_scan_robots
        for robot in robot_scan_robots():
            if robot.robot_type == "pseudo":
                continue
            found.append(DetectedDevice(
                kind="robot",
                dtype=robot.robot_type,
                device_id=robot.device_id,
                label=robot.label,
                properties={
                    "num_joints": robot.n_dof,
                    "can_interface": robot.properties.get("can_interface"),
                    **robot.properties,
                }
            ))
    except ImportError:
        pass

    if include_simulated:
        found.extend(_build_pseudo_robot_devices(simulated_count))

    return found
