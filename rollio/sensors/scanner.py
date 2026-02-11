"""Hardware scanner — detect cameras and robots."""
from __future__ import annotations

import glob
import subprocess
from dataclasses import dataclass, field
from typing import Literal

from rollio.sensors.base import CameraFormat


@dataclass
class DetectedDevice:
    """A detected hardware device."""
    kind: Literal["camera", "robot"]
    dtype: str                       # "pseudo", "v4l2", "realsense", "airbot"…
    device_id: int | str             # index or path
    label: str                       # human-readable description
    properties: dict                 # width, height, fps, num_joints, etc.
    formats: list[CameraFormat] = field(default_factory=list)  # for cameras
    id_path: str = ""                # udev ID_PATH for stable identification


def _get_udev_id_path(device: str) -> str:
    """Get ID_PATH from udevadm for a device (e.g., /dev/video0).

    Returns empty string if unavailable.
    """
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("ID_PATH="):
                    return line.split("=", 1)[1]
    except Exception:
        pass
    return ""


def scan_cameras() -> list[DetectedDevice]:
    """Scan for available cameras."""
    from rollio.sensors.v4l2_camera import probe_v4l2_formats

    found: list[DetectedDevice] = []

    # Always offer a pseudo camera
    found.append(DetectedDevice(
        kind="camera", dtype="pseudo", device_id=0,
        label="Pseudo Camera (test pattern)",
        properties={"width": 640, "height": 480, "fps": 30},
        formats=[]))

    # Probe v4l2 devices
    for vdev in sorted(glob.glob("/dev/video*")):
        try:
            import cv2
            idx = int(vdev.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                cap.release()

                # Probe available formats
                formats = probe_v4l2_formats(idx)

                # Get udev ID_PATH for stable identification
                id_path = _get_udev_id_path(vdev)

                found.append(DetectedDevice(
                    kind="camera", dtype="v4l2", device_id=idx,
                    label=f"USB Camera {vdev} ({w}×{h}@{fps}fps)",
                    properties={"width": w, "height": h, "fps": fps},
                    formats=formats,
                    id_path=id_path))
            else:
                cap.release()
        except Exception:
            pass

    # Probe RealSense
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        for dev in ctx.query_devices():
            sn = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            found.append(DetectedDevice(
                kind="camera", dtype="realsense", device_id=sn,
                label=f"RealSense {name} (SN:{sn})",
                properties={"width": 640, "height": 480, "fps": 30}))
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
