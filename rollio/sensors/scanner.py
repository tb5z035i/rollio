"""Hardware scanner — detect cameras and robots."""
from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Literal


@dataclass
class DetectedDevice:
    """A detected hardware device."""
    kind: Literal["camera", "robot"]
    dtype: str                       # "pseudo", "v4l2", "realsense", "airbot"…
    device_id: int | str             # index or path
    label: str                       # human-readable description
    properties: dict                 # width, height, fps, num_joints, etc.


def scan_cameras() -> list[DetectedDevice]:
    """Scan for available cameras."""
    found: list[DetectedDevice] = []

    # Always offer a pseudo camera
    found.append(DetectedDevice(
        kind="camera", dtype="pseudo", device_id=0,
        label="Pseudo Camera (test pattern)",
        properties={"width": 640, "height": 480, "fps": 30}))

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
                found.append(DetectedDevice(
                    kind="camera", dtype="v4l2", device_id=idx,
                    label=f"USB Camera {vdev} ({w}×{h}@{fps}fps)",
                    properties={"width": w, "height": h, "fps": fps}))
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
