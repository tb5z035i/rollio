"""V4L2 Camera — real USB camera with format enumeration and live config."""
from __future__ import annotations

import re
import subprocess
from typing import Any

import cv2
import numpy as np

from rollio.sensors.base import (
    CameraFormat, CameraMode, CameraSettings, ImageSensor, SensorInfo,
)
from rollio.utils.time import monotonic_sec


def _parse_v4l2_formats(device: str | int) -> list[CameraFormat]:
    """Parse output of v4l2-ctl --list-formats-ext.

    Returns list of CameraFormat with available modes.
    """
    dev_path = f"/dev/video{device}" if isinstance(device, int) else device
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, "--list-formats-ext"],
            capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
    except Exception:
        return []

    formats: list[CameraFormat] = []
    current_format: CameraFormat | None = None

    # Parse patterns like:
    #   [0]: 'YUYV' (YUYV 4:2:2)
    #         Size: Discrete 640x480
    #             Interval: Discrete 0.033s (30.000 fps)
    format_re = re.compile(r"\[\d+\]:\s+'(\w+)'\s+\(([^)]+)\)")
    size_re = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    fps_re = re.compile(r"\((\d+(?:\.\d+)?)\s*fps\)")

    current_width = 0
    current_height = 0

    for line in result.stdout.splitlines():
        # Check for new format
        m = format_re.search(line)
        if m:
            if current_format:
                formats.append(current_format)
            current_format = CameraFormat(
                fourcc=m.group(1),
                description=m.group(2),
                modes=[])
            continue

        # Check for size
        m = size_re.search(line)
        if m:
            current_width = int(m.group(1))
            current_height = int(m.group(2))
            continue

        # Check for fps
        m = fps_re.search(line)
        if m and current_format and current_width > 0:
            fps = int(float(m.group(1)))
            current_format.modes.append(CameraMode(
                width=current_width,
                height=current_height,
                fps=fps))

    if current_format:
        formats.append(current_format)

    return formats


def probe_v4l2_formats(device: str | int) -> list[CameraFormat]:
    """Probe available formats for a V4L2 device.

    This is the public interface for format enumeration.
    """
    formats = _parse_v4l2_formats(device)
    if not formats:
        # Fallback: try to get current mode via OpenCV
        dev_idx = int(device) if isinstance(device, int) else int(
            device.replace("/dev/video", ""))
        try:
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                cap.release()
                formats = [CameraFormat(
                    fourcc="AUTO",
                    description="Auto-detected",
                    modes=[CameraMode(w, h, fps)])]
        except Exception:
            pass
    return formats


class V4L2Camera(ImageSensor):
    """V4L2 camera with format enumeration and runtime configuration."""

    # Common FourCC codes
    FOURCC_MAP = {
        "YUYV": cv2.VideoWriter_fourcc(*"YUYV"),
        "MJPG": cv2.VideoWriter_fourcc(*"MJPG"),
        "RGB3": cv2.VideoWriter_fourcc(*"RGB3"),
        "BGR3": cv2.VideoWriter_fourcc(*"BGR3"),
        "GREY": cv2.VideoWriter_fourcc(*"GREY"),
        "Y16 ": cv2.VideoWriter_fourcc(*"Y16 "),
    }

    def __init__(self, name: str, device: int | str = 0,
                 width: int = 640, height: int = 480, fps: int = 30,
                 pixel_format: str = "MJPG") -> None:
        self._name = name
        self._device = device
        self._device_idx = int(device) if isinstance(device, int) else int(
            str(device).replace("/dev/video", ""))
        self._width = width
        self._height = height
        self._fps = fps
        self._pixel_format = pixel_format
        self._cap: cv2.VideoCapture | None = None
        self._formats: list[CameraFormat] | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._device}")
        self._apply_settings()

    def _apply_settings(self) -> None:
        """Apply current width/height/fps/format to the capture device."""
        if self._cap is None:
            return

        # Set pixel format
        fourcc = self.FOURCC_MAP.get(self._pixel_format)
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Set FPS
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Read back actual values
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or self._fps

    def read(self) -> tuple[float, np.ndarray]:
        ts = monotonic_sec()
        if self._cap is None:
            return ts, np.zeros((self._height, self._width, 3), np.uint8)
        ret, frame = self._cap.read()
        if not ret:
            return ts, np.zeros((self._height, self._width, 3), np.uint8)
        return ts, frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def info(self) -> SensorInfo:
        return SensorInfo(
            name=self._name,
            sensor_type="camera",
            properties={
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "type": "v4l2",
                "device": self._device,
                "pixel_format": self._pixel_format,
            })

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    # ── Parameter probing interface ───────────────────────────────────

    def list_formats(self) -> list[CameraFormat]:
        if self._formats is None:
            self._formats = probe_v4l2_formats(self._device)
        return self._formats

    def get_config(self) -> CameraSettings:
        return CameraSettings(
            width=self._width,
            height=self._height,
            fps=self._fps,
            pixel_format=self._pixel_format)

    def apply_config(self, width: int, height: int, fps: int,
                     pixel_format: str) -> bool:
        """Apply new configuration. Camera must be open."""
        self._width = width
        self._height = height
        self._fps = fps
        self._pixel_format = pixel_format

        if self._cap is not None:
            self._apply_settings()
            return True
        return False

    def supports_config_change(self) -> bool:
        return True
