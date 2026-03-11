"""V4L2 Camera — real USB camera with format enumeration and live config."""

from __future__ import annotations

import glob
import re
import subprocess
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rollio.sensors.base import (
    CameraChannel,
    CameraFormat,
    CameraMode,
    CameraSettings,
    ImageSensor,
    SensorInfo,
)
from rollio.utils.time import monotonic_sec

if TYPE_CHECKING:
    from rollio.sensors.scanner import DetectedDevice


def _parse_v4l2_formats(device: str | int) -> list[CameraFormat]:
    """Parse output of v4l2-ctl --list-formats-ext.

    Returns list of CameraFormat with available modes.
    """
    dev_path = f"/dev/video{device}" if isinstance(device, int) else device
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
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
                fourcc=m.group(1), description=m.group(2), modes=[]
            )
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
            current_format.modes.append(
                CameraMode(width=current_width, height=current_height, fps=fps)
            )

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
        dev_idx = (
            int(device)
            if isinstance(device, int)
            else int(device.replace("/dev/video", ""))
        )
        try:
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                cap.release()
                formats = [
                    CameraFormat(
                        fourcc="AUTO",
                        description="Auto-detected",
                        modes=[CameraMode(w, h, fps)],
                    )
                ]
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
            pass
    return formats


def _get_udev_properties(device: str) -> dict[str, str]:
    """Get udev properties for a device (e.g., /dev/video0)."""
    props = {}
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    props[key] = value
    except (OSError, subprocess.SubprocessError):
        pass
    return props


def _get_udev_id_path(device: str) -> str:
    """Get ID_PATH from udevadm for a device (e.g., /dev/video0)."""
    return _get_udev_properties(device).get("ID_PATH", "")


def _is_realsense_device(device: str) -> bool:
    """Check if a /dev/video* device belongs to an Intel RealSense camera.

    RealSense cameras appear as V4L2 devices but should be handled by the
    RealSense SDK instead for proper multi-stream support.
    """
    props = _get_udev_properties(device)

    # Check vendor ID (Intel = 8086)
    if props.get("ID_VENDOR_ID") == "8086":
        return True

    # Check model/product name for RealSense
    model = props.get("ID_MODEL", "").lower()
    product = props.get("ID_PRODUCT", "").lower()
    vendor = props.get("ID_VENDOR", "").lower()

    if "realsense" in model or "realsense" in product:
        return True
    if "intel" in vendor and (
        "d4" in model or "d5" in model or "l5" in model or "sr3" in model
    ):
        # D4xx, D5xx, L5xx, SR3xx series
        return True

    return False


class V4L2Camera(ImageSensor):
    """V4L2 camera with format enumeration and runtime configuration."""

    SENSOR_TYPE = "v4l2"

    # Common FourCC codes
    FOURCC_MAP = {
        "YUYV": cv2.VideoWriter_fourcc(*"YUYV"),
        "MJPG": cv2.VideoWriter_fourcc(*"MJPG"),
        "RGB3": cv2.VideoWriter_fourcc(*"RGB3"),
        "BGR3": cv2.VideoWriter_fourcc(*"BGR3"),
        "GREY": cv2.VideoWriter_fourcc(*"GREY"),
        "Y16 ": cv2.VideoWriter_fourcc(*"Y16 "),
    }

    # ── Factory / scanning class methods ──────────────────────────────

    @classmethod
    def scan(cls) -> list["DetectedDevice"]:
        """Scan for available V4L2 camera devices.

        Skips devices that belong to Intel RealSense cameras, as those
        should be handled by the RealSense SDK for proper multi-stream support.
        """
        from rollio.sensors.scanner import DetectedDevice

        found: list[DetectedDevice] = []
        for vdev in sorted(glob.glob("/dev/video*")):
            try:
                # Skip RealSense devices - they have their own handler
                if _is_realsense_device(vdev):
                    continue

                idx = int(vdev.replace("/dev/video", ""))
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    cap.release()

                    formats = probe_v4l2_formats(idx)
                    id_path = _get_udev_id_path(vdev)

                    # Determine default pixel format (prefer MJPG if available)
                    pix_fmt = "MJPG"
                    if formats:
                        pix_fmt = formats[0].fourcc
                        for fmt in formats:
                            if fmt.fourcc == "MJPG":
                                pix_fmt = "MJPG"
                                break

                    found.append(
                        DetectedDevice(
                            kind="camera",
                            dtype=cls.SENSOR_TYPE,
                            device_id=idx,
                            label=f"USB Camera {vdev} ({w}×{h}@{fps}fps)",
                            properties={},
                            formats=formats,
                            id_path=id_path,
                            channels=[
                                CameraChannel(
                                    name="color",
                                    default_width=w,
                                    default_height=h,
                                    default_fps=fps,
                                    description="RGB Color stream",
                                )
                            ],
                            width=w,
                            height=h,
                            fps=fps,
                            pixel_format=pix_fmt,
                        )
                    )
                else:
                    cap.release()
            except (OSError, ValueError, RuntimeError):
                pass
        return found

    @classmethod
    def probe_formats(cls, device_id: int | str) -> list[CameraFormat]:
        """Probe available formats for a V4L2 device."""
        return probe_v4l2_formats(device_id)

    @classmethod
    def get_channels(cls) -> list[CameraChannel]:
        """V4L2 cameras have a single color channel."""
        return [CameraChannel(name="color", description="RGB Color stream")]

    # ── Instance methods ──────────────────────────────────────────────

    def __init__(
        self,
        name: str,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pixel_format: str = "MJPG",
    ) -> None:
        self._name = name
        self._device = device
        self._device_idx = (
            int(device)
            if isinstance(device, int)
            else int(str(device).replace("/dev/video", ""))
        )
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
            },
        )

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
            pixel_format=self._pixel_format,
        )

    def apply_config(
        self, width: int, height: int, fps: int, pixel_format: str
    ) -> bool:
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
