"""Intel RealSense Camera — multi-channel depth camera with RGB, depth, and IR streams."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rollio.sensors.base import (
    CameraChannel, CameraFormat, CameraMode, CameraSettings, ImageSensor, SensorInfo,
)
from rollio.utils.time import monotonic_sec

if TYPE_CHECKING:
    from rollio.sensors.scanner import DetectedDevice

# Lazy import pyrealsense2 to avoid import errors when not installed
rs = None


def _ensure_rs():
    """Lazy import pyrealsense2."""
    global rs
    if rs is None:
        import pyrealsense2 as _rs
        rs = _rs
    return rs


class RealSenseCamera(ImageSensor):
    """Intel RealSense camera with multi-channel support (color, depth, infrared).

    This camera can provide multiple synchronized streams:
    - color: RGB color image
    - depth: 16-bit depth map (z16 format, values in mm)
    - infrared: IR image (for stereo cameras like D4xx series)
    """

    SENSOR_TYPE = "realsense"

    # ── Factory / scanning class methods ──────────────────────────────

    @classmethod
    def scan(cls) -> list["DetectedDevice"]:
        """Scan for available RealSense devices."""
        from rollio.sensors.scanner import DetectedDevice

        found: list[DetectedDevice] = []
        try:
            rs = _ensure_rs()
            ctx = rs.context()
            for dev in ctx.query_devices():
                sn = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)

                # Enumerate available channels
                channels = []
                has_infrared = False

                for sensor in dev.query_sensors():
                    for profile in sensor.get_stream_profiles():
                        stream_type = profile.stream_type()
                        if stream_type == rs.stream.color and not any(
                                c.name == "color" for c in channels):
                            vp = profile.as_video_stream_profile()
                            channels.append(CameraChannel(
                                name="color",
                                default_width=vp.width(),
                                default_height=vp.height(),
                                default_fps=vp.fps(),
                                pixel_format="rgb24",
                                description="RGB Color stream"))
                        elif stream_type == rs.stream.depth and not any(
                                c.name == "depth" for c in channels):
                            vp = profile.as_video_stream_profile()
                            channels.append(CameraChannel(
                                name="depth",
                                default_width=vp.width(),
                                default_height=vp.height(),
                                default_fps=vp.fps(),
                                pixel_format="z16",
                                description="Depth stream (16-bit mm)"))
                        elif stream_type == rs.stream.infrared and not has_infrared:
                            vp = profile.as_video_stream_profile()
                            channels.append(CameraChannel(
                                name="infrared",
                                default_width=vp.width(),
                                default_height=vp.height(),
                                default_fps=vp.fps(),
                                pixel_format="y8",
                                description="Infrared stream"))
                            has_infrared = True

                # Get default resolution from color stream
                w, h, fps = 640, 480, 30
                for ch in channels:
                    if ch.name == "color":
                        w, h, fps = ch.default_width, ch.default_height, ch.default_fps
                        break

                found.append(DetectedDevice(
                    kind="camera",
                    dtype=cls.SENSOR_TYPE,
                    device_id=sn,
                    label=f"RealSense {name} (SN:{sn})",
                    properties={"width": w, "height": h, "fps": fps},
                    formats=cls.probe_formats(sn),
                    id_path="",
                    channels=channels))
        except Exception:
            pass
        return found

    @classmethod
    def probe_formats(cls, device_id: int | str) -> list[CameraFormat]:
        """Probe available formats for a RealSense device."""
        formats: list[CameraFormat] = []
        try:
            rs = _ensure_rs()
            ctx = rs.context()
            for dev in ctx.query_devices():
                sn = dev.get_info(rs.camera_info.serial_number)
                if str(sn) != str(device_id):
                    continue

                # Collect color stream formats
                color_modes: list[CameraMode] = []
                for sensor in dev.query_sensors():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type() == rs.stream.color:
                            vp = profile.as_video_stream_profile()
                            mode = CameraMode(vp.width(), vp.height(), vp.fps())
                            if mode not in color_modes:
                                color_modes.append(mode)

                if color_modes:
                    # Sort by resolution (largest first), then fps
                    color_modes.sort(
                        key=lambda m: (m.width * m.height, m.fps), reverse=True)
                    formats.append(CameraFormat(
                        fourcc="RGB",
                        description="RGB Color",
                        modes=color_modes))
                break
        except Exception:
            pass
        return formats

    @classmethod
    def get_channels(cls) -> list[CameraChannel]:
        """RealSense cameras typically have color, depth, and infrared channels."""
        return [
            CameraChannel(name="color", description="RGB Color stream"),
            CameraChannel(name="depth", pixel_format="z16",
                          description="Depth stream (16-bit mm)"),
            CameraChannel(name="infrared", pixel_format="y8",
                          description="Infrared stream"),
        ]

    # ── Instance methods ──────────────────────────────────────────────

    def __init__(self, name: str, device: str,
                 width: int = 640, height: int = 480, fps: int = 30,
                 enable_color: bool = True,
                 enable_depth: bool = True,
                 enable_infrared: bool = False) -> None:
        """Initialize RealSense camera.

        Args:
            name: Human-readable name for this camera
            device: Serial number of the RealSense device
            width: Resolution width for color stream
            height: Resolution height for color stream
            fps: Target framerate
            enable_color: Enable RGB color stream
            enable_depth: Enable depth stream
            enable_infrared: Enable infrared stream
        """
        self._name = name
        self._serial = str(device)
        self._width = width
        self._height = height
        self._fps = fps
        self._enable_color = enable_color
        self._enable_depth = enable_depth
        self._enable_infrared = enable_infrared

        self._pipeline = None
        self._config = None
        self._align = None  # For aligning depth to color
        self._last_frames: dict[str, np.ndarray] = {}

    def open(self) -> None:
        """Start the RealSense pipeline."""
        rs = _ensure_rs()

        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Enable device by serial number
        self._config.enable_device(self._serial)

        # Configure streams
        if self._enable_color:
            self._config.enable_stream(
                rs.stream.color, self._width, self._height,
                rs.format.bgr8, self._fps)

        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth, self._width, self._height,
                rs.format.z16, self._fps)

        if self._enable_infrared:
            self._config.enable_stream(
                rs.stream.infrared, 1,  # index 1 = left IR
                self._width, self._height,
                rs.format.y8, self._fps)

        # Start pipeline
        profile = self._pipeline.start(self._config)

        # Create align object for depth-to-color alignment
        if self._enable_color and self._enable_depth:
            self._align = rs.align(rs.stream.color)

        # Get actual resolution from profile
        if self._enable_color:
            color_profile = profile.get_stream(rs.stream.color)
            vp = color_profile.as_video_stream_profile()
            self._width = vp.width()
            self._height = vp.height()
            self._fps = vp.fps()

    def read(self) -> tuple[float, np.ndarray]:
        """Read frames from the camera.

        Returns the color frame as the primary output.
        Use read_all() to get all enabled streams.
        """
        ts = monotonic_sec()

        if self._pipeline is None:
            return ts, np.zeros((self._height, self._width, 3), np.uint8)

        try:
            rs = _ensure_rs()
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)

            # Align depth to color if both enabled
            if self._align is not None:
                frames = self._align.process(frames)

            # Extract frames
            if self._enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    self._last_frames["color"] = np.asanyarray(
                        color_frame.get_data())

            if self._enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    self._last_frames["depth"] = np.asanyarray(
                        depth_frame.get_data())

            if self._enable_infrared:
                ir_frame = frames.get_infrared_frame(1)
                if ir_frame:
                    self._last_frames["infrared"] = np.asanyarray(
                        ir_frame.get_data())

            # Return color frame as primary (or zeros if not enabled)
            if "color" in self._last_frames:
                return ts, self._last_frames["color"]
            else:
                return ts, np.zeros((self._height, self._width, 3), np.uint8)

        except Exception:
            return ts, np.zeros((self._height, self._width, 3), np.uint8)

    def read_all(self) -> tuple[float, dict[str, np.ndarray]]:
        """Read all enabled streams.

        Returns:
            Tuple of (timestamp, dict mapping channel name to frame)
        """
        ts, _ = self.read()  # This populates self._last_frames
        return ts, dict(self._last_frames)

    def read_depth(self) -> tuple[float, np.ndarray | None]:
        """Read depth frame (16-bit, values in mm)."""
        return monotonic_sec(), self._last_frames.get("depth")

    def read_infrared(self) -> tuple[float, np.ndarray | None]:
        """Read infrared frame."""
        return monotonic_sec(), self._last_frames.get("infrared")

    def close(self) -> None:
        """Stop the pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._config = None
        self._align = None
        self._last_frames.clear()

    def info(self) -> SensorInfo:
        return SensorInfo(
            name=self._name,
            sensor_type="camera",
            properties={
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "type": "realsense",
                "device": self._serial,
                "channels": {
                    "color": self._enable_color,
                    "depth": self._enable_depth,
                    "infrared": self._enable_infrared,
                },
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
    def serial(self) -> str:
        return self._serial

    # ── Parameter probing interface ───────────────────────────────────

    def list_formats(self) -> list[CameraFormat]:
        return self.probe_formats(self._serial)

    def get_config(self) -> CameraSettings:
        return CameraSettings(
            width=self._width,
            height=self._height,
            fps=self._fps,
            pixel_format="RGB")

    def apply_config(self, width: int, height: int, fps: int,
                     pixel_format: str) -> bool:
        """Apply new configuration. Requires reopening the pipeline."""
        was_open = self._pipeline is not None
        if was_open:
            self.close()

        self._width = width
        self._height = height
        self._fps = fps

        if was_open:
            self.open()
            return True
        return False

    def supports_config_change(self) -> bool:
        return True
