"""Intel RealSense Camera — multi-channel depth camera with RGB, depth, and IR streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        """Scan for available RealSense devices.

        Each channel (color, depth, infrared) is exposed as a separate camera
        to simplify the setup wizard.
        """
        from rollio.sensors.scanner import DetectedDevice

        found: list[DetectedDevice] = []
        try:
            rs = _ensure_rs()
            ctx = rs.context()
            for dev in ctx.query_devices():
                sn = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)

                # Collect channel info: (channel_name, width, height, fps, pixel_format)
                # We need to find compatible resolution/format combinations
                channel_info: dict[str, tuple[int, int, int, str]] = {}

                for sensor in dev.query_sensors():
                    for profile in sensor.get_stream_profiles():
                        stream_type = profile.stream_type()
                        vp = profile.as_video_stream_profile()
                        fmt = str(vp.format()).replace("format.", "")

                        # Color - prefer bgr8 format
                        if (
                            stream_type == rs.stream.color
                            and "color" not in channel_info
                        ):
                            if fmt == "bgr8":
                                channel_info["color"] = (
                                    vp.width(),
                                    vp.height(),
                                    vp.fps(),
                                    "bgr8",
                                )
                        # Depth - prefer z16 format
                        elif (
                            stream_type == rs.stream.depth
                            and "depth" not in channel_info
                        ):
                            if fmt == "z16":
                                channel_info["depth"] = (
                                    vp.width(),
                                    vp.height(),
                                    vp.fps(),
                                    "z16",
                                )
                        # Infrared - prefer y8 format (8-bit grayscale)
                        elif (
                            stream_type == rs.stream.infrared
                            and "infrared" not in channel_info
                        ):
                            if fmt == "y8":
                                channel_info["infrared"] = (
                                    vp.width(),
                                    vp.height(),
                                    vp.fps(),
                                    "y8",
                                )

                # Create a separate DetectedDevice for each channel
                for ch_name, (w, h, fps, pf) in channel_info.items():
                    # Device ID encodes serial:channel for unique identification
                    device_id = f"{sn}:{ch_name}"

                    # Probe formats for this specific channel
                    formats = cls.probe_channel_formats(sn, ch_name)

                    # Channel type for display
                    ch_label = {"color": "RGB", "depth": "Depth", "infrared": "IR"}[
                        ch_name
                    ]

                    found.append(
                        DetectedDevice(
                            kind="camera",
                            dtype=f"realsense_{ch_name}",  # e.g., "realsense_color"
                            device_id=device_id,
                            label=f"RealSense {ch_label} - {name} (SN:{sn})",
                            properties={
                                "channel": ch_name,
                                "serial": sn,
                            },
                            formats=formats,
                            id_path="",
                            channels=[
                                CameraChannel(
                                    name=ch_name,
                                    default_width=w,
                                    default_height=h,
                                    default_fps=fps,
                                    pixel_format=pf,
                                )
                            ],
                            width=w,
                            height=h,
                            fps=fps,
                            pixel_format=pf,
                        )
                    )
        except Exception:
            pass
        return found

    @classmethod
    def probe_formats(cls, device_id: int | str) -> list[CameraFormat]:
        """Probe available formats for all channels of a RealSense device.

        Returns formats for color, depth, and infrared channels.
        """
        return cls.probe_channel_formats(device_id, channel=None)

    @classmethod
    def probe_channel_formats(
        cls, device_id: int | str, channel: str | None = None
    ) -> list[CameraFormat]:
        """Probe available formats for a specific channel or all channels.

        Parameters
        ----------
        device_id : int | str
            RealSense serial number.
        channel : str | None
            Channel name ("color", "depth", "infrared") or None for all.

        Returns list of CameraFormat for the requested channel(s).
        """
        formats: list[CameraFormat] = []
        try:
            rs = _ensure_rs()
            ctx = rs.context()
            for dev in ctx.query_devices():
                sn = dev.get_info(rs.camera_info.serial_number)
                if str(sn) != str(device_id):
                    continue

                # Map channel names to stream types
                stream_map = {
                    "color": rs.stream.color,
                    "depth": rs.stream.depth,
                    "infrared": rs.stream.infrared,
                }

                channels_to_probe = (
                    [channel] if channel else ["color", "depth", "infrared"]
                )

                for ch_name in channels_to_probe:
                    stream_type = stream_map.get(ch_name)
                    if stream_type is None:
                        continue

                    # Collect modes and formats for this channel
                    format_modes: dict[str, list[CameraMode]] = {}

                    for sensor in dev.query_sensors():
                        for profile in sensor.get_stream_profiles():
                            if profile.stream_type() != stream_type:
                                continue
                            vp = profile.as_video_stream_profile()
                            fmt_name = str(vp.format()).replace("format.", "")
                            mode = CameraMode(vp.width(), vp.height(), vp.fps())

                            if fmt_name not in format_modes:
                                format_modes[fmt_name] = []
                            if mode not in format_modes[fmt_name]:
                                format_modes[fmt_name].append(mode)

                    # Create CameraFormat for each pixel format
                    for fmt_name, modes in format_modes.items():
                        modes.sort(
                            key=lambda m: (m.width * m.height, m.fps), reverse=True
                        )
                        # Use channel name in description for clarity
                        desc = f"{ch_name.capitalize()} ({fmt_name})"
                        formats.append(
                            CameraFormat(fourcc=fmt_name, description=desc, modes=modes)
                        )
                break
        except Exception:
            pass
        return formats

    @classmethod
    def get_channels(cls) -> list[CameraChannel]:
        """RealSense cameras typically have color, depth, and infrared channels."""
        return [
            CameraChannel(name="color", description="RGB Color stream"),
            CameraChannel(
                name="depth", pixel_format="z16", description="Depth stream (16-bit mm)"
            ),
            CameraChannel(
                name="infrared", pixel_format="y8", description="Infrared stream"
            ),
        ]

    # ── Instance methods ──────────────────────────────────────────────

    def __init__(
        self,
        name: str,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
        enable_infrared: bool = False,
        depth_width: int | None = None,
        depth_height: int | None = None,
        depth_fps: int | None = None,
        ir_width: int | None = None,
        ir_height: int | None = None,
        ir_fps: int | None = None,
        depth_format: str = "z16",
        ir_format: str = "y8",
        preview_channel: str = "color",
    ) -> None:
        """Initialize RealSense camera.

        Args:
            name: Human-readable name for this camera
            device: Serial number of the RealSense device
            width: Resolution width for color stream
            height: Resolution height for color stream
            fps: Target framerate for color stream
            enable_color: Enable RGB color stream
            enable_depth: Enable depth stream
            enable_infrared: Enable infrared stream
            depth_width/height/fps: Override resolution for depth stream
            ir_width/height/fps: Override resolution for infrared stream
            depth_format: Pixel format for depth ("z16")
            ir_format: Pixel format for infrared ("y8" or "y16")
            preview_channel: Which channel to return from read() ("color", "depth", "infrared")
        """
        self._name = name
        self._serial = str(device)
        self._width = width
        self._height = height
        self._fps = fps
        self._enable_color = enable_color
        self._enable_depth = enable_depth
        self._enable_infrared = enable_infrared

        # Per-channel resolution (defaults to main resolution)
        self._depth_width = depth_width or width
        self._depth_height = depth_height or height
        self._depth_fps = depth_fps or fps
        self._ir_width = ir_width or width
        self._ir_height = ir_height or height
        self._ir_fps = ir_fps or fps

        # Per-channel pixel format
        self._depth_format = depth_format
        self._ir_format = ir_format

        self._preview_channel = preview_channel
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

        # Map format strings to rs.format enum
        format_map = {
            "bgr8": rs.format.bgr8,
            "rgb8": rs.format.rgb8,
            "z16": rs.format.z16,
            "y8": rs.format.y8,
            "y16": rs.format.y16,
        }

        # Configure streams with per-channel resolution
        if self._enable_color:
            self._config.enable_stream(
                rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
            )

        if self._enable_depth:
            depth_fmt = format_map.get(self._depth_format, rs.format.z16)
            self._config.enable_stream(
                rs.stream.depth,
                self._depth_width,
                self._depth_height,
                depth_fmt,
                self._depth_fps,
            )

        if self._enable_infrared:
            ir_fmt = format_map.get(self._ir_format, rs.format.y8)
            self._config.enable_stream(
                rs.stream.infrared,
                1,  # index 1 = left IR
                self._ir_width,
                self._ir_height,
                ir_fmt,
                self._ir_fps,
            )

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

        Returns the frame for the current preview channel.
        Use read_all() to get all enabled streams.
        """
        ts = monotonic_sec()

        if self._pipeline is None:
            return ts, self._empty_frame()

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
                    self._last_frames["color"] = np.asanyarray(color_frame.get_data())

            if self._enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    self._last_frames["depth"] = np.asanyarray(depth_frame.get_data())

            if self._enable_infrared:
                ir_frame = frames.get_infrared_frame(1)
                if ir_frame:
                    self._last_frames["infrared"] = np.asanyarray(ir_frame.get_data())

            # Return the selected preview channel
            return ts, self._get_preview_frame()

        except Exception:
            return ts, self._empty_frame()

    def _empty_frame(self) -> np.ndarray:
        """Return an empty frame for the current preview channel."""
        ch = self._preview_channel
        if ch == "depth":
            return np.zeros((self._depth_height, self._depth_width), np.uint16)
        elif ch == "infrared":
            # Use uint16 for y16 format, uint8 for y8
            dtype = np.uint16 if self._ir_format == "y16" else np.uint8
            return np.zeros((self._ir_height, self._ir_width), dtype)
        else:
            return np.zeros((self._height, self._width, 3), np.uint8)

    def _get_preview_frame(self) -> np.ndarray:
        """Get the frame for the current preview channel."""
        ch = self._preview_channel
        if ch in self._last_frames:
            return self._last_frames[ch]
        # Fallback to color or first available
        for fallback in ["color", "depth", "infrared"]:
            if fallback in self._last_frames:
                return self._last_frames[fallback]
        return self._empty_frame()

    @property
    def preview_channel(self) -> str:
        """Current preview channel name."""
        return self._preview_channel

    @preview_channel.setter
    def preview_channel(self, value: str) -> None:
        """Set the preview channel."""
        if value in ("color", "depth", "infrared"):
            self._preview_channel = value

    def get_channel_resolution(self, channel: str) -> tuple[int, int, int]:
        """Get (width, height, fps) for a specific channel."""
        if channel == "color":
            return self._width, self._height, self._fps
        elif channel == "depth":
            return self._depth_width, self._depth_height, self._depth_fps
        elif channel == "infrared":
            return self._ir_width, self._ir_height, self._ir_fps
        return 0, 0, 0

    def is_channel_enabled(self, channel: str) -> bool:
        """Check if a channel is enabled."""
        if channel == "color":
            return self._enable_color
        elif channel == "depth":
            return self._enable_depth
        elif channel == "infrared":
            return self._enable_infrared
        return False

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
    def serial(self) -> str:
        return self._serial

    # ── Parameter probing interface ───────────────────────────────────

    def list_formats(self) -> list[CameraFormat]:
        return self.probe_formats(self._serial)

    def get_config(self) -> CameraSettings:
        return CameraSettings(
            width=self._width, height=self._height, fps=self._fps, pixel_format="RGB"
        )

    def apply_config(
        self, width: int, height: int, fps: int, pixel_format: str
    ) -> bool:
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
