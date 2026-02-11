"""Pseudo camera — generates frames with an incrementing frame counter."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rollio.sensors.base import (
    CameraChannel, CameraFormat, CameraMode, ImageSensor, SensorInfo,
)
from rollio.utils.time import monotonic_sec

if TYPE_CHECKING:
    from rollio.sensors.scanner import DetectedDevice


class PseudoCamera(ImageSensor):
    """Fake camera that produces colourful gradient frames with a visible
    frame number overlay.  Useful for end-to-end pipeline testing without
    real hardware.
    """

    SENSOR_TYPE = "pseudo"

    # ── Factory / scanning class methods ──────────────────────────────

    @classmethod
    def scan(cls) -> list["DetectedDevice"]:
        """Return a single pseudo camera device."""
        from rollio.sensors.scanner import DetectedDevice

        return [DetectedDevice(
            kind="camera",
            dtype=cls.SENSOR_TYPE,
            device_id=0,
            label="Pseudo Camera (test pattern)",
            properties={"width": 640, "height": 480, "fps": 30},
            formats=[CameraFormat(
                fourcc="RGB",
                description="RGB24",
                modes=[
                    CameraMode(640, 480, 30),
                    CameraMode(320, 240, 30),
                    CameraMode(1280, 720, 30),
                ])],
            id_path="",
            channels=[CameraChannel(
                name="color",
                default_width=640,
                default_height=480,
                default_fps=30,
                description="Synthetic RGB pattern")])]

    @classmethod
    def probe_formats(cls, device_id: int | str) -> list[CameraFormat]:
        """Return pseudo formats (any resolution is supported)."""
        return [CameraFormat(
            fourcc="RGB",
            description="RGB24",
            modes=[
                CameraMode(640, 480, 30),
                CameraMode(320, 240, 30),
                CameraMode(1280, 720, 30),
            ])]

    @classmethod
    def get_channels(cls) -> list[CameraChannel]:
        """Pseudo cameras have a single color channel."""
        return [CameraChannel(name="color", description="Synthetic RGB pattern")]

    # ── Instance methods ──────────────────────────────────────────────

    def __init__(self, name: str = "pseudo_cam",
                 width: int = 640, height: int = 480,
                 fps: int = 30) -> None:
        self._name = name
        self._w = width
        self._h = height
        self._fps = fps
        self._frame_idx = 0
        self._open = False

        # Pre-compute static gradient (re-coloured each frame)
        ys = np.linspace(0, 1, height, dtype=np.float32)
        xs = np.linspace(0, 1, width, dtype=np.float32)
        self._grid_y, self._grid_x = np.meshgrid(ys, xs, indexing="ij")

    # ── interface ──────────────────────────────────────────────────

    def open(self) -> None:
        self._frame_idx = 0
        self._open = True

    def read(self) -> tuple[float, np.ndarray]:
        ts = monotonic_sec()
        frame = self._generate()
        self._frame_idx += 1
        return ts, frame

    def close(self) -> None:
        self._open = False

    def info(self) -> SensorInfo:
        return SensorInfo(
            name=self._name, sensor_type="camera",
            properties={"width": self._w, "height": self._h,
                        "fps": self._fps, "type": "pseudo"})

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def fps(self) -> int:
        return self._fps

    # ── frame generation ──────────────────────────────────────────

    def _generate(self) -> np.ndarray:
        n = self._frame_idx
        t = n / max(self._fps, 1)

        # Slowly shifting colour gradient
        r = ((self._grid_y + math.sin(t * 0.3) * 0.3) * 200 + 30)
        g = ((self._grid_x + math.cos(t * 0.5) * 0.3) * 180 + 40)
        b = (((self._grid_x + self._grid_y) * 0.5
              + math.sin(t * 0.7) * 0.3) * 200 + 30)

        frame = np.stack([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8),
        ], axis=2)

        # Overlay frame number
        text = f"#{n:06d}"
        org = (self._w // 2 - 120, self._h // 2 + 20)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Small label
        cv2.putText(frame, self._name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return frame
