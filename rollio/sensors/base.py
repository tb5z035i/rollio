"""Abstract base classes for all sensors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorInfo:
    """Metadata describing a sensor."""
    name: str
    sensor_type: str                          # "camera", "robot"
    properties: dict[str, Any] = field(default_factory=dict)


class ImageSensor(ABC):
    """Abstract interface for cameras."""

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


class RobotSensor(ABC):
    """Abstract interface for robot proprioception."""

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
