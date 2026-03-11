"""Pseudo robot — 6-DOF random-walk joint trajectories."""

from __future__ import annotations

import math

import numpy as np

from rollio.sensors.base import RobotSensor, SensorInfo
from rollio.utils.time import monotonic_sec


class PseudoRobot(RobotSensor):
    """Fake robot that generates smooth sinusoidal joint trajectories.

    Produces position, velocity, and effort arrays that look plausible
    for a 6-DOF arm (values in radians / rad-s / Nm-like units).
    """

    def __init__(
        self, name: str = "pseudo_arm", n_joints: int = 6, role: str = "follower"
    ) -> None:
        self._name = name
        self._n = n_joints
        self._role = role
        self._open = False
        self._t0 = 0.0

        # Random per-joint parameters for variety
        rng = np.random.default_rng(hash(name) % (2**31))
        self._freq = rng.uniform(0.1, 0.6, size=n_joints)  # Hz
        self._amp = rng.uniform(0.3, 1.2, size=n_joints)  # rad
        self._phase = rng.uniform(0, 2 * math.pi, size=n_joints)
        self._offset = rng.uniform(-0.5, 0.5, size=n_joints)

    # ── interface ──────────────────────────────────────────────────

    def open(self) -> None:
        self._t0 = monotonic_sec()
        self._open = True

    def read(self) -> tuple[float, dict[str, np.ndarray]]:
        ts = monotonic_sec()
        t = ts - self._t0
        pos = self._offset + self._amp * np.sin(
            2 * math.pi * self._freq * t + self._phase
        )
        vel = (
            self._amp
            * 2
            * math.pi
            * self._freq
            * np.cos(2 * math.pi * self._freq * t + self._phase)
        )
        eff = -0.5 * pos + 0.1 * np.sin(t * 3.0)  # fake effort
        return ts, {
            "position": pos.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "effort": eff.astype(np.float32),
        }

    def close(self) -> None:
        self._open = False

    def info(self) -> SensorInfo:
        return SensorInfo(
            name=self._name,
            sensor_type="robot",
            properties={"num_joints": self._n, "type": "pseudo", "role": self._role},
        )

    @property
    def num_joints(self) -> int:
        return self._n
