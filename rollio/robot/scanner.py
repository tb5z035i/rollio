"""Robot scanner — detect available robot entities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rollio.robot.base import RobotArm


@dataclass
class DetectedRobot:
    """A detected robot device."""
    robot_type: str              # "pseudo", "airbot_play", etc.
    device_id: int | str         # CAN interface, serial port, etc.
    label: str                   # Human-readable description
    n_dof: int                   # Degrees of freedom
    properties: dict = field(default_factory=dict)


# ─── Robot type registry ─────────────────────────────────────────────


def _get_robot_classes() -> list[type["RobotArm"]]:
    """Return robot classes that should be scanned directly."""
    from rollio.robot.pseudo_robot import PseudoRobotArm

    return [PseudoRobotArm]


def scan_robots() -> list[DetectedRobot]:
    """Scan for available robots using registered robot classes.

    Each robot class implements its own scan() method.
    This function aggregates results from all registered types.
    """
    found: list[DetectedRobot] = []

    for robot_cls in _get_robot_classes():
        try:
            devices = robot_cls.scan()
            found.extend(devices)
        except Exception:
            pass

    try:
        from rollio.robot.airbot.shared import scan_airbot_detected_robots

        found.extend(scan_airbot_detected_robots())
    except ImportError:
        pass

    return found
