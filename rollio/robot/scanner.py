"""Robot scanner — detect available robot entities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectedRobot:
    """A detected robot device."""

    robot_type: str  # "pseudo", "airbot_play", etc.
    device_id: int | str  # CAN interface, serial port, etc.
    label: str  # Human-readable description
    n_dof: int  # Degrees of freedom
    properties: dict = field(default_factory=dict)


def _build_pseudo_detected_robots(count: int) -> list[DetectedRobot]:
    if count <= 0:
        return []
    return [
        DetectedRobot(
            robot_type="pseudo",
            device_id=idx,
            label=f"Pseudo Robot {idx + 1} (6-DOF simulation)",
            n_dof=6,
            properties={"num_joints": 6, "simulated": True},
        )
        for idx in range(count)
    ]


def scan_robots(
    *,
    include_simulated: bool = False,
    simulated_count: int = 0,
) -> list[DetectedRobot]:
    """Scan for available robots using registered robot classes.

    Hardware detection is always attempted. Simulated pseudo robots are only
    included when explicitly requested.
    """
    found: list[DetectedRobot] = []

    try:
        from rollio.robot.airbot.shared import scan_airbot_detected_robots

        found.extend(scan_airbot_detected_robots())
    except ImportError:
        pass

    if include_simulated:
        found.extend(_build_pseudo_detected_robots(simulated_count))

    return found
