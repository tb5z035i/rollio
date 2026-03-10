"""Shared AIRBOT helpers for discovery and SDK imports."""
from __future__ import annotations

from rollio.robot.airbot.can import probe_airbot_device, query_airbot_properties
from rollio.robot.can_utils import is_can_interface_up, scan_can_interfaces
from rollio.robot.scanner import DetectedRobot

_ah = None
_AH_AVAILABLE: bool | None = None

AIRBOT_ARM_ROBOT_TYPE = "airbot_play"
AIRBOT_EEF_TYPE_TO_ROBOT_TYPE = {
    "E2B": "airbot_e2b",
    "G2": "airbot_g2",
}
AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF = {
    robot_type: eef_type
    for eef_type, robot_type in AIRBOT_EEF_TYPE_TO_ROBOT_TYPE.items()
}
AIRBOT_ROBOT_TYPE_TO_SDK_EEF = {
    "airbot_e2b": "E2",
    "airbot_g2": "G2",
}
AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR = {
    "airbot_e2b": "OD",
    "airbot_g2": "DM",
}


def _import_airbot_hardware():
    """Lazy import ``airbot_hardware_py``."""
    global _ah, _AH_AVAILABLE
    if _AH_AVAILABLE is None:
        try:
            import airbot_hardware_py as ah

            _ah = ah
            _AH_AVAILABLE = True
        except ImportError:
            _AH_AVAILABLE = False
    return _ah, bool(_AH_AVAILABLE)


def is_airbot_available() -> bool:
    """Check if ``airbot_hardware_py`` is available."""
    _, available = _import_airbot_hardware()
    return available


def normalize_airbot_eef_type(eef_type: str | None) -> str:
    """Normalize AIRBOT-reported end-effector names."""
    if eef_type is None:
        return "none"
    normalized = str(eef_type).strip().lower()
    if normalized in {"", "none", "na"}:
        return "none"
    if normalized in {"e2", "e2b"}:
        return "E2B"
    if normalized == "g2":
        return "G2"
    return str(eef_type).strip()


def _airbot_label(prefix: str, interface: str, serial_number: str | None) -> str:
    label = f"{prefix} ({interface})"
    if serial_number:
        label = f"{label} SN:{serial_number}"
    return label


def scan_airbot_detected_robots() -> list[DetectedRobot]:
    """Scan AIRBOT arms plus attached end-effectors once per interface."""
    if not is_airbot_available():
        return []

    found: list[DetectedRobot] = []
    for interface in scan_can_interfaces():
        if not is_can_interface_up(interface):
            continue
        if not probe_airbot_device(interface, timeout=0.5):
            continue

        properties = query_airbot_properties(interface, timeout=0.5)
        properties["can_interface"] = interface
        properties["motor_types"] = ["OD", "OD", "OD", "DM", "DM", "DM"]

        serial_number = properties.get("serial_number")
        arm_props = dict(properties)
        found.append(
            DetectedRobot(
                robot_type=AIRBOT_ARM_ROBOT_TYPE,
                device_id=interface,
                label=_airbot_label("AIRBOT Play", interface, serial_number),
                n_dof=6,
                properties=arm_props,
            )
        )

        eef_type = normalize_airbot_eef_type(properties.get("end_effector_type"))
        robot_type = AIRBOT_EEF_TYPE_TO_ROBOT_TYPE.get(eef_type)
        if robot_type is None:
            continue
        eef_props = dict(properties)
        eef_props["attached_robot_type"] = AIRBOT_ARM_ROBOT_TYPE
        eef_props["end_effector_type"] = eef_type
        found.append(
            DetectedRobot(
                robot_type=robot_type,
                device_id=interface,
                label=_airbot_label(f"AIRBOT {eef_type}", interface, serial_number),
                n_dof=1,
                properties=eef_props,
            )
        )
    return found


__all__ = [
    "AIRBOT_ARM_ROBOT_TYPE",
    "AIRBOT_EEF_TYPE_TO_ROBOT_TYPE",
    "AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR",
    "AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF",
    "AIRBOT_ROBOT_TYPE_TO_SDK_EEF",
    "is_airbot_available",
    "normalize_airbot_eef_type",
    "scan_airbot_detected_robots",
]
