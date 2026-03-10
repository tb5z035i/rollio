"""Compatibility bridge for AIRBOT shared helpers."""
from __future__ import annotations

from rollio.robot.airbot_shared import (
    AIRBOT_ARM_ROBOT_TYPE,
    AIRBOT_EEF_TYPE_TO_ROBOT_TYPE,
    AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR,
    AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF,
    AIRBOT_ROBOT_TYPE_TO_SDK_EEF,
    _import_airbot_hardware,
    is_airbot_available,
    normalize_airbot_eef_type,
    scan_airbot_detected_robots,
)

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
