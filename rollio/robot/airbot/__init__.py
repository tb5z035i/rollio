"""AIRBOT-specific robot drivers and protocol helpers."""

from rollio.robot.airbot.can import (
    AIRBOT_BROADCAST_ID,
    AIRBOT_EEF_QUERY_ID,
    AIRBOT_EEF_RESPONSE_ID,
    AIRBOT_EEF_TYPE_CMD,
    AIRBOT_EEF_TYPES,
    AIRBOT_GRAVITY_COEFF_CMD,
    AIRBOT_GRAVITY_EEF_PREFIXES,
    AIRBOT_IDENTIFY_CMD,
    AIRBOT_LED_BLINK_ORANGE,
    AIRBOT_LED_CONTROL_ID,
    AIRBOT_LED_NORMAL,
    AIRBOT_RESPONSE_ID,
    AIRBOT_SERIAL_CMD,
    probe_airbot_device,
    query_airbot_end_effector,
    query_airbot_gravity_coefficients,
    query_airbot_properties,
    query_airbot_serial,
    set_airbot_led,
)
from rollio.robot.airbot.shared import (
    AIRBOT_ARM_ROBOT_TYPE,
    AIRBOT_EEF_TYPE_TO_ROBOT_TYPE,
    AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR,
    AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF,
    AIRBOT_ROBOT_TYPE_TO_SDK_EEF,
    is_airbot_available,
    normalize_airbot_eef_type,
    scan_airbot_detected_robots,
)

try:
    from rollio.robot.airbot.play import AIRBOTPlay
except ImportError:
    AIRBOTPlay = None  # type: ignore

try:
    from rollio.robot.airbot.eef import AIRBOTE2B, AIRBOTG2
except ImportError:
    AIRBOTE2B = None  # type: ignore
    AIRBOTG2 = None  # type: ignore

__all__ = [
    "AIRBOT_BROADCAST_ID",
    "AIRBOT_EEF_QUERY_ID",
    "AIRBOT_EEF_RESPONSE_ID",
    "AIRBOT_EEF_TYPE_CMD",
    "AIRBOT_EEF_TYPES",
    "AIRBOT_GRAVITY_COEFF_CMD",
    "AIRBOT_GRAVITY_EEF_PREFIXES",
    "AIRBOT_IDENTIFY_CMD",
    "AIRBOT_LED_BLINK_ORANGE",
    "AIRBOT_LED_CONTROL_ID",
    "AIRBOT_LED_NORMAL",
    "AIRBOT_RESPONSE_ID",
    "AIRBOT_SERIAL_CMD",
    "probe_airbot_device",
    "query_airbot_end_effector",
    "query_airbot_gravity_coefficients",
    "query_airbot_properties",
    "query_airbot_serial",
    "set_airbot_led",
    "AIRBOTE2B",
    "AIRBOTG2",
    "AIRBOTPlay",
    "AIRBOT_ARM_ROBOT_TYPE",
    "AIRBOT_EEF_TYPE_TO_ROBOT_TYPE",
    "AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR",
    "AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF",
    "AIRBOT_ROBOT_TYPE_TO_SDK_EEF",
    "is_airbot_available",
    "normalize_airbot_eef_type",
    "scan_airbot_detected_robots",
]
