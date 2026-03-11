"""Shared AIRBOT helpers for discovery and SDK imports."""

from __future__ import annotations

import threading
from typing import Any

from rollio.robot.airbot.control_loop import AirbotCommandPump
from rollio.robot.airbot.can import probe_airbot_device, query_airbot_properties
from rollio.robot.base import ControlMode
from rollio.robot.can_utils import is_can_interface_up, scan_can_interfaces
from rollio.robot.scanner import DetectedRobot

_AIRBOT_SHARED_RUNTIME_LOCK = threading.Lock()
_AIRBOT_SHARED_RUNTIMES: dict[int, tuple[Any, Any, Any]] = {}

AIRBOT_SHARED_EXECUTOR_WORKERS = 12

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


def is_airbot_available() -> bool:
    """Check if ``airbot_hardware_py`` is available."""
    try:
        import airbot_hardware_py  # pylint: disable=import-outside-toplevel,unused-import

        return True
    except ImportError:
        return False


def get_shared_airbot_runtime(ah: Any) -> tuple[Any, Any]:
    """Return the shared AIRBOT executor and ``io_context``.

    The SDK is typically imported once per process, so all AIRBOT arms and
    standalone EEFs will share the same executor created with the configured
    shared-worker count.

    The cache is keyed by SDK-module identity so tests can swap in isolated
    mocked SDK modules without leaking shared state between runs.
    """
    runtime_key = id(ah)
    with _AIRBOT_SHARED_RUNTIME_LOCK:
        cached_runtime = _AIRBOT_SHARED_RUNTIMES.get(runtime_key)
        if cached_runtime is not None and cached_runtime[0] is ah:
            return cached_runtime[1], cached_runtime[2]

        executor = ah.create_asio_executor(AIRBOT_SHARED_EXECUTOR_WORKERS)
        io_context = executor.get_io_context()
        _AIRBOT_SHARED_RUNTIMES[runtime_key] = (ah, executor, io_context)
        return executor, io_context


def start_airbot_command_pump(
    current_pump: AirbotCommandPump | None,
    *,
    name: str,
    period_sec: float,
    apply_enabled: Any,
    apply_mode: Any,
    cycle: Any,
    initial_enabled: bool,
    initial_mode: ControlMode,
) -> AirbotCommandPump:
    """Create and start one command pump when absent."""

    if current_pump is not None:
        return current_pump
    command_pump = AirbotCommandPump(
        name=name,
        period_sec=period_sec,
        apply_enabled=apply_enabled,
        apply_mode=apply_mode,
        cycle=cycle,
        initial_enabled=initial_enabled,
        initial_mode=initial_mode,
    )
    command_pump.start()
    return command_pump


def stop_airbot_command_pump(command_pump: AirbotCommandPump | None) -> None:
    """Stop one command pump when present."""

    if command_pump is not None:
        command_pump.stop()


def publish_airbot_command(
    command_pump: AirbotCommandPump | None,
    command: Any,
    *,
    owner: str | None = None,
) -> bool:
    """Publish one latest command when the pump is active."""

    if command_pump is None:
        return False
    if owner is None:
        return command_pump.publish_command(command)
    return command_pump.publish_command(command, owner=owner)


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
    "AIRBOT_SHARED_EXECUTOR_WORKERS",
    "get_shared_airbot_runtime",
    "is_airbot_available",
    "normalize_airbot_eef_type",
    "publish_airbot_command",
    "scan_airbot_detected_robots",
    "start_airbot_command_pump",
    "stop_airbot_command_pump",
]
