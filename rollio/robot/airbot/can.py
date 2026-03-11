"""AIRBOT CAN protocol helpers built on top of SocketCAN utilities."""

from __future__ import annotations

import struct
import time

from rollio.robot.can_utils import CANBus, is_can_interface_up

# CAN IDs for AIRBOT protocol
AIRBOT_BROADCAST_ID = 0x000
AIRBOT_EEF_QUERY_ID = 0x008
AIRBOT_LED_CONTROL_ID = 0x080
AIRBOT_RESPONSE_ID = 0x100
AIRBOT_EEF_RESPONSE_ID = 0x108

# Commands
AIRBOT_SERIAL_CMD = 0x04
AIRBOT_EEF_TYPE_CMD = 0x05
AIRBOT_IDENTIFY_CMD = 0x07
AIRBOT_GRAVITY_COEFF_CMD = 0x17

# Gravity compensation coefficient joint ID prefixes by end effector type.
# Joint IDs are encoded as: (prefix << 4) | joint_number (1-6)
AIRBOT_GRAVITY_EEF_PREFIXES = {
    "none": 0x00,
    "E2B": 0x10,
    "G2": 0x20,
    "other": 0x30,
}

# LED control
AIRBOT_LED_BLINK_ORANGE = bytes([0x15, 0x01, 0x22])
AIRBOT_LED_NORMAL = bytes([0x15, 0x01, 0x1F])

# End effector type mapping
AIRBOT_EEF_TYPES = {
    0x00: "none",
    0x02: "E2B",
    0x03: "G2",
}


def _collect_airbot_frames(
    interface: str,
    *,
    request_arb_id: int,
    request_payload: bytes,
    timeout: float,
    recv_timeout: float,
    accept_frame,
    stop_when=None,
) -> list[tuple[int, bytes]] | None:
    if not is_can_interface_up(interface):
        return None
    try:
        with CANBus(interface) as bus:
            bus.recv_all(timeout=0.1, max_messages=50)
            bus.send(request_arb_id, request_payload)

            responses: list[tuple[int, bytes]] = []
            end_time = time.time() + timeout
            while time.time() < end_time:
                result = bus.recv(timeout=recv_timeout)
                if result is None:
                    if stop_when is not None and stop_when(responses):
                        break
                    continue
                arb_id, data = result
                if not accept_frame(arb_id, data):
                    continue
                responses.append((arb_id, data))
                if stop_when is not None and stop_when(responses):
                    break
            return responses
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def probe_airbot_device(interface: str, timeout: float = 1.0) -> bool:
    """Probe for an AIRBOT device on a CAN interface."""
    responses = _collect_airbot_frames(
        interface,
        request_arb_id=AIRBOT_BROADCAST_ID,
        request_payload=bytes([AIRBOT_IDENTIFY_CMD]),
        timeout=timeout,
        recv_timeout=0.1,
        accept_frame=lambda arb_id, data: (
            arb_id == AIRBOT_RESPONSE_ID
            and len(data) >= 2
            and data[0] == AIRBOT_IDENTIFY_CMD
        ),
        stop_when=lambda frames: len(frames) >= 6,
    )
    if responses is None:
        return False
    response_payloads = [data for _, data in responses]
    if len(response_payloads) < 4:
        return False

    response_str = ""
    for resp in sorted(response_payloads, key=lambda x: x[1] if len(x) > 1 else 0):
        if len(resp) > 2:
            response_str += resp[2:].decode("ascii", errors="ignore")

    if "arm-" in response_str.lower() or "airbot" in response_str.lower():
        return True
    return len(response_payloads) >= 4


def set_airbot_led(interface: str, blink_orange: bool = True) -> bool:
    """Set the AIRBOT LED state."""
    if not is_can_interface_up(interface):
        return False

    try:
        with CANBus(interface) as bus:
            payload = AIRBOT_LED_BLINK_ORANGE if blink_orange else AIRBOT_LED_NORMAL
            return bus.send(AIRBOT_LED_CONTROL_ID, payload)

    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def query_airbot_serial(interface: str, timeout: float = 1.0) -> str | None:
    """Query the AIRBOT robot serial number."""
    responses = _collect_airbot_frames(
        interface,
        request_arb_id=AIRBOT_BROADCAST_ID,
        request_payload=bytes([AIRBOT_SERIAL_CMD]),
        timeout=timeout,
        recv_timeout=0.1,
        accept_frame=lambda arb_id, data: (
            arb_id == AIRBOT_RESPONSE_ID
            and len(data) >= 2
            and data[0] == AIRBOT_SERIAL_CMD
        ),
        stop_when=lambda frames: len(frames) >= 4,
    )
    if not responses:
        return None
    ordered_responses = sorted(
        ((data[1], data) for _, data in responses),
        key=lambda item: item[0],
    )
    serial_parts = [
        data[2:].decode("ascii", errors="ignore")
        for _, data in ordered_responses
        if len(data) > 2
    ]
    return "".join(serial_parts).strip("\x00")


def query_airbot_end_effector(interface: str, timeout: float = 1.0) -> dict | None:
    """Query the AIRBOT end-effector type."""
    responses = _collect_airbot_frames(
        interface,
        request_arb_id=AIRBOT_EEF_QUERY_ID,
        request_payload=bytes([AIRBOT_EEF_TYPE_CMD]),
        timeout=timeout,
        recv_timeout=0.1,
        accept_frame=lambda arb_id, data: (
            arb_id == AIRBOT_EEF_RESPONSE_ID
            and len(data) >= 3
            and data[0] == AIRBOT_EEF_TYPE_CMD
        ),
        stop_when=lambda frames: len(frames) >= 1,
    )
    if not responses:
        return None
    _, data = responses[0]
    type_code = data[2]
    return {
        "type_code": type_code,
        "type_name": AIRBOT_EEF_TYPES.get(type_code, f"unknown_{type_code:02x}"),
    }


def query_airbot_gravity_coefficients(
    interface: str,
    timeout: float = 1.0,
) -> dict[str, list[float]] | None:
    """Query the AIRBOT gravity compensation coefficients."""
    response_frames = _collect_airbot_frames(
        interface,
        request_arb_id=AIRBOT_BROADCAST_ID,
        request_payload=bytes([AIRBOT_GRAVITY_COEFF_CMD]),
        timeout=timeout,
        recv_timeout=0.05,
        accept_frame=lambda arb_id, data: (
            arb_id == AIRBOT_RESPONSE_ID
            and len(data) >= 6
            and data[0] == AIRBOT_GRAVITY_COEFF_CMD
        ),
        stop_when=lambda frames: len(frames) >= 24,
    )
    if not response_frames:
        return None
    responses = {data[1]: data[2:6] for _, data in response_frames}
    if len(responses) < 6:
        return None

    coefficients: dict[str, list[float]] = {}
    for eef_name, prefix in AIRBOT_GRAVITY_EEF_PREFIXES.items():
        coeff_list = []
        for joint_num in range(1, 7):
            joint_id = prefix | joint_num
            if joint_id in responses:
                coeff_list.append(struct.unpack("<f", responses[joint_id])[0])
            else:
                coeff_list.append(1.0)
        coefficients[eef_name] = coeff_list

    return coefficients if coefficients else None


def query_airbot_properties(interface: str, timeout: float = 1.0) -> dict:
    """Query all available AIRBOT properties."""
    properties = {}

    serial = query_airbot_serial(interface, timeout)
    if serial:
        properties["serial_number"] = serial

    eef = query_airbot_end_effector(interface, timeout)
    if eef:
        properties["end_effector_type"] = eef["type_name"]
        properties["end_effector_code"] = eef["type_code"]

    properties["can_interface"] = interface
    return properties


__all__ = [
    "AIRBOT_BROADCAST_ID",
    "AIRBOT_EEF_QUERY_ID",
    "AIRBOT_LED_CONTROL_ID",
    "AIRBOT_RESPONSE_ID",
    "AIRBOT_EEF_RESPONSE_ID",
    "AIRBOT_SERIAL_CMD",
    "AIRBOT_EEF_TYPE_CMD",
    "AIRBOT_IDENTIFY_CMD",
    "AIRBOT_GRAVITY_COEFF_CMD",
    "AIRBOT_GRAVITY_EEF_PREFIXES",
    "AIRBOT_LED_BLINK_ORANGE",
    "AIRBOT_LED_NORMAL",
    "AIRBOT_EEF_TYPES",
    "probe_airbot_device",
    "set_airbot_led",
    "query_airbot_serial",
    "query_airbot_end_effector",
    "query_airbot_gravity_coefficients",
    "query_airbot_properties",
]
