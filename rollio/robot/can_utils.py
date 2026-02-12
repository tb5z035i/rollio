"""CAN bus utilities using python-can.

This module provides utilities for SocketCAN interface detection,
communication, and device identification.
"""
from __future__ import annotations

import socket
import struct
from pathlib import Path
from typing import TYPE_CHECKING

# Lazy import python-can
_can = None
_CAN_AVAILABLE = None


def _import_can():
    """Lazy import python-can."""
    global _can, _CAN_AVAILABLE
    if _CAN_AVAILABLE is None:
        try:
            import can
            _can = can
            _CAN_AVAILABLE = True
        except ImportError:
            _CAN_AVAILABLE = False
    return _can, _CAN_AVAILABLE


def is_python_can_available() -> bool:
    """Check if python-can is available."""
    _, available = _import_can()
    return available


def scan_can_interfaces() -> list[str]:
    """Scan for available SocketCAN interfaces.
    
    Uses the Linux network interface enumeration via /sys/class/net.
    
    Returns:
        List of CAN interface names (e.g., ["can0", "can1"])
    """
    interfaces = []
    net_path = Path("/sys/class/net")
    
    if not net_path.exists():
        return interfaces
    
    for iface_path in net_path.iterdir():
        iface_name = iface_path.name
        # Check if it's a CAN interface by looking at the type
        type_path = iface_path / "type"
        if type_path.exists():
            try:
                iface_type = type_path.read_text().strip()
                # CAN interfaces have type 280 (ARPHRD_CAN)
                if iface_type == "280":
                    interfaces.append(iface_name)
            except (OSError, IOError):
                pass
    
    return sorted(interfaces)


def is_can_interface_up(interface: str) -> bool:
    """Check if a CAN interface is up and running.
    
    Uses the Linux network interface flags via /sys/class/net.
    
    Args:
        interface: CAN interface name (e.g., "can0")
        
    Returns:
        True if interface is UP
    """
    flags_path = Path(f"/sys/class/net/{interface}/flags")
    
    if not flags_path.exists():
        return False
    
    try:
        flags_hex = flags_path.read_text().strip()
        flags = int(flags_hex, 16)
        # IFF_UP = 0x1
        return bool(flags & 0x1)
    except (OSError, IOError, ValueError):
        return False


def get_can_interface_info(interface: str) -> dict:
    """Get information about a CAN interface.
    
    Args:
        interface: CAN interface name
        
    Returns:
        Dict with interface info (bitrate, state, etc.)
    """
    info = {
        "name": interface,
        "exists": False,
        "is_up": False,
        "bitrate": None,
    }
    
    iface_path = Path(f"/sys/class/net/{interface}")
    if not iface_path.exists():
        return info
    
    info["exists"] = True
    info["is_up"] = is_can_interface_up(interface)
    
    # Try to get bitrate from /sys/class/net/<iface>/can_bittiming/bitrate
    bitrate_path = iface_path / "can_bittiming" / "bitrate"
    if bitrate_path.exists():
        try:
            info["bitrate"] = int(bitrate_path.read_text().strip())
        except (OSError, IOError, ValueError):
            pass
    
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# CAN Communication
# ═══════════════════════════════════════════════════════════════════════════════


class CANBus:
    """Context manager for python-can bus communication."""
    
    def __init__(
        self, 
        interface: str, 
        bustype: str = "socketcan",
        bitrate: int | None = None,
    ) -> None:
        """Initialize CAN bus.
        
        Args:
            interface: CAN interface name (e.g., "can0")
            bustype: CAN bus type (default: "socketcan")
            bitrate: Bitrate (only used if interface needs to be configured)
        """
        can, available = _import_can()
        if not available:
            raise ImportError(
                "python-can is required for CAN communication. "
                "Install with: pip install python-can"
            )
        
        self._can = can
        self._interface = interface
        self._bustype = bustype
        self._bitrate = bitrate
        self._bus = None
    
    def __enter__(self) -> "CANBus":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def open(self) -> None:
        """Open the CAN bus."""
        if self._bus is not None:
            return
        
        self._bus = self._can.interface.Bus(
            channel=self._interface,
            interface=self._bustype,
        )
    
    def close(self) -> None:
        """Close the CAN bus."""
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
    
    def send(
        self, 
        arbitration_id: int, 
        data: bytes | list[int],
        is_extended_id: bool = False,
    ) -> bool:
        """Send a CAN message.
        
        Args:
            arbitration_id: CAN ID (11-bit standard or 29-bit extended)
            data: Data bytes (up to 8 bytes)
            is_extended_id: Use extended (29-bit) ID
            
        Returns:
            True if sent successfully
        """
        if self._bus is None:
            return False
        
        if isinstance(data, list):
            data = bytes(data)
        
        msg = self._can.Message(
            arbitration_id=arbitration_id,
            data=data,
            is_extended_id=is_extended_id,
        )
        
        try:
            self._bus.send(msg)
            return True
        except self._can.CanError:
            return False
    
    def recv(self, timeout: float = 1.0) -> tuple[int, bytes] | None:
        """Receive a CAN message.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            (arbitration_id, data) tuple or None if timeout
        """
        if self._bus is None:
            return None
        
        try:
            msg = self._bus.recv(timeout=timeout)
            if msg is not None:
                return (msg.arbitration_id, bytes(msg.data))
        except self._can.CanError:
            pass
        
        return None
    
    def recv_all(
        self, 
        timeout: float = 0.5,
        max_messages: int = 100,
    ) -> list[tuple[int, bytes]]:
        """Receive all available CAN messages.
        
        Args:
            timeout: Timeout for each receive
            max_messages: Maximum messages to receive
            
        Returns:
            List of (arbitration_id, data) tuples
        """
        messages = []
        
        for _ in range(max_messages):
            result = self.recv(timeout=timeout)
            if result is None:
                break
            messages.append(result)
        
        return messages


# ═══════════════════════════════════════════════════════════════════════════════
# AIRBOT-specific CAN Protocol
# ═══════════════════════════════════════════════════════════════════════════════


# CAN IDs for AIRBOT protocol
AIRBOT_BROADCAST_ID = 0x000
AIRBOT_EEF_QUERY_ID = 0x008       # End effector query
AIRBOT_LED_CONTROL_ID = 0x080
AIRBOT_RESPONSE_ID = 0x100
AIRBOT_EEF_RESPONSE_ID = 0x108   # End effector response

# Commands
AIRBOT_SERIAL_CMD = 0x04          # Query serial number
AIRBOT_EEF_TYPE_CMD = 0x05        # Query end effector type
AIRBOT_IDENTIFY_CMD = 0x07        # Query device identity
AIRBOT_GRAVITY_COEFF_CMD = 0x17   # Query gravity compensation coefficients

# Gravity compensation coefficient joint ID prefixes by end effector type
# Joint IDs are encoded as: (prefix << 4) | joint_number (1-6)
AIRBOT_GRAVITY_EEF_PREFIXES = {
    "none": 0x00,   # 0x01-0x06
    "E2B": 0x10,    # 0x11-0x16
    "G2": 0x20,     # 0x21-0x26
    "other": 0x30,  # 0x31-0x36
}

# LED control
AIRBOT_LED_BLINK_ORANGE = bytes([0x15, 0x01, 0x22])
AIRBOT_LED_NORMAL = bytes([0x15, 0x01, 0x1F])

# End effector type mapping
AIRBOT_EEF_TYPES = {
    0x00: "none",      # No end effector
    0x02: "E2B",       # E2B end effector
    0x03: "G2",        # G2 gripper
}


def probe_airbot_device(interface: str, timeout: float = 1.0) -> bool:
    """Probe for an AIRBOT device on a CAN interface.
    
    Sends the identify command (0x000#07) and checks for the expected
    response pattern that identifies an AIRBOT arm-interface-board.
    
    Args:
        interface: CAN interface name
        timeout: Timeout for response
        
    Returns:
        True if AIRBOT device is detected
    """
    if not is_can_interface_up(interface):
        return False
    
    can, available = _import_can()
    if not available:
        return False
    
    try:
        with CANBus(interface) as bus:
            # Clear any pending messages
            bus.recv_all(timeout=0.1, max_messages=50)
            
            # Send identify command: 0x000#07
            bus.send(AIRBOT_BROADCAST_ID, bytes([AIRBOT_IDENTIFY_CMD]))
            
            # Collect responses
            responses = []
            end_time = __import__('time').time() + timeout
            
            while __import__('time').time() < end_time:
                result = bus.recv(timeout=0.1)
                if result is None:
                    continue
                
                arb_id, data = result
                
                # AIRBOT response comes on ID 0x100
                if arb_id == AIRBOT_RESPONSE_ID and len(data) >= 2:
                    if data[0] == AIRBOT_IDENTIFY_CMD:
                        responses.append(data)
                        
                        # Check if we have enough responses
                        if len(responses) >= 6:
                            break
            
            # Verify response pattern: should spell out "arm-interface-board-base"
            if len(responses) >= 4:
                # Reconstruct the response string
                response_str = ""
                for resp in sorted(responses, key=lambda x: x[1] if len(x) > 1 else 0):
                    if len(resp) > 2:
                        response_str += resp[2:].decode('ascii', errors='ignore')
                
                # Check for AIRBOT identifier
                if "arm-" in response_str.lower() or "airbot" in response_str.lower():
                    return True
                
                # Also accept if we got any valid response pattern
                if len(responses) >= 4:
                    return True
            
            return False
            
    except Exception:
        return False


def set_airbot_led(interface: str, blink_orange: bool = True) -> bool:
    """Set the AIRBOT LED state.
    
    Args:
        interface: CAN interface name
        blink_orange: True to blink orange, False to return to normal
        
    Returns:
        True if command sent successfully
    """
    if not is_can_interface_up(interface):
        return False
    
    try:
        with CANBus(interface) as bus:
            if blink_orange:
                return bus.send(AIRBOT_LED_CONTROL_ID, AIRBOT_LED_BLINK_ORANGE)
            else:
                return bus.send(AIRBOT_LED_CONTROL_ID, AIRBOT_LED_NORMAL)
    except Exception:
        return False


def query_airbot_serial(interface: str, timeout: float = 1.0) -> str | None:
    """Query the AIRBOT robot serial number.
    
    Sends command 0x000#04 and collects the response which contains
    the serial number split across multiple CAN frames.
    
    Args:
        interface: CAN interface name
        timeout: Timeout for response
        
    Returns:
        Serial number string, or None if query failed
    """
    if not is_can_interface_up(interface):
        return None
    
    try:
        with CANBus(interface) as bus:
            # Clear any pending messages
            bus.recv_all(timeout=0.1, max_messages=50)
            
            # Send serial number query: 0x000#04
            bus.send(AIRBOT_BROADCAST_ID, bytes([AIRBOT_SERIAL_CMD]))
            
            # Collect responses
            responses: list[tuple[int, bytes]] = []
            end_time = __import__('time').time() + timeout
            
            while __import__('time').time() < end_time:
                result = bus.recv(timeout=0.1)
                if result is None:
                    continue
                
                arb_id, data = result
                
                # Response comes on ID 0x100 with command byte 0x04
                if arb_id == AIRBOT_RESPONSE_ID and len(data) >= 2:
                    if data[0] == AIRBOT_SERIAL_CMD:
                        responses.append((data[1], data))  # (sequence, data)
                        
                        # Serial number is typically 4 frames
                        if len(responses) >= 4:
                            break
            
            if not responses:
                return None
            
            # Sort by sequence number and reconstruct serial
            responses.sort(key=lambda x: x[0])
            serial_parts = []
            for _, data in responses:
                if len(data) > 2:
                    serial_parts.append(data[2:].decode('ascii', errors='ignore'))
            
            return ''.join(serial_parts).strip('\x00')
            
    except Exception:
        return None


def query_airbot_end_effector(interface: str, timeout: float = 1.0) -> dict | None:
    """Query the AIRBOT end effector type.
    
    Sends command 0x008#05 and parses the response to determine
    the attached end effector type.
    
    Args:
        interface: CAN interface name
        timeout: Timeout for response
        
    Returns:
        Dictionary with end effector info:
        - "type_code": Raw type code (int)
        - "type_name": Human-readable type name (str)
        Or None if query failed
    """
    if not is_can_interface_up(interface):
        return None
    
    try:
        with CANBus(interface) as bus:
            # Clear any pending messages
            bus.recv_all(timeout=0.1, max_messages=50)
            
            # Send end effector query: 0x008#05
            bus.send(AIRBOT_EEF_QUERY_ID, bytes([AIRBOT_EEF_TYPE_CMD]))
            
            # Wait for response
            end_time = __import__('time').time() + timeout
            
            while __import__('time').time() < end_time:
                result = bus.recv(timeout=0.1)
                if result is None:
                    continue
                
                arb_id, data = result
                
                # Response comes on ID 0x108 with command byte 0x05
                if arb_id == AIRBOT_EEF_RESPONSE_ID and len(data) >= 3:
                    if data[0] == AIRBOT_EEF_TYPE_CMD:
                        type_code = data[2]  # Third byte is the type
                        type_name = AIRBOT_EEF_TYPES.get(type_code, f"unknown_{type_code:02x}")
                        return {
                            "type_code": type_code,
                            "type_name": type_name,
                        }
            
            return None
            
    except Exception:
        return None


def query_airbot_gravity_coefficients(
    interface: str, 
    timeout: float = 1.0
) -> dict[str, list[float]] | None:
    """Query the AIRBOT gravity compensation coefficients.
    
    Sends command 0x000#17 and collects 24 responses (6 joints × 4 end effector types).
    Each response contains a float32 coefficient for a specific joint and EEF type.
    
    Response format: [0x17, joint_id, float32 (4 bytes, little-endian)]
    Joint ID encoding: (eef_prefix << 4) | joint_number
    - 0x01-0x06: no end effector
    - 0x11-0x16: E2B end effector
    - 0x21-0x26: G2 gripper
    - 0x31-0x36: other
    
    Args:
        interface: CAN interface name
        timeout: Timeout for response collection
        
    Returns:
        Dictionary mapping end effector type to list of 6 coefficients:
        {
            "none": [c1, c2, c3, c4, c5, c6],
            "E2B": [c1, c2, c3, c4, c5, c6],
            "G2": [c1, c2, c3, c4, c5, c6],
            "other": [c1, c2, c3, c4, c5, c6],
        }
        Or None if query failed
    """
    if not is_can_interface_up(interface):
        return None
    
    try:
        with CANBus(interface) as bus:
            # Clear any pending messages
            bus.recv_all(timeout=0.1, max_messages=50)
            
            # Send gravity coefficients query: 0x000#17
            bus.send(AIRBOT_BROADCAST_ID, bytes([AIRBOT_GRAVITY_COEFF_CMD]))
            
            # Collect responses - expect 24 (6 joints × 4 EEF types)
            responses: dict[int, bytes] = {}  # joint_id -> float bytes
            end_time = __import__('time').time() + timeout
            
            while __import__('time').time() < end_time:
                result = bus.recv(timeout=0.05)
                if result is None:
                    # Check if we have all 24 responses
                    if len(responses) >= 24:
                        break
                    continue
                
                arb_id, data = result
                
                # Response comes on ID 0x100 with command byte 0x17
                if arb_id == AIRBOT_RESPONSE_ID and len(data) >= 6:
                    if data[0] == AIRBOT_GRAVITY_COEFF_CMD:
                        joint_id = data[1]
                        float_bytes = data[2:6]
                        responses[joint_id] = float_bytes
                        
                        if len(responses) >= 24:
                            break
            
            if len(responses) < 6:
                # Need at least one complete set
                return None
            
            # Parse coefficients into dictionary by EEF type
            coefficients: dict[str, list[float]] = {}
            
            for eef_name, prefix in AIRBOT_GRAVITY_EEF_PREFIXES.items():
                coeff_list = []
                for joint_num in range(1, 7):  # Joints 1-6
                    joint_id = prefix | joint_num
                    if joint_id in responses:
                        # Unpack little-endian float32
                        value = struct.unpack('<f', responses[joint_id])[0]
                        coeff_list.append(value)
                    else:
                        # Use default value if not found
                        coeff_list.append(1.0)
                
                if len(coeff_list) == 6:
                    coefficients[eef_name] = coeff_list
            
            return coefficients if coefficients else None
            
    except Exception:
        return None


def query_airbot_properties(interface: str, timeout: float = 1.0) -> dict:
    """Query all available AIRBOT properties.
    
    Queries serial number, end effector type, and device identity.
    
    Args:
        interface: CAN interface name
        timeout: Timeout for each query
        
    Returns:
        Dictionary of properties (may be empty or partial if queries fail)
    """
    properties = {}
    
    # Query serial number
    serial = query_airbot_serial(interface, timeout)
    if serial:
        properties["serial_number"] = serial
    
    # Query end effector
    eef = query_airbot_end_effector(interface, timeout)
    if eef:
        properties["end_effector_type"] = eef["type_name"]
        properties["end_effector_code"] = eef["type_code"]
    
    # Add CAN interface info
    properties["can_interface"] = interface
    
    return properties
