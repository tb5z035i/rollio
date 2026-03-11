"""CAN bus utilities using python-can.

This module provides utilities for SocketCAN interface detection,
interface inspection, and raw CAN communication.
"""

from __future__ import annotations

from pathlib import Path

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
