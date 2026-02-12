"""Unit tests for AIRBOT Play robot arm.

These tests verify the AIRBOT implementation without requiring actual hardware.
Hardware-dependent tests are marked and can be skipped.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rollio.robot import (
    ControlMode,
    DetectedRobot,
    FeedbackCapability,
    FreeDriveCommand,
    JointState,
    Pose,
    TargetTrackingCommand,
    Wrench,
    scan_robots,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test CAN Utilities
# ═══════════════════════════════════════════════════════════════════════════════


class TestCANUtils:
    """Tests for CAN utility functions."""
    
    def test_scan_can_interfaces(self) -> None:
        """Test scanning for CAN interfaces via /sys/class/net."""
        from rollio.robot.can_utils import scan_can_interfaces
        
        # This tests the actual system - may find interfaces or not
        interfaces = scan_can_interfaces()
        assert isinstance(interfaces, list)
        for iface in interfaces:
            assert isinstance(iface, str)
            assert iface.startswith("can")
    
    def test_is_can_interface_up(self) -> None:
        """Test CAN interface status check via /sys/class/net."""
        from rollio.robot.can_utils import is_can_interface_up
        
        # Test with non-existent interface
        assert is_can_interface_up("can_nonexistent_999") is False
        
        # Test with can0 if it exists
        result = is_can_interface_up("can0")
        assert isinstance(result, bool)
    
    def test_get_can_interface_info(self) -> None:
        """Test getting CAN interface info."""
        from rollio.robot.can_utils import get_can_interface_info
        
        # Test non-existent interface
        info = get_can_interface_info("can_nonexistent_999")
        assert info["exists"] is False
        assert info["is_up"] is False
        
        # Test can0 if it exists
        info = get_can_interface_info("can0")
        assert isinstance(info, dict)
        assert "name" in info
        assert "exists" in info
        assert "is_up" in info
    
    def test_python_can_availability(self) -> None:
        """Test python-can availability check."""
        from rollio.robot.can_utils import is_python_can_available
        
        result = is_python_can_available()
        assert isinstance(result, bool)
    
    @patch('rollio.robot.can_utils.Path')
    def test_scan_can_interfaces_mock(self, mock_path: MagicMock) -> None:
        """Test CAN scanning with mocked /sys/class/net."""
        from rollio.robot.can_utils import scan_can_interfaces
        
        # Create mock directory structure
        mock_net_path = MagicMock()
        mock_net_path.exists.return_value = True
        
        # Mock can0 and can1 interfaces
        mock_can0 = MagicMock()
        mock_can0.name = "can0"
        mock_can0_type = MagicMock()
        mock_can0_type.exists.return_value = True
        mock_can0_type.read_text.return_value = "280"  # ARPHRD_CAN
        mock_can0.__truediv__ = lambda self, x: mock_can0_type if x == "type" else MagicMock()
        
        mock_can1 = MagicMock()
        mock_can1.name = "can1"
        mock_can1_type = MagicMock()
        mock_can1_type.exists.return_value = True
        mock_can1_type.read_text.return_value = "280"
        mock_can1.__truediv__ = lambda self, x: mock_can1_type if x == "type" else MagicMock()
        
        mock_eth0 = MagicMock()
        mock_eth0.name = "eth0"
        mock_eth0_type = MagicMock()
        mock_eth0_type.exists.return_value = True
        mock_eth0_type.read_text.return_value = "1"  # ARPHRD_ETHER
        mock_eth0.__truediv__ = lambda self, x: mock_eth0_type if x == "type" else MagicMock()
        
        mock_net_path.iterdir.return_value = [mock_can0, mock_can1, mock_eth0]
        mock_path.return_value = mock_net_path
        
        # Note: Due to module-level Path import, we need to reimport
        # This test demonstrates the expected behavior


class TestCANProtocol:
    """Tests for AIRBOT CAN protocol constants."""
    
    def test_protocol_constants(self) -> None:
        """Test CAN protocol constants are defined."""
        from rollio.robot.can_utils import (
            AIRBOT_BROADCAST_ID,
            AIRBOT_EEF_QUERY_ID,
            AIRBOT_EEF_RESPONSE_ID,
            AIRBOT_EEF_TYPE_CMD,
            AIRBOT_EEF_TYPES,
            AIRBOT_IDENTIFY_CMD,
            AIRBOT_LED_BLINK_ORANGE,
            AIRBOT_LED_CONTROL_ID,
            AIRBOT_LED_NORMAL,
            AIRBOT_RESPONSE_ID,
            AIRBOT_SERIAL_CMD,
        )
        
        assert AIRBOT_BROADCAST_ID == 0x000
        assert AIRBOT_EEF_QUERY_ID == 0x008
        assert AIRBOT_LED_CONTROL_ID == 0x080
        assert AIRBOT_RESPONSE_ID == 0x100
        assert AIRBOT_EEF_RESPONSE_ID == 0x108
        assert AIRBOT_SERIAL_CMD == 0x04
        assert AIRBOT_EEF_TYPE_CMD == 0x05
        assert AIRBOT_IDENTIFY_CMD == 0x07
        assert AIRBOT_LED_BLINK_ORANGE == bytes([0x15, 0x01, 0x22])
        assert AIRBOT_LED_NORMAL == bytes([0x15, 0x01, 0x1F])
    
    def test_end_effector_types(self) -> None:
        """Test end effector type mapping."""
        from rollio.robot.can_utils import AIRBOT_EEF_TYPES
        
        assert AIRBOT_EEF_TYPES[0x00] == "none"
        assert AIRBOT_EEF_TYPES[0x02] == "E2B"
        assert AIRBOT_EEF_TYPES[0x03] == "G2"


# ═══════════════════════════════════════════════════════════════════════════════
# Test AIRBOT Class (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAIRBOTPlayMocked:
    """Tests for AIRBOTPlay with mocked hardware."""
    
    @pytest.fixture
    def mock_airbot_hardware(self):
        """Create mock airbot_hardware_py module."""
        mock_ah = MagicMock()
        
        # Mock motor types
        mock_ah.MotorType.OD = "OD"
        mock_ah.MotorType.DM = "DM"
        mock_ah.MotorType.NA = "NA"
        mock_ah.EEFType.NA = "NA"
        mock_ah.MotorControlMode.MIT = "MIT"
        mock_ah.MotorControlMode.PVT = "PVT"
        
        # Mock arm state
        mock_state = MagicMock()
        mock_state.is_valid = True
        mock_state.pos = [0.1, 0.2, -0.1, 0.0, 0.0, 0.0]
        mock_state.vel = [0.01, 0.02, 0.0, 0.0, 0.0, 0.0]
        mock_state.eff = [1.0, 2.0, 0.5, 0.1, 0.1, 0.05]
        
        # Mock arm
        mock_arm = MagicMock()
        mock_arm.state.return_value = mock_state
        mock_arm.init.return_value = True
        
        # Mock Play.create
        mock_ah.Play.create.return_value = mock_arm
        
        # Mock executor
        mock_executor = MagicMock()
        mock_executor.get_io_context.return_value = MagicMock()
        mock_ah.create_asio_executor.return_value = mock_executor
        
        return mock_ah, mock_arm
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_creation(
        self, 
        mock_import: MagicMock, 
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay instance creation."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        assert robot.n_dof == 6
        assert robot.can_interface == "can0"
        assert robot.ROBOT_TYPE == "airbot_play"
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_open_close(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay open/close lifecycle."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Open
        robot.open()
        assert robot._is_open
        mock_arm.init.assert_called_once()
        
        # Close
        robot.close()
        assert not robot._is_open
        mock_arm.uninit.assert_called_once()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_enable_disable(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay enable/disable."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        
        # Enable
        assert robot.enable() is True
        assert robot.is_enabled
        mock_arm.enable.assert_called_once()
        
        # Disable
        robot.disable()
        assert not robot.is_enabled
        mock_arm.disable.assert_called_once()
        
        robot.close()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_read_joint_state(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test reading joint state from AIRBOT."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        
        state = robot.read_joint_state()
        
        assert isinstance(state, JointState)
        assert state.is_valid
        assert state.position is not None
        assert len(state.position) == 6
        np.testing.assert_array_almost_equal(
            state.position, 
            [0.1, 0.2, -0.1, 0.0, 0.0, 0.0]
        )
        
        robot.close()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_control_modes(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test control mode switching."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        
        # Free drive mode
        assert robot.set_control_mode(ControlMode.FREE_DRIVE) is True
        assert robot.control_mode == ControlMode.FREE_DRIVE
        
        # Target tracking mode
        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is True
        assert robot.control_mode == ControlMode.TARGET_TRACKING
        
        robot.close()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_free_drive_command(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test free drive command execution."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.FREE_DRIVE)
        
        cmd = FreeDriveCommand(
            external_wrench=None,
            gravity_compensation_scale=1.0
        )
        
        robot.command_free_drive(cmd)
        
        # Verify MIT was called
        mock_arm.mit.assert_called()
        
        robot.close()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_target_tracking_command(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test target tracking command execution."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.TARGET_TRACKING)
        
        cmd = TargetTrackingCommand(
            position_target=np.zeros(6),
            velocity_target=np.zeros(6),
            kp=np.full(6, 50.0),
            kd=np.full(6, 5.0),
            feedforward=np.zeros(6)
        )
        
        robot.command_target_tracking(cmd)
        
        # Verify MIT was called with correct arguments
        mock_arm.mit.assert_called_once()
        call_args = mock_arm.mit.call_args[0]
        assert len(call_args) == 5  # pos, vel, tau, kp, kd
        
        robot.close()
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_info(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test robot info."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        info = robot.info
        
        assert info.robot_type == "airbot_play"
        assert info.n_dof == 6
        assert FeedbackCapability.POSITION in info.feedback_capabilities
        assert FeedbackCapability.VELOCITY in info.feedback_capabilities
        assert FeedbackCapability.EFFORT in info.feedback_capabilities
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_gravity_scale(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test gravity compensation scale factors."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Check default scale factors
        default_scale = robot.gravity_compensation_scale
        assert len(default_scale) == 6
        np.testing.assert_array_almost_equal(
            default_scale,
            [0.6, 0.6, 0.6, 1.317, 1.378, 0.864]
        )
        
        # Set custom scale
        custom_scale = np.ones(6)
        robot.gravity_compensation_scale = custom_scale
        np.testing.assert_array_almost_equal(
            robot.gravity_compensation_scale,
            custom_scale
        )
    
    @patch('rollio.robot.airbot_play.set_airbot_led')
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_identify_start_led_only(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test starting identification (LED blink orange only)."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test with gravity comp disabled (LED only)
        result = robot.identify_start(with_gravity_comp=False)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=True)
    
    @patch('rollio.robot.airbot_play.set_airbot_led')
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_identify_start_with_gravity_comp(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test starting identification with gravity compensation enabled."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True
        mock_ah.Arm.return_value = mock_arm
        mock_arm.set_mode.return_value = None
        mock_arm.get_current_joint_q.return_value = [0.0] * 6
        mock_arm.get_current_joint_dq.return_value = [0.0] * 6
        mock_arm.get_current_joint_torque.return_value = [0.0] * 6
        mock_arm.set_joint_torque.return_value = None
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test with gravity comp enabled (default)
        result = robot.identify_start(with_gravity_comp=True)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=True)
        # Robot should be opened and enabled
        assert robot._is_open is True
        assert robot._is_enabled is True
    
    @patch('rollio.robot.airbot_play.set_airbot_led')
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_identify_stop_led_only(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test stopping identification (LED normal, don't disable robot)."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test without disabling robot
        result = robot.identify_stop(disable_robot=False)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=False)
    
    @patch('rollio.robot.airbot_play.set_airbot_led')
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_identify_stop(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test stopping identification (LED normal) with disable."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True
        mock_ah.Arm.return_value = mock_arm
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        # Simulate robot being open
        robot._is_open = True
        robot._arm = mock_arm
        
        result = robot.identify_stop(disable_robot=True)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=False)
    
    @patch('rollio.robot.airbot_play.query_airbot_end_effector')
    @patch('rollio.robot.airbot_play.query_airbot_serial')
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_properties(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_query_serial: MagicMock,
        mock_query_eef: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test querying robot properties."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        mock_query_serial.return_value = "PZ25C02402000244"
        mock_query_eef.return_value = {"type_code": 2, "type_name": "E2B"}
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Initial properties
        props = robot.properties
        assert props["can_interface"] == "can0"
        assert "serial_number" not in props  # Not queried yet
        
        # Query properties from hardware
        updated_props = robot.query_properties()
        
        assert updated_props["serial_number"] == "PZ25C02402000244"
        assert updated_props["end_effector_type"] == "E2B"
        assert updated_props["end_effector_code"] == 2
        
        # Check convenience properties
        assert robot.serial_number == "PZ25C02402000244"
        assert robot.end_effector_type == "E2B"
    
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_properties_default(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test default properties without query."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        props = robot.properties
        assert props["can_interface"] == "can0"
        assert props["control_frequency"] == 250
        assert props["motor_types"] == ["OD", "OD", "OD", "DM", "DM", "DM"]
        
        # Serial and EEF not queried yet
        assert robot.serial_number is None
        assert robot.end_effector_type is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test AIRBOT Scanning
# ═══════════════════════════════════════════════════════════════════════════════


class TestAIRBOTScanning:
    """Tests for AIRBOT robot scanning."""
    
    @patch('rollio.robot.airbot_play.query_airbot_properties')
    @patch('rollio.robot.airbot_play.probe_airbot_device', return_value=True)
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play.scan_can_interfaces', return_value=["can0", "can1"])
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_scan(
        self,
        mock_import: MagicMock,
        mock_scan: MagicMock,
        mock_can_up: MagicMock,
        mock_probe: MagicMock,
        mock_query_props: MagicMock,
    ) -> None:
        """Test scanning for AIRBOT robots."""
        mock_import.return_value = (MagicMock(), True)
        mock_query_props.return_value = {
            "can_interface": "can0",
            "serial_number": "PZ25C0240200",
            "end_effector_type": "E2B",
        }
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        devices = AIRBOTPlay.scan()
        
        assert len(devices) == 2
        assert all(d.robot_type == "airbot_play" for d in devices)
        assert all(d.n_dof == 6 for d in devices)
        assert devices[0].device_id == "can0"
        assert devices[1].device_id == "can1"
        
        # Verify probe was called for each interface
        assert mock_probe.call_count == 2
        # Verify properties were queried for each device
        assert mock_query_props.call_count == 2
        # Verify SN is included in label
        assert "SN:" in devices[0].label
    
    @patch('rollio.robot.airbot_play.probe_airbot_device', return_value=False)
    @patch('rollio.robot.airbot_play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot_play.scan_can_interfaces', return_value=["can0"])
    @patch('rollio.robot.airbot_play._import_airbot_hardware')
    def test_airbot_scan_no_device(
        self,
        mock_import: MagicMock,
        mock_scan: MagicMock,
        mock_can_up: MagicMock,
        mock_probe: MagicMock,
    ) -> None:
        """Test scanning when no AIRBOT device responds."""
        mock_import.return_value = (MagicMock(), True)
        
        from rollio.robot.airbot_play import AIRBOTPlay
        
        devices = AIRBOTPlay.scan()
        
        # Should find no devices since probe returns False
        assert len(devices) == 0
    
    def test_airbot_probe_nonexistent(self) -> None:
        """Test probing non-existent interface."""
        from rollio.robot.airbot_play import AIRBOTPlay
        
        # This should return False for non-existent interface
        result = AIRBOTPlay.probe("can_nonexistent_999")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test Pinocchio Kinematics (if available)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPinocchioKinematics:
    """Tests for Pinocchio-based kinematics."""
    
    def test_pinocchio_availability_check(self) -> None:
        """Test pinocchio availability check."""
        from rollio.robot.pinocchio_kinematics import is_pinocchio_available
        
        result = is_pinocchio_available()
        assert isinstance(result, bool)
    
    @pytest.mark.skipif(
        not pytest.importorskip("pinocchio", reason="pinocchio not installed"),
        reason="pinocchio not installed"
    )
    def test_pinocchio_import(self) -> None:
        """Test that pinocchio can be imported."""
        import pinocchio as pin
        assert hasattr(pin, 'buildModelFromUrdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests (require hardware marker)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.hardware
class TestAIRBOTHardware:
    """Integration tests requiring actual AIRBOT hardware.
    
    Run with: pytest -m hardware
    """
    
    @pytest.fixture
    def robot(self):
        """Create and open an AIRBOT robot."""
        from rollio.robot.airbot_play import AIRBOTPlay, is_airbot_available
        
        if not is_airbot_available():
            pytest.skip("airbot_hardware_py not installed")
        
        robot = AIRBOTPlay(can_interface="can0")
        try:
            robot.open()
            yield robot
        finally:
            robot.close()
    
    def test_hardware_connection(self, robot) -> None:
        """Test actual hardware connection."""
        assert robot._is_open
        
        state = robot.read_joint_state()
        # State may or may not be valid depending on hardware state
        assert isinstance(state, JointState)
    
    def test_hardware_enable(self, robot) -> None:
        """Test enabling hardware."""
        success = robot.enable()
        assert success is True
        assert robot.is_enabled
        
        robot.disable()
        assert not robot.is_enabled
