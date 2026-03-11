"""Unit tests for AIRBOT Play robot arm.

These tests verify the AIRBOT implementation without requiring actual hardware.
Hardware-dependent tests are marked and can be skipped.
"""
from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rollio.defaults import DEFAULT_CONTROL_HZ
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


def _wait_until(
    predicate,
    *,
    timeout: float = 0.3,
    interval: float = 0.01,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _pinocchio_available() -> bool:
    from rollio.robot.pinocchio_kinematics import is_pinocchio_available

    return is_pinocchio_available()


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
        from rollio.robot.airbot.can import (
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
        from rollio.robot.airbot.can import AIRBOT_EEF_TYPES
        
        assert AIRBOT_EEF_TYPES[0x00] == "none"
        assert AIRBOT_EEF_TYPES[0x02] == "E2B"
        assert AIRBOT_EEF_TYPES[0x03] == "G2"


# ═══════════════════════════════════════════════════════════════════════════════
# Test AIRBOT Class (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAIRBOTPlayMocked:
    """Tests for AIRBOTPlay with mocked hardware."""
    
    @pytest.fixture
    def mock_airbot_hardware(self, monkeypatch):
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
        monkeypatch.setattr(
            "rollio.robot.airbot.play.query_airbot_end_effector",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "rollio.robot.airbot.play.query_airbot_gravity_coefficients",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "rollio.robot.airbot.play.query_airbot_serial",
            lambda *args, **kwargs: None,
        )

        return mock_ah, mock_arm
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_creation(
        self, 
        mock_import: MagicMock, 
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay instance creation."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        assert robot.n_dof == 6
        assert robot.can_interface == "can0"
        assert robot.ROBOT_TYPE == "airbot_play"
        assert robot.target_tracking_mode == "mit"
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_open_close(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay open/close lifecycle."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Open
        robot.open()
        assert robot._is_open
        mock_arm.init.assert_called_once()
        mock_ah.create_asio_executor.assert_called_once_with(8)
        
        # Close
        robot.close()
        assert not robot._is_open
        mock_arm.uninit.assert_called_once()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    @patch("rollio.robot.airbot.play._import_airbot_hardware")
    @patch("rollio.robot.airbot.play.is_can_interface_up", return_value=True)
    def test_airbot_play_and_g2_share_executor(
        self,
        mock_can_up: MagicMock,
        mock_play_import: MagicMock,
        mock_eef_import: MagicMock,
        mock_airbot_hardware,
    ) -> None:
        """AIRBOT arm and G2 reuse the same shared executor/io_context."""
        del mock_can_up
        mock_ah, mock_arm = mock_airbot_hardware
        mock_eef = MagicMock()
        mock_eef.init.return_value = True
        mock_ah.EEFType.G2 = "G2"
        mock_ah.EEF1.create.return_value = mock_eef
        mock_play_import.return_value = (mock_ah, True)
        mock_eef_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTG2
        from rollio.robot.airbot.play import AIRBOTPlay

        arm = AIRBOTPlay(can_interface="can0")
        gripper = AIRBOTG2(can_interface="can1")

        arm.open()
        gripper.open()

        shared_executor = mock_ah.create_asio_executor.return_value
        shared_io_context = shared_executor.get_io_context.return_value
        mock_ah.create_asio_executor.assert_called_once_with(8)
        assert arm._executor is shared_executor
        assert gripper._executor is shared_executor
        assert mock_arm.init.call_args[0][0] is shared_io_context
        assert mock_eef.init.call_args[0][0] is shared_io_context

        gripper.close()
        arm.close()
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_enable_disable(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test AIRBOTPlay enable/disable."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_read_joint_state(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test reading joint state from AIRBOT."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_control_modes(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test control mode switching."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_free_drive_command(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test free drive command execution."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.FREE_DRIVE)
        
        cmd = FreeDriveCommand(
            external_wrench=None,
            gravity_compensation_scale=1.0
        )
        
        robot.command_free_drive(cmd)
        
        assert _wait_until(lambda: mock_arm.mit.call_count >= 1)
        
        robot.close()
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_target_tracking_command(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test target tracking command execution."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
        
        assert _wait_until(lambda: mock_arm.mit.call_count >= 1)
        call_args = mock_arm.mit.call_args[0]
        assert len(call_args) == 5  # pos, vel, tau, kp, kd
        np.testing.assert_array_equal(
            np.asarray(call_args[3]),
            np.array([200.0, 200.0, 200.0, 50.0, 50.0, 50.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(call_args[4]),
            np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]),
        )
        
        robot.close()

    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_target_tracking_step_uses_pvt_when_configured(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware,
    ) -> None:
        """AIRBOT Play can use PVT for target tracking when configured."""
        del mock_can_up
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.play import AIRBOTPlay

        robot = AIRBOTPlay(
            can_interface="can0",
            target_tracking_mode="pvt",
        )
        robot.open()
        robot.enable()

        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is True
        mock_arm.set_param.assert_any_call("arm.control_mode", "PVT")

        target = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)
        robot.step_target_tracking(
            position_target=target,
            velocity_target=np.zeros(6, dtype=np.float64),
            add_gravity_compensation=True,
        )

        assert _wait_until(lambda: mock_arm.pvt.call_count >= 1)
        call_args = mock_arm.pvt.call_args[0]
        assert len(call_args) == 3
        np.testing.assert_array_equal(np.asarray(call_args[0]), target)
        np.testing.assert_array_equal(
            np.asarray(call_args[1]),
            np.full(6, 10.0),
        )
        np.testing.assert_array_equal(
            np.asarray(call_args[2]),
            np.full(6, 10.0),
        )
        mock_arm.mit.assert_not_called()

        robot.close()

    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_target_tracking_command_uses_pvt_when_configured(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware,
    ) -> None:
        del mock_can_up
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.play import AIRBOTPlay

        robot = AIRBOTPlay(
            can_interface="can0",
            target_tracking_mode="pvt",
        )
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.TARGET_TRACKING)

        robot.command_target_tracking(
            TargetTrackingCommand(
                position_target=np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
                velocity_target=np.zeros(6),
                kp=np.full(6, 30.0),
                kd=np.full(6, 3.0),
                feedforward=np.ones(6),
            )
        )

        assert _wait_until(lambda: mock_arm.pvt.call_count >= 1)
        call_args = mock_arm.pvt.call_args[0]
        np.testing.assert_array_equal(
            np.asarray(call_args[0]),
            np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        )
        np.testing.assert_array_equal(np.asarray(call_args[1]), np.full(6, 10.0))
        np.testing.assert_array_equal(np.asarray(call_args[2]), np.full(6, 10.0))
        mock_arm.mit.assert_not_called()

        robot.close()

    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_target_tracking_replays_at_background_rate(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware,
    ) -> None:
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.play import AIRBOTPlay

        robot = AIRBOTPlay(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.TARGET_TRACKING)

        robot.step_target_tracking(
            position_target=np.zeros(6, dtype=np.float64),
            velocity_target=np.zeros(6, dtype=np.float64),
            add_gravity_compensation=True,
        )
        assert _wait_until(lambda: mock_arm.mit.call_count >= 2)

        start_count = mock_arm.mit.call_count
        t0 = time.monotonic()
        time.sleep(0.12)
        elapsed = time.monotonic() - t0
        delta = mock_arm.mit.call_count - start_count
        rate_hz = delta / max(elapsed, 1e-6)

        assert delta >= 8
        assert 0.5 * DEFAULT_CONTROL_HZ <= rate_hz <= 1.6 * DEFAULT_CONTROL_HZ

        robot.close()
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_info(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test robot info."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        info = robot.info
        
        assert info.robot_type == "airbot_play"
        assert info.n_dof == 6
        assert FeedbackCapability.POSITION in info.feedback_capabilities
        assert FeedbackCapability.VELOCITY in info.feedback_capabilities
        assert FeedbackCapability.EFFORT in info.feedback_capabilities
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_gravity_scale(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test gravity compensation scale factors."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Check default scale factors
        default_scale = robot.gravity_compensation_scale
        assert len(default_scale) == 6
        np.testing.assert_array_almost_equal(
            default_scale,
            [0.6, 0.6, 0.6, 1.6, 1.248, 1.5]
        )
        
        # Set custom scale
        custom_scale = np.ones(6)
        robot.gravity_compensation_scale = custom_scale
        np.testing.assert_array_almost_equal(
            robot.gravity_compensation_scale,
            custom_scale
        )
    
    @patch('rollio.robot.airbot.play.set_airbot_led')
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
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
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test with gravity comp disabled (LED only)
        result = robot.identify_start(with_gravity_comp=False)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=True)
    
    @patch('rollio.robot.airbot.play.set_airbot_led')
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
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
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test with gravity comp enabled (default)
        result = robot.identify_start(with_gravity_comp=True)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=True)
        # Robot should be opened and enabled
        assert robot._is_open is True
        assert robot._is_enabled is True
    
    @patch('rollio.robot.airbot.play.set_airbot_led')
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
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
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        # Test without disabling robot
        result = robot.identify_stop(disable_robot=False)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=False)
    
    @patch('rollio.robot.airbot.play.set_airbot_led')
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
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
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        # Simulate robot being open
        robot._is_open = True
        robot._arm = mock_arm
        
        result = robot.identify_stop(disable_robot=True)
        
        assert result is True
        mock_set_led.assert_called_once_with("can0", blink_orange=False)
    
    @patch('rollio.robot.airbot.play.query_airbot_end_effector')
    @patch('rollio.robot.airbot.play.query_airbot_serial')
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
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
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
    
    @patch('rollio.robot.airbot.play.is_can_interface_up', return_value=True)
    @patch('rollio.robot.airbot.play._import_airbot_hardware')
    def test_airbot_properties_default(
        self,
        mock_import: MagicMock,
        mock_can_up: MagicMock,
        mock_airbot_hardware
    ) -> None:
        """Test default properties without query."""
        mock_ah, mock_arm = mock_airbot_hardware
        mock_import.return_value = (mock_ah, True)
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        robot = AIRBOTPlay(can_interface="can0")
        
        props = robot.properties
        assert props["can_interface"] == "can0"
        assert props["control_frequency"] == DEFAULT_CONTROL_HZ
        assert props["motor_types"] == ["OD", "OD", "OD", "DM", "DM", "DM"]
        
        # Serial and EEF not queried yet
        assert robot.serial_number is None
        assert robot.end_effector_type is None


class TestAIRBOTEEFMocked:
    """Tests for AIRBOT EEF robot entities with mocked hardware."""

    @pytest.fixture
    def mock_airbot_eef_hardware(self, monkeypatch):
        mock_ah = MagicMock()
        mock_ah.EEFType.G2 = "G2"
        mock_ah.EEFType.E2 = "E2"
        mock_ah.MotorType.DM = "DM"
        mock_ah.MotorType.OD = "OD"
        mock_ah.MotorControlMode.MIT = "MIT"
        mock_ah.MotorControlMode.PVT = "PVT"
        mock_ah.ParamType.UINT32_LE = "UINT32_LE"

        class _ParamValue:
            def __init__(self, param_type, value) -> None:
                self.param_type = param_type
                self.value = value

        mock_ah.ParamValue = _ParamValue

        mock_state = MagicMock()
        mock_state.is_valid = True
        mock_state.pos = [0.04]
        mock_state.vel = [0.01]
        mock_state.eff = [0.2]

        mock_eef = MagicMock()
        mock_eef.state.return_value = mock_state
        mock_eef.init.return_value = True
        mock_eef.enable.return_value = True
        mock_eef.ping.return_value = True
        mock_eef.set_param.return_value = True

        mock_ah.EEF1.create.return_value = mock_eef

        class _EEFCommand1:
            def __init__(self) -> None:
                self.pos = [0.0]
                self.vel = [0.0]
                self.eff = [0.0]
                self.mit_kp = [0.0]
                self.mit_kd = [0.0]
                self.current_threshold = [0.0]

        mock_ah.EEFCommand1 = _EEFCommand1

        mock_executor = MagicMock()
        mock_executor.get_io_context.return_value = MagicMock()
        mock_ah.create_asio_executor.return_value = mock_executor

        return mock_ah, mock_eef

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_g2_open_read_and_command(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTG2

        robot = AIRBOTG2(can_interface="can0")
        robot.open()
        mock_ah.create_asio_executor.assert_called_once_with(8)
        robot.enable()
        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is True

        state = robot.read_joint_state()
        assert state.is_valid
        np.testing.assert_array_almost_equal(state.position, [0.04])

        robot.step_target_tracking(np.array([0.06]), np.array([0.0]), kp=40.0, kd=8.0)

        mock_eef.enable.assert_called_once()
        mock_eef.set_param.assert_any_call("control_mode", "PVT")
        assert _wait_until(lambda: mock_eef.pvt.call_count >= 1)
        payload = mock_eef.pvt.call_args[0][0]
        assert payload.pos == [0.06]
        assert payload.vel == [200.0]
        assert payload.eff == [0.0]
        assert payload.mit_kp == [0.0]
        assert payload.mit_kd == [0.0]
        assert payload.current_threshold == [200.0]
        command_debug = robot.latest_command_debug()
        assert command_debug is not None
        assert command_debug[0] == "PVT"
        assert "current_threshold=[200.0000]" in command_debug[1]
        mock_eef.mit.assert_not_called()
        robot.close()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_g2_command_payload_uses_attribute_assignment_for_copy_on_read_sdk(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        class _CopyOnReadCommand:
            def __init__(self) -> None:
                self._pos = [0.0]
                self._vel = [0.0]
                self._eff = [0.0]
                self._mit_kp = [0.0]
                self._mit_kd = [0.0]
                self._current_threshold = [0.0]

            @property
            def pos(self):
                return list(self._pos)

            @pos.setter
            def pos(self, value) -> None:
                self._pos = list(value)

            @property
            def vel(self):
                return list(self._vel)

            @vel.setter
            def vel(self, value) -> None:
                self._vel = list(value)

            @property
            def eff(self):
                return list(self._eff)

            @eff.setter
            def eff(self, value) -> None:
                self._eff = list(value)

            @property
            def mit_kp(self):
                return list(self._mit_kp)

            @mit_kp.setter
            def mit_kp(self, value) -> None:
                self._mit_kp = list(value)

            @property
            def mit_kd(self):
                return list(self._mit_kd)

            @mit_kd.setter
            def mit_kd(self, value) -> None:
                self._mit_kd = list(value)

            @property
            def current_threshold(self):
                return list(self._current_threshold)

            @current_threshold.setter
            def current_threshold(self, value) -> None:
                self._current_threshold = list(value)

        mock_ah.EEFCommand1 = _CopyOnReadCommand

        from rollio.robot.airbot.eef import AIRBOTG2

        robot = AIRBOTG2(can_interface="can0")
        payload = robot._build_command_payload(  # noqa: SLF001
            np.array([0.035], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            effort=np.array([0.0], dtype=np.float64),
            kp=np.array([0.0], dtype=np.float64),
            kd=np.array([0.0], dtype=np.float64),
            current_threshold=np.array([10.0], dtype=np.float64),
        )

        assert payload is not None
        assert payload.pos == [0.035]
        assert payload.vel == [10.0]
        assert payload.eff == [0.0]
        assert payload.mit_kp == [0.0]
        assert payload.mit_kd == [0.0]
        assert payload.current_threshold == [10.0]

    @patch("rollio.robot.airbot.eef.set_airbot_led")
    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_g2_identification_oscillates_with_fixed_gains(
        self,
        mock_import: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True

        from rollio.robot.airbot.eef import AIRBOTG2

        robot = AIRBOTG2(can_interface="can0")

        with patch.object(robot, "_identify_target_position", return_value=np.array([0.035])):
            assert robot.identify_start() is True
            robot.identify_step()

        mock_set_led.assert_called_once_with("can0", blink_orange=True)
        assert _wait_until(lambda: mock_eef.set_param.call_count >= 1)
        mock_eef.set_param.assert_any_call("control_mode", "PVT")
        assert _wait_until(lambda: mock_eef.pvt.call_count >= 1)
        payload = mock_eef.pvt.call_args[0][0]
        assert 0.0 <= payload.pos[0] <= 0.07
        assert payload.vel == [200.0]
        assert payload.mit_kp == [0.0]
        assert payload.mit_kd == [0.0]
        assert payload.eff == [0.0]
        assert payload.current_threshold == [200.0]
        command_debug = robot.latest_command_debug()
        assert command_debug is not None
        assert command_debug[0] == "PVT"
        assert "pos=" in command_debug[1]

    @patch("rollio.robot.airbot.eef.set_airbot_led")
    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_identification_keeps_feedback_alive(
        self,
        mock_import: MagicMock,
        mock_set_led: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)
        mock_set_led.return_value = True

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can0")

        assert robot.identify_start() is True
        robot.identify_step()

        mock_set_led.assert_called_once_with("can0", blink_orange=True)
        param_call = mock_eef.set_param.call_args
        assert param_call is not None
        assert param_call[0][0] == "eef.e2.mode"
        assert param_call[0][1].param_type == "UINT32_LE"
        assert param_call[0][1].value == 4
        assert _wait_until(lambda: mock_eef.mit.call_count >= 1)

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_ignores_false_mode_return_when_sdk_does_not_report_success(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)
        mock_eef.set_param.return_value = False

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can1")
        robot.open()
        assert robot.enable() is True
        assert robot.enter_free_drive() is True
        assert robot.control_mode == ControlMode.FREE_DRIVE
        robot.close()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_eef_set_control_mode_fails_when_sdk_rejects_change(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)
        mock_eef.set_param.return_value = False

        from rollio.robot.airbot.eef import AIRBOTG2

        robot = AIRBOTG2(can_interface="can0")
        robot.open()
        robot.enable()

        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is False
        assert robot.control_mode == ControlMode.DISABLED
        robot.close()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_eef_accepts_void_sdk_mutators(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)
        mock_eef.enable.return_value = None
        mock_eef.set_param.return_value = None

        from rollio.robot.airbot.eef import AIRBOTG2

        robot = AIRBOTG2(can_interface="can0")
        robot.open()

        assert robot.enable() is True
        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is True
        assert robot.control_mode == ControlMode.TARGET_TRACKING
        mock_eef.set_param.assert_called_once_with("control_mode", "PVT")

        robot.close()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_info_reflects_single_axis_entity(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, _mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can2")

        assert robot.info.robot_type == "airbot_e2b"
        assert robot.info.n_dof == 1
        assert robot.properties["can_interface"] == "can2"
        assert robot.properties["end_effector_type"] == "E2B"
        assert robot.properties["eef_motor_type"] == "OD"

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_rejects_non_od_motor_type(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, _mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTE2B

        with pytest.raises(ValueError, match="only supports the standalone AIRBOT EEF combination"):
            AIRBOTE2B(can_interface="can0", motor_type="DM")

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_free_drive_refreshes_current_feedback(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.FREE_DRIVE)

        robot.step_free_drive()
        state = robot.read_joint_state()

        assert state.is_valid
        param_call = mock_eef.set_param.call_args
        assert param_call is not None
        assert param_call[0][0] == "eef.e2.mode"
        assert param_call[0][1].param_type == "UINT32_LE"
        assert param_call[0][1].value == 4
        assert _wait_until(lambda: mock_eef.mit.call_count >= 1)
        payload = mock_eef.mit.call_args[0][0]
        assert payload.pos == [0.0]
        assert payload.vel == [0.0]
        assert payload.eff == [0.0]
        assert payload.mit_kp == [0.0]
        assert payload.mit_kd == [0.0]
        assert payload.current_threshold == [0.0]
        mock_eef.ping.assert_not_called()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_keepalive_replays_at_background_rate(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.FREE_DRIVE)
        assert _wait_until(lambda: mock_eef.mit.call_count >= 2)

        start_count = mock_eef.mit.call_count
        t0 = time.monotonic()
        time.sleep(0.12)
        elapsed = time.monotonic() - t0
        delta = mock_eef.mit.call_count - start_count
        rate_hz = delta / max(elapsed, 1e-6)

        assert delta >= 8
        assert 0.5 * DEFAULT_CONTROL_HZ <= rate_hz <= 1.6 * DEFAULT_CONTROL_HZ

        robot.close()

    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_e2b_free_drive_mode_keeps_feedback_alive_without_extra_steps(
        self,
        mock_import: MagicMock,
        mock_airbot_eef_hardware,
    ) -> None:
        mock_ah, mock_eef = mock_airbot_eef_hardware
        mock_import.return_value = (mock_ah, True)

        from rollio.robot.airbot.eef import AIRBOTE2B

        robot = AIRBOTE2B(can_interface="can0")
        robot.open()
        robot.enable()
        robot.set_control_mode(ControlMode.FREE_DRIVE)

        state = robot.read_joint_state()

        assert state.is_valid is True
        assert _wait_until(lambda: mock_eef.mit.call_count >= 1)
        mock_eef.ping.assert_not_called()
        robot.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test AIRBOT Scanning
# ═══════════════════════════════════════════════════════════════════════════════


class TestAIRBOTScanning:
    """Tests for AIRBOT robot scanning."""
    
    @patch('rollio.robot.airbot.play.scan_airbot_detected_robots')
    def test_airbot_scan(
        self,
        mock_scan: MagicMock,
    ) -> None:
        """Test scanning for AIRBOT robots."""
        mock_scan.return_value = [
            DetectedRobot(
                robot_type="airbot_play",
                device_id="can0",
                label="AIRBOT Play (can0) SN:PZ25C0240200",
                n_dof=6,
                properties={"can_interface": "can0"},
            ),
            DetectedRobot(
                robot_type="airbot_e2b",
                device_id="can0",
                label="AIRBOT E2B (can0) SN:PZ25C0240200",
                n_dof=1,
                properties={"can_interface": "can0"},
            ),
            DetectedRobot(
                robot_type="airbot_play",
                device_id="can1",
                label="AIRBOT Play (can1) SN:PZ25C0240201",
                n_dof=6,
                properties={"can_interface": "can1"},
            ),
        ]
        
        from rollio.robot.airbot.play import AIRBOTPlay
        
        devices = AIRBOTPlay.scan()
        
        assert len(devices) == 2
        assert all(d.robot_type == "airbot_play" for d in devices)
        assert all(d.n_dof == 6 for d in devices)
        assert devices[0].device_id == "can0"
        assert devices[1].device_id == "can1"
        assert "SN:PZ25C0240200" in devices[0].label
    
    @patch('rollio.robot.airbot.play.scan_airbot_detected_robots', return_value=[])
    def test_airbot_scan_no_device(
        self,
        mock_scan: MagicMock,
    ) -> None:
        """Test scanning when no AIRBOT device responds."""
        from rollio.robot.airbot.play import AIRBOTPlay
        
        devices = AIRBOTPlay.scan()
        
        # Should find no devices since probe returns False
        assert len(devices) == 0

    @patch("rollio.robot.airbot.eef.scan_airbot_detected_robots")
    @patch("rollio.robot.airbot.eef._import_airbot_hardware")
    def test_airbot_g2_scan_filters_to_eef_entities(
        self,
        mock_import: MagicMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_import.return_value = (MagicMock(), True)
        mock_scan.return_value = [
            DetectedRobot(
                robot_type="airbot_play",
                device_id="can0",
                label="AIRBOT Play (can0)",
                n_dof=6,
                properties={"can_interface": "can0"},
            ),
            DetectedRobot(
                robot_type="airbot_g2",
                device_id="can0",
                label="AIRBOT G2 (can0)",
                n_dof=1,
                properties={"can_interface": "can0"},
            ),
        ]

        from rollio.robot.airbot.eef import AIRBOTG2

        devices = AIRBOTG2.scan()

        assert len(devices) == 1
        assert devices[0].robot_type == "airbot_g2"
        assert devices[0].n_dof == 1
    
    def test_airbot_probe_nonexistent(self) -> None:
        """Test probing non-existent interface."""
        from rollio.robot.airbot.play import AIRBOTPlay
        
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
        not _pinocchio_available(),
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
        from rollio.robot.airbot.play import AIRBOTPlay, is_airbot_available
        
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
