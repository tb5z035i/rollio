"""AIRBOT Play robot arm implementation.

This module provides the robot arm implementation for AIRBOT Play series
robots, which communicate via SocketCAN interface.

The AIRBOT Play is a 6-DOF collaborative robot arm with:
- Joints 0-2: OD (high-torque) motors for shoulder/elbow
- Joints 3-5: DM motors for wrist
- Optional end-effector (gripper)

Control modes:
- MIT mode: Direct torque control with PD gains (free drive, impedance control)
- PVT mode: Position-velocity-torque trajectory tracking
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rollio.robot.base import (
    ControlMode,
    FeedbackCapability,
    FreeDriveCommand,
    JointState,
    KinematicsModel,
    RobotArm,
    RobotInfo,
    TargetTrackingCommand,
    Wrench,
)
from rollio.robot.can_utils import (
    is_can_interface_up,
    is_python_can_available,
    probe_airbot_device,
    query_airbot_end_effector,
    query_airbot_gravity_coefficients,
    query_airbot_properties,
    query_airbot_serial,
    scan_can_interfaces,
    set_airbot_led,
)
from rollio.robot.scanner import DetectedRobot
from rollio.utils.time import monotonic_sec

# Lazy imports for optional dependencies
_ah = None
_AH_AVAILABLE = None


def _import_airbot_hardware():
    """Lazy import airbot_hardware_py."""
    global _ah, _AH_AVAILABLE
    if _AH_AVAILABLE is None:
        try:
            import airbot_hardware_py as ah
            _ah = ah
            _AH_AVAILABLE = True
        except ImportError:
            _AH_AVAILABLE = False
    return _ah, _AH_AVAILABLE


def is_airbot_available() -> bool:
    """Check if airbot_hardware_py is available."""
    _, available = _import_airbot_hardware()
    return available


# ═══════════════════════════════════════════════════════════════════════════════
# AIRBOT Kinematics Wrapper (applies per-joint EEF gravity coefficients)
# ═══════════════════════════════════════════════════════════════════════════════


class AIRBOTKinematicsModel(KinematicsModel):
    """Wraps any ``KinematicsModel`` and applies AIRBOT-specific per-joint
    gravity compensation coefficients.

    All methods delegate to the inner model unchanged **except**
    ``gravity_compensation`` which scales the raw inverse-dynamics result
    by the EEF-dependent coefficients.  This means every caller — free
    drive, target tracking, ``step_target_tracking``, ``move_to_position``
    — automatically gets the correct scaled torques.
    """

    def __init__(self, inner: KinematicsModel, arm: "AIRBOTPlay") -> None:
        self._inner = inner
        self._arm = arm  # back-reference for coefficients lookup

    # ── delegate all abstract properties / methods ─────────────────────

    @property
    def n_dof(self) -> int:
        return self._inner.n_dof

    @property
    def end_effector_names(self) -> list[str]:
        return self._inner.end_effector_names

    def forward_kinematics(self, q, end_effector=None):
        return self._inner.forward_kinematics(q, end_effector)

    def inverse_kinematics(self, target_pose, q_init=None,
                           end_effector=None, max_iterations=100,
                           tolerance=1e-6):
        return self._inner.inverse_kinematics(
            target_pose, q_init, end_effector, max_iterations, tolerance)

    def jacobian(self, q, end_effector=None):
        return self._inner.jacobian(q, end_effector)

    def inverse_dynamics(self, q, qd, qdd):
        return self._inner.inverse_dynamics(q, qd, qdd)

    # ── AIRBOT-specific override ──────────────────────────────────────

    def gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """Gravity compensation with AIRBOT per-joint EEF coefficients.

        Returns ``inverse_dynamics(q, 0, 0) * coefficients`` where
        *coefficients* are looked up from the robot's current end-effector
        type (or user override).
        """
        tau = np.asarray(self._inner.gravity_compensation(q), dtype=np.float64)
        tau *= self._arm._get_gravity_coefficients_for_eef()
        return tau

    # ── expose inner model for callers that need Pinocchio specifics ──

    @property
    def inner(self) -> KinematicsModel:
        """Access the unwrapped kinematics model (e.g. PinocchioKinematicsModel)."""
        return self._inner


# ═══════════════════════════════════════════════════════════════════════════════
# AIRBOT Play Robot Arm
# ═══════════════════════════════════════════════════════════════════════════════


class AIRBOTPlay(RobotArm):
    """AIRBOT Play 6-DOF collaborative robot arm.
    
    This robot communicates via SocketCAN and supports:
    - Free drive mode with gravity compensation
    - Target tracking mode with PD control
    - MIT mode for direct torque control
    - PVT mode for trajectory tracking
    
    Args:
        can_interface: CAN interface name (e.g., "can0")
        urdf_path: Path to URDF file for kinematics (optional)
        control_frequency: Control loop frequency in Hz (default: 250)
        gravity_coefficients: Per-joint gravity compensation coefficients override.
            If None, coefficients are read from hardware and selected based on
            the detected end effector type.
    """
    
    ROBOT_TYPE = "airbot_play"
    N_DOF = 6
    
    # Arm joint names (joints 1-6, excludes gripper joints)
    ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    
    # Default gravity coefficients (fallback if CAN query fails)
    DEFAULT_GRAVITY_COEFFICIENTS = {
        "none": np.array([0.6, 0.6, 0.6, 1.6, 1.248, 1.5]),
        "E2B": np.array([0.6, 0.6, 0.6, 1.338, 1.236, 0.893]),
        "G2": np.array([0.6, 0.6, 0.6, 1.303, 1.181, 1.5]),
        "other": np.array([0.6, 0.6, 0.6, 1.5, 1.5, 1.5]),
    }
    
    # ── Class methods ─────────────────────────────────────────────────────
    
    @classmethod
    def scan(cls) -> list[DetectedRobot]:
        """Scan for available AIRBOT Play robots via CAN interfaces.
        
        Probes each CAN interface by sending the identify command (0x000#07)
        and checking for the expected AIRBOT response pattern.
        Also queries serial number and end effector type.
        """
        if not is_airbot_available():
            return []
        
        found = []
        for iface in scan_can_interfaces():
            if is_can_interface_up(iface):
                # Probe the interface to verify it's an AIRBOT device
                if probe_airbot_device(iface, timeout=0.5):
                    # Query additional properties (SN, end effector)
                    props = query_airbot_properties(iface, timeout=0.5)
                    props["motor_types"] = ["OD", "OD", "OD", "DM", "DM", "DM"]
                    
                    # Build label with serial number if available
                    sn = props.get("serial_number", "")
                    label = f"AIRBOT Play ({iface})"
                    if sn:
                        label = f"AIRBOT Play ({iface}) SN:{sn}"
                    
                    found.append(DetectedRobot(
                        robot_type=cls.ROBOT_TYPE,
                        device_id=iface,
                        label=label,
                        n_dof=cls.N_DOF,
                        properties=props
                    ))
        
        return found
    
    @classmethod
    def probe(cls, device_id: int | str) -> bool:
        """Check if an AIRBOT robot exists on the given CAN interface.
        
        Sends the identify command (0x000#07) and verifies the response
        pattern that identifies an AIRBOT arm-interface-board.
        """
        if not is_airbot_available():
            return False
        
        interface = str(device_id)
        if not is_can_interface_up(interface):
            return False
        
        # Probe with AIRBOT-specific identify protocol
        return probe_airbot_device(interface, timeout=1.0)
    
    # ── Instance initialization ───────────────────────────────────────────
    
    def __init__(
        self,
        can_interface: str = "can0",
        urdf_path: str | Path | None = None,
        control_frequency: int = 250,
        gravity_coefficients: np.ndarray | None = None,
        end_effector_frame: str | None = None,
    ) -> None:
        ah, available = _import_airbot_hardware()
        if not available:
            raise ImportError(
                "airbot_hardware_py is required for AIRBOTPlay. "
                "Install the AIRBOT SDK."
            )
        
        self._ah = ah
        self._can_interface = can_interface
        self._control_frequency = control_frequency
        self._dt = 1.0 / control_frequency
        
        # State
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
        self._arm = None
        self._executor = None
        
        # Kinematics model (lazy loaded)
        self._kinematics: KinematicsModel | None = None
        self._urdf_path = urdf_path
        self._ee_frame = end_effector_frame
        
        # Robot properties (cached, populated by query_properties())
        self._properties: dict[str, Any] = {
            "can_interface": can_interface,
            "control_frequency": control_frequency,
            "motor_types": ["OD", "OD", "OD", "DM", "DM", "DM"],
        }
        
        # Query hardware properties (end effector type, gravity coefficients)
        self._query_end_effector_type()
        self._query_gravity_coefficients()
        
        # User override for gravity coefficients (None = use auto-detected)
        if gravity_coefficients is not None:
            self._properties["gravity_coefficients_override"] = np.asarray(
                gravity_coefficients, dtype=np.float64
            )
        
        # Info
        self._info = RobotInfo(
            name=f"airbot_play_{can_interface}",
            robot_type=self.ROBOT_TYPE,
            n_dof=self.N_DOF,
            feedback_capabilities={
                FeedbackCapability.POSITION,
                FeedbackCapability.VELOCITY,
                FeedbackCapability.EFFORT,
                FeedbackCapability.END_POSE,
                FeedbackCapability.END_TWIST,
            },
            properties=self._properties
        )
    
    def _load_kinematics(self) -> KinematicsModel:
        """Load kinematics model from URDF.
        
        Uses Pinocchio if available, with the arm joints locked to exclude
        gripper joints from the dynamics model.
        
        The returned model is wrapped in ``AIRBOTKinematicsModel`` so that
        ``gravity_compensation()`` automatically applies the per-joint EEF
        coefficients.  This means **all** callers (free drive, target
        tracking, ``step_target_tracking``, ``move_to_position``, …) get
        the correct AIRBOT-specific gravity torques without needing to know
        about coefficients.
        """
        from rollio.robot.pinocchio_kinematics import (
            PinocchioKinematicsModel,
            get_bundled_urdf,
            is_pinocchio_available,
        )
        from rollio.robot.pseudo_robot import PseudoKinematicsModel
        
        base_model: KinematicsModel | None = None
        
        # Try to load Pinocchio model
        if is_pinocchio_available():
            urdf_path = self._urdf_path
            if urdf_path is None:
                # Try bundled URDF first
                urdf_path = get_bundled_urdf("play_e2")
                
                # Fall back to other common paths
                if urdf_path is None:
                    for path in [
                        Path.home() / ".airbot/urdf/play_e2.urdf",
                        "/usr/share/airbot/urdf/play_e2.urdf",
                    ]:
                        if Path(path).exists():
                            urdf_path = path
                            break
            
            if urdf_path and Path(urdf_path).exists():
                base_model = PinocchioKinematicsModel(
                    urdf_path=urdf_path,
                    end_effector_frame=self._ee_frame,
                    arm_joints=self.ARM_JOINTS,  # Lock gripper joints
                )
        
        if base_model is None:
            # Fall back to pseudo kinematics
            import warnings
            warnings.warn(
                "Pinocchio not available or URDF not found. "
                "Using simplified pseudo kinematics model. "
                "Install Pinocchio for accurate kinematics: pip install pin"
            )
            base_model = PseudoKinematicsModel(n_dof=self.N_DOF)
        
        # Wrap with AIRBOT-specific gravity coefficient scaling
        return AIRBOTKinematicsModel(base_model, self)
    
    def _query_gravity_coefficients(self) -> None:
        """Query gravity compensation coefficients from hardware via CAN.
        
        Sends command 0x000#17 to read coefficients for all end effector types.
        Falls back to default values if query fails.
        
        Stores result in _properties["gravity_coefficients_by_eef"].
        """
        coefficients = None
        try:
            result = query_airbot_gravity_coefficients(
                self._can_interface, 
                timeout=0.5
            )
            if result:
                # Convert lists to numpy arrays
                coefficients = {
                    eef_type: np.array(coeffs, dtype=np.float64)
                    for eef_type, coeffs in result.items()
                }
        except Exception:
            pass
        
        # Fall back to default values if query failed
        if coefficients is None:
            coefficients = {
                eef_type: coeffs.copy()
                for eef_type, coeffs in self.DEFAULT_GRAVITY_COEFFICIENTS.items()
            }
        
        self._properties["gravity_coefficients_by_eef"] = coefficients
    
    def _query_end_effector_type(self) -> None:
        """Query and cache the end effector type from hardware."""
        try:
            eef = query_airbot_end_effector(self._can_interface, timeout=0.5)
            if eef:
                self._properties["end_effector_type"] = eef["type_name"]
                self._properties["end_effector_code"] = eef["type_code"]
        except Exception:
            pass
    
    def _get_gravity_coefficients_for_eef(self, eef_type: str | None = None) -> np.ndarray:
        """Get gravity compensation coefficients for the given end effector type.
        
        Args:
            eef_type: End effector type name ("none", "E2B", "G2", "other").
                     If None, uses the current end effector type from properties.
        
        Returns:
            Numpy array of 6 gravity compensation coefficients
        """
        # If user provided explicit override, use that
        override = self._properties.get("gravity_coefficients_override")
        if override is not None:
            return override
        
        # Determine end effector type
        if eef_type is None:
            eef_type = self._properties.get("end_effector_type", "none")
        
        # Normalize the type name
        eef_type_normalized = eef_type.lower() if eef_type else "none"
        
        # Map to known types
        if eef_type_normalized in ("none", "na", ""):
            key = "none"
        elif eef_type_normalized == "e2b":
            key = "E2B"
        elif eef_type_normalized == "g2":
            key = "G2"
        else:
            key = "other"
        
        # Return coefficients for this EEF type
        coefficients_by_eef = self._properties.get("gravity_coefficients_by_eef", {})
        if key in coefficients_by_eef:
            return coefficients_by_eef[key]
        
        # Fallback to default for the detected EEF type (or "none" if unavailable)
        return self.DEFAULT_GRAVITY_COEFFICIENTS.get(
            key, self.DEFAULT_GRAVITY_COEFFICIENTS["none"]
        ).copy()
    
    # ── Properties ────────────────────────────────────────────────────────
    
    @property
    def n_dof(self) -> int:
        return self.N_DOF
    
    @property
    def info(self) -> RobotInfo:
        return self._info
    
    @property
    def kinematics(self) -> KinematicsModel:
        if self._kinematics is None:
            self._kinematics = self._load_kinematics()
        return self._kinematics
    
    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode
    
    @property
    def is_enabled(self) -> bool:
        return self._is_enabled
    
    @property
    def can_interface(self) -> str:
        """Get the CAN interface name."""
        return self._can_interface
    
    @property
    def gravity_coefficients(self) -> np.ndarray:
        """Get the per-joint gravity compensation coefficients.
        
        Returns the coefficients for the current end effector type,
        or the user-provided override if set.
        """
        return self._get_gravity_coefficients_for_eef()
    
    @gravity_coefficients.setter
    def gravity_coefficients(self, coefficients: np.ndarray | None) -> None:
        """Set the per-joint gravity compensation coefficients.
        
        Args:
            coefficients: Override coefficients, or None to use auto-detected
                         coefficients based on end effector type.
        """
        if coefficients is not None:
            self._properties["gravity_coefficients_override"] = np.asarray(
                coefficients, dtype=np.float64
            )
        else:
            self._properties.pop("gravity_coefficients_override", None)
    
    @property
    def gravity_coefficients_by_eef(self) -> dict[str, np.ndarray]:
        """Get all gravity compensation coefficients by end effector type.
        
        Returns:
            Dictionary mapping end effector type to numpy array of 6 coefficients
        """
        coefficients = self._properties.get("gravity_coefficients_by_eef", {})
        return {k: v.copy() for k, v in coefficients.items()}
    
    @property
    def properties(self) -> dict[str, Any]:
        """Get robot properties dictionary.
        
        Returns cached properties. Call query_properties() to refresh
        from hardware.
        
        Common properties:
        - "can_interface": CAN interface name
        - "serial_number": Robot serial number (after query_properties)
        - "end_effector_type": Attached end effector type (after query_properties)
        - "control_frequency": Control loop frequency
        - "motor_types": List of motor types per joint
        """
        return self._properties.copy()
    
    def query_properties(self) -> dict[str, Any]:
        """Query robot properties from hardware via CAN.
        
        Queries:
        - Serial number (via 0x000#04)
        - End effector type (via 0x008#05)
        
        Returns:
            Updated properties dictionary
        """
        # Query serial number
        serial = query_airbot_serial(self._can_interface, timeout=0.5)
        if serial:
            self._properties["serial_number"] = serial
        
        # Query end effector
        eef = query_airbot_end_effector(self._can_interface, timeout=0.5)
        if eef:
            self._properties["end_effector_type"] = eef["type_name"]
            self._properties["end_effector_code"] = eef["type_code"]
        
        # Update info's properties reference
        self._info = RobotInfo(
            name=self._info.name,
            robot_type=self._info.robot_type,
            n_dof=self._info.n_dof,
            feedback_capabilities=self._info.feedback_capabilities,
            properties=self._properties
        )
        
        return self._properties.copy()
    
    @property
    def serial_number(self) -> str | None:
        """Get robot serial number (cached, call query_properties() to refresh)."""
        return self._properties.get("serial_number")
    
    @property
    def end_effector_type(self) -> str | None:
        """Get end effector type name (cached, call query_properties() to refresh)."""
        return self._properties.get("end_effector_type")
    
    # ── Lifecycle ─────────────────────────────────────────────────────────
    
    def open(self) -> None:
        """Initialize connection to AIRBOT arm via CAN."""
        if self._is_open:
            return
        
        # Check CAN interface
        if not is_can_interface_up(self._can_interface):
            raise RuntimeError(
                f"CAN interface '{self._can_interface}' is not available or not UP. "
                f"Run: sudo ip link set {self._can_interface} up type can bitrate 1000000"
            )
        
        # Create ASIO executor
        self._executor = self._ah.create_asio_executor(1)
        io_context = self._executor.get_io_context()
        
        # Create arm with motor configuration:
        # Joints 0-2: OD motors (high torque)
        # Joints 3-5: DM motors (wrist)
        # No end-effector, no gripper motor
        self._arm = self._ah.Play.create(
            self._ah.MotorType.OD,
            self._ah.MotorType.OD,
            self._ah.MotorType.OD,
            self._ah.MotorType.DM,
            self._ah.MotorType.DM,
            self._ah.MotorType.DM,
            self._ah.EEFType.NA,
            self._ah.MotorType.NA,
        )
        
        # Initialize arm
        if not self._arm.init(io_context, self._can_interface, self._control_frequency):
            self._arm = None
            self._executor = None
            raise RuntimeError(
                f"Failed to initialize AIRBOT arm on {self._can_interface}. "
                "Check CAN connection and motor power."
            )
        
        self._is_open = True
    
    def close(self) -> None:
        """Close connection to AIRBOT arm."""
        if not self._is_open:
            return
        
        if self._is_enabled:
            self.disable()
        
        if self._arm is not None:
            self._arm.uninit()
            self._arm = None
        
        self._executor = None
        self._is_open = False
    
    def enable(self) -> bool:
        """Enable AIRBOT motors."""
        if not self._is_open or self._arm is None:
            return False
        
        self._arm.enable()
        self._is_enabled = True
        return True
    
    def disable(self) -> None:
        """Disable AIRBOT motors (safe state)."""
        if self._arm is not None and self._is_enabled:
            # First, switch to PVT mode and go to safe position if needed
            if self._control_mode == ControlMode.FREE_DRIVE:
                self._arm.set_param(
                    "arm.control_mode", 
                    self._ah.MotorControlMode.PVT
                )
            self._arm.disable()
        
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
    
    # ── State Reading ─────────────────────────────────────────────────────
    
    def read_joint_state(self) -> JointState:
        """Read current joint state from AIRBOT arm."""
        ts = monotonic_sec()
        
        if self._arm is None or not self._is_open:
            return JointState(
                timestamp=ts,
                position=None,
                velocity=None,
                effort=None,
                is_valid=False
            )
        
        state = self._arm.state()
        
        if not state.is_valid:
            return JointState(
                timestamp=ts,
                position=None,
                velocity=None,
                effort=None,
                is_valid=False
            )
        
        return JointState(
            timestamp=ts,
            position=np.array(state.pos, dtype=np.float32),
            velocity=np.array(state.vel, dtype=np.float32),
            effort=np.array(state.eff, dtype=np.float32),
            is_valid=True
        )
    
    # ── Control Mode Setting ──────────────────────────────────────────────
    
    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode."""
        if not self._is_enabled or self._arm is None:
            return False
        
        if mode == ControlMode.FREE_DRIVE:
            # MIT mode for free drive (torque control)
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.MIT
            )
        elif mode == ControlMode.TARGET_TRACKING:
            # MIT mode for target tracking (PD + feedforward)
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.MIT
            )
        elif mode == ControlMode.DISABLED:
            # PVT mode for safe state
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.PVT
            )
        else:
            return False
        
        self._control_mode = mode
        return True
    
    # ── Free Drive Mode ───────────────────────────────────────────────────
    
    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        """Execute free drive command on AIRBOT arm.
        
        In free drive mode:
        1. Compute gravity compensation torques (EEF coefficients are already
           baked into ``self.kinematics.gravity_compensation()`` via the
           ``AIRBOTKinematicsModel`` wrapper)
        2. Apply the caller's gravity scale
        3. If external wrench is provided, transform to joint torques via Jacobian
        4. Send combined torques via MIT mode with zero position/velocity targets
        """
        if self._control_mode != ControlMode.FREE_DRIVE or self._arm is None:
            return
        
        # Read current joint state
        state = self._arm.state()
        if not state.is_valid:
            return
        
        q = np.array(state.pos)
        
        # Gravity compensation (already includes AIRBOT EEF coefficients)
        tau_gravity = np.array(
            self.kinematics.gravity_compensation(q),
            dtype=np.float64,
        )
        tau_gravity *= cmd.gravity_compensation_scale
        
        # Compute external wrench contribution
        tau_ext = np.zeros(self.N_DOF)
        if cmd.external_wrench is not None:
            tau_ext = self.kinematics.wrench_to_joint_torques(
                q, cmd.external_wrench
            )
        
        # Combined torques
        tau_total = tau_gravity + tau_ext
        
        # Send MIT command: position=0, velocity=0, torque=tau, kp=0, kd=0
        self._arm.mit(
            [0.0] * self.N_DOF,
            [0.0] * self.N_DOF,
            tau_total.tolist(),
            [0.0] * self.N_DOF,
            [0.0] * self.N_DOF,
        )
    
    # ── Target Tracking Mode ──────────────────────────────────────────────
    
    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        """Execute target tracking command on AIRBOT arm.
        
        MIT control law: tau = kp * (q_target - q) + kd * (qd_target - qd) + ff
        
        Note: gravity compensation is NOT added here.  The base class
        convenience method ``step_target_tracking()`` already computes
        gravity via ``self.kinematics.gravity_compensation()`` — which,
        for AIRBOT, is the ``AIRBOTKinematicsModel`` wrapper that
        automatically applies the per-joint EEF coefficients.
        This method is a thin passthrough to the hardware MIT command.
        """
        if self._control_mode != ControlMode.TARGET_TRACKING or self._arm is None:
            return
        
        tau_ff = np.zeros(self.N_DOF)
        if cmd.feedforward is not None:
            tau_ff = np.asarray(cmd.feedforward, dtype=np.float64)
        
        self._arm.mit(
            cmd.position_target.tolist(),
            cmd.velocity_target.tolist(),
            tau_ff.tolist(),
            cmd.kp.tolist(),
            cmd.kd.tolist(),
        )
    
    # ── PVT Mode (Position-Velocity-Torque) ───────────────────────────────
    
    def command_pvt(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        torque: np.ndarray,
    ) -> None:
        """Send a PVT (Position-Velocity-Torque) command.
        
        This is a lower-level interface for trajectory tracking.
        The arm must be in PVT mode (not FREE_DRIVE or TARGET_TRACKING).
        
        Args:
            position: Target joint positions (rad)
            velocity: Target joint velocities (rad/s)
            torque: Feedforward torques (Nm)
        """
        if self._arm is None:
            return
        
        self._arm.pvt(
            np.asarray(position).tolist(),
            np.asarray(velocity).tolist(),
            np.asarray(torque).tolist(),
        )
    
    def move_to_home(self, timeout: float = 10.0) -> bool:
        """Move the robot to home position (all joints at zero).
        
    This uses PVT mode for safe trajectory execution.
        
        Args:
            timeout: Maximum time to wait for completion (seconds)
            
        Returns:
            True if home position reached
        """
        import time
        
        if self._arm is None or not self._is_enabled:
            return False
        
        # Switch to PVT mode
        self._arm.set_param("arm.control_mode", self._ah.MotorControlMode.PVT)
        
        home_pos = [0.0] * self.N_DOF
        move_vel = [0.5] * self.N_DOF  # rad/s
        move_tau = [10.0] * self.N_DOF  # Nm (feedforward)
        
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < timeout:
            state = self._arm.state()
            if state.is_valid:
                self._arm.pvt(home_pos, move_vel, move_tau)
                
                # Check if at home
                if all(abs(p) < 0.01 for p in state.pos):
                    return True
            
            time.sleep(self._dt)
        
        return False
    
    # ── Convenience Methods ───────────────────────────────────────────────
    
    def step_free_drive_with_admittance(
        self,
        external_wrench: Wrench | None = None,
        mass: float = 5.0,
        damping: float = 50.0,
        stiffness: float = 0.0,
    ) -> None:
        """Free drive step with admittance control.
        
        Implements: M * xdd + D * xd + K * (x - x_eq) = F_ext
        
        This provides compliant behavior where external forces cause motion.
        
        Args:
            external_wrench: External wrench at end-effector
            mass: Virtual mass (kg)
            damping: Virtual damping (Ns/m)
            stiffness: Virtual stiffness (N/m), 0 for pure admittance
        """
        if external_wrench is None:
            external_wrench = Wrench.zero()
        
        cmd = FreeDriveCommand(
            external_wrench=external_wrench,
            gravity_compensation_scale=1.0
        )
        self.command_free_drive(cmd)
    
    # ── Identification ────────────────────────────────────────────────────
    
    def identify_start(self, with_gravity_comp: bool = True) -> bool:
        """Start visual identification by blinking LED orange.
        
        Optionally enables gravity compensation mode so the user can
        feel and move the arm during identification.
        
        Sends CAN command 0x080#150122 to make the LED blink orange,
        allowing the user to identify which physical arm this instance
        corresponds to.
        
        Args:
            with_gravity_comp: If True, also enable gravity compensation mode
        
        Returns:
            True if started successfully
        """
        led_ok = set_airbot_led(self._can_interface, blink_orange=True)
        
        if with_gravity_comp:
            try:
                if not self._is_open:
                    self.open()
                if not self._is_enabled:
                    self.enable()
                self.enter_free_drive()
            except Exception:
                pass  # LED identification still works even if gravity comp fails
        
        return led_ok
    
    def identify_stop(
        self, 
        disable_robot: bool = True,
        return_to_zero: bool = True,
        zero_timeout: float = 10.0,
    ) -> bool:
        """Stop visual identification and return LED to normal.
        
        Sends CAN command 0x080#15011F to return the LED to its normal state.
        Optionally returns the arm to zero position, then disables the robot.
        
        Args:
            disable_robot: If True, disable and close the robot connection
            return_to_zero: If True, move arm to zero position before disabling
            zero_timeout: Timeout for returning to zero (seconds)
        
        Returns:
            True if stopped successfully
        """
        led_ok = set_airbot_led(self._can_interface, blink_orange=False)
        
        if self._is_open:
            try:
                # Return to zero position before disabling
                if return_to_zero and self._is_enabled:
                    self.move_to_home(timeout=zero_timeout)
                
                if disable_robot:
                    if self._is_enabled:
                        self.disable()
                    self.close()
            except Exception:
                pass
        
        return led_ok
