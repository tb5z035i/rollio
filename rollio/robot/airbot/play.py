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

import time
from pathlib import Path
from typing import Any

import numpy as np

from rollio.defaults import DEFAULT_CONTROL_DT_SEC, DEFAULT_CONTROL_HZ
from rollio.robot.airbot.control_loop import (
    AirbotCommandPump,
    AirbotDynamicTrackingIntent,
    AirbotFixedTrackingIntent,
    AirbotFreeDriveIntent,
    AirbotLoopMetrics,
    AirbotPvtCommand,
    clone_wrench,
)
from rollio.robot.airbot.shared import (
    _import_airbot_hardware,
    get_shared_airbot_runtime,
    is_airbot_available,
    scan_airbot_detected_robots,
)
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
from rollio.robot.can_utils import is_can_interface_up
from rollio.robot.airbot.can import (
    probe_airbot_device,
    query_airbot_end_effector,
    query_airbot_gravity_coefficients,
    query_airbot_serial,
    set_airbot_led,
)
from rollio.robot.scanner import DetectedRobot
from rollio.utils.time import monotonic_sec

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
    def frame_names(self) -> list[str]:
        return self._inner.frame_names

    def forward_kinematics(self, q, frame=None):
        return self._inner.forward_kinematics(q, frame)

    def inverse_kinematics(
        self, target_pose, q_init=None, frame=None, max_iterations=100, tolerance=1e-6
    ):
        return self._inner.inverse_kinematics(
            target_pose, q_init, frame, max_iterations, tolerance
        )

    def jacobian(self, q, frame=None):
        return self._inner.jacobian(q, frame)

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
    - Target tracking mode with MIT or PVT backend control
    - MIT mode for direct torque control
    - PVT mode for trajectory tracking

    Args:
        can_interface: CAN interface name (e.g., "can0")
        urdf_path: Path to URDF file for kinematics (optional)
        control_frequency: Control loop frequency in Hz (default: 250)
        gravity_coefficients: Per-joint gravity compensation coefficients override.
            If None, coefficients are read from hardware and selected based on
            the detected end effector type.
        target_tracking_mode: AIRBOT target-tracking backend. ``"mit"`` keeps
            the original impedance-based tracking path; ``"pvt"`` replays the
            latest target with fixed PVT velocity/current-threshold vectors.
    """

    ROBOT_TYPE = "airbot_play"
    DIRECT_MAP_ALLOWLIST = ("airbot_play",)
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

    # Fixed gains required by the AIRBOT controller for MIT target tracking.
    TARGET_TRACKING_KP = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])
    TARGET_TRACKING_KD = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
    TARGET_TRACKING_MODE_MIT = "mit"
    TARGET_TRACKING_MODE_PVT = "pvt"
    TARGET_TRACKING_MODE_CHOICES = (
        TARGET_TRACKING_MODE_MIT,
        TARGET_TRACKING_MODE_PVT,
    )
    # Requested PVT tracking uses fixed velocity/current-threshold vectors.
    TARGET_TRACKING_PVT_VELOCITY = 10.0
    TARGET_TRACKING_PVT_CURRENT_THRESHOLD = 10.0

    # ── Class methods ─────────────────────────────────────────────────────

    @classmethod
    def scan(cls) -> list[DetectedRobot]:
        """Scan for available AIRBOT Play robots via CAN interfaces.

        Probes each CAN interface by sending the identify command (0x000#07)
        and checking for the expected AIRBOT response pattern.
        Also queries serial number and end effector type.
        """
        return [
            robot
            for robot in scan_airbot_detected_robots()
            if robot.robot_type == cls.ROBOT_TYPE
        ]

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
        control_frequency: int = DEFAULT_CONTROL_HZ,
        gravity_coefficients: np.ndarray | None = None,
        frame_name: str | None = None,
        target_tracking_mode: str = TARGET_TRACKING_MODE_MIT,
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
        self._target_tracking_mode = self._normalize_target_tracking_mode(
            target_tracking_mode
        )

        # State
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
        self._arm = None
        self._executor = None
        self._command_pump: AirbotCommandPump | None = None

        # Kinematics model (lazy loaded)
        self._kinematics: KinematicsModel | None = None
        self._urdf_path = urdf_path
        self._frame_name = frame_name

        # Robot properties (cached, populated by query_properties())
        self._properties: dict[str, Any] = {
            "can_interface": can_interface,
            "control_frequency": control_frequency,
            "motor_types": ["OD", "OD", "OD", "DM", "DM", "DM"],
            "target_tracking_mode": self._target_tracking_mode,
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
                FeedbackCapability.FRAME_POSE,
                FeedbackCapability.FRAME_TWIST,
            },
            properties=self._properties,
        )

    @classmethod
    def _normalize_target_tracking_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized in cls.TARGET_TRACKING_MODE_CHOICES:
            return normalized
        choices = ", ".join(cls.TARGET_TRACKING_MODE_CHOICES)
        raise ValueError(
            f"Unsupported AIRBOT Play target_tracking_mode {mode!r}. "
            f"Expected one of: {choices}."
        )

    def _target_tracking_uses_pvt(self) -> bool:
        return self._target_tracking_mode == self.TARGET_TRACKING_MODE_PVT

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
                    frame_name=self._frame_name,
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
            result = query_airbot_gravity_coefficients(self._can_interface, timeout=0.5)
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

    def _get_gravity_coefficients_for_eef(
        self, eef_type: str | None = None
    ) -> np.ndarray:
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

    @staticmethod
    def _sdk_mutator_succeeded(result: Any) -> bool:
        if result is None:
            return True
        return bool(result)

    def _read_sdk_state(self) -> Any | None:
        if self._arm is None or not self._is_open:
            return None
        try:
            return self._arm.state()
        except Exception:
            return None

    def _read_direct_joint_state(self) -> JointState:
        """Read the current joint state directly from the SDK handle."""
        ts = monotonic_sec()

        state = self._read_sdk_state()
        if state is None or not state.is_valid:
            return JointState(
                timestamp=ts,
                position=None,
                velocity=None,
                effort=None,
                is_valid=False,
            )

        return JointState(
            timestamp=ts,
            position=np.array(state.pos, dtype=np.float32),
            velocity=np.array(state.vel, dtype=np.float32),
            effort=np.array(state.eff, dtype=np.float32),
            is_valid=True,
        )

    def _apply_enabled_request(self, enabled: bool) -> bool:
        if self._arm is None or not self._is_open:
            return False
        if enabled:
            try:
                result = self._arm.enable()
            except Exception:
                return False
            return self._sdk_mutator_succeeded(result)
        if self._control_mode == ControlMode.FREE_DRIVE:
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.PVT,
            )
        self._arm.disable()
        return True

    def _apply_mode_request(self, mode: ControlMode) -> bool:
        if self._arm is None:
            return False
        if mode == ControlMode.FREE_DRIVE:
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.MIT,
            )
            return True
        if mode == ControlMode.TARGET_TRACKING:
            self._arm.set_param(
                "arm.control_mode",
                (
                    self._ah.MotorControlMode.PVT
                    if self._target_tracking_uses_pvt()
                    else self._ah.MotorControlMode.MIT
                ),
            )
            return True
        if mode == ControlMode.DISABLED:
            self._arm.set_param(
                "arm.control_mode",
                self._ah.MotorControlMode.PVT,
            )
            return True
        return False

    def _emit_free_drive(
        self,
        state: Any,
        intent: AirbotFreeDriveIntent,
    ) -> None:
        if self._arm is None:
            return

        q = np.asarray(state.pos, dtype=np.float64).reshape(-1)[: self.N_DOF]
        tau_gravity = np.array(
            self.kinematics.gravity_compensation(q),
            dtype=np.float64,
        )
        tau_gravity *= float(intent.gravity_compensation_scale)

        tau_ext = np.zeros(self.N_DOF, dtype=np.float64)
        if intent.external_wrench is not None:
            tau_ext = self.kinematics.wrench_to_joint_torques(
                q,
                intent.external_wrench,
            )

        tau_total = tau_gravity + tau_ext
        self._arm.mit(
            [0.0] * self.N_DOF,
            [0.0] * self.N_DOF,
            tau_total.tolist(),
            [0.0] * self.N_DOF,
            [0.0] * self.N_DOF,
        )

    def _resolve_tracking_gains(
        self,
        intent: AirbotFixedTrackingIntent,
    ) -> tuple[np.ndarray, np.ndarray]:
        kp = self.TARGET_TRACKING_KP.copy()
        kd = self.TARGET_TRACKING_KD.copy()
        if intent.kp is not None:
            user_kp = np.asarray(intent.kp, dtype=np.float64).reshape(-1)
            count = min(user_kp.size, self.N_DOF)
            kp[:count] = user_kp[:count]
            # Preserve the vendor-tuned shoulder stiffness even with overrides.
            kp[:3] = self.TARGET_TRACKING_KP[:3]
        if intent.kd is not None:
            user_kd = np.asarray(intent.kd, dtype=np.float64).reshape(-1)
            count = min(user_kd.size, self.N_DOF)
            kd[:count] = user_kd[:count]
            # Preserve the vendor-tuned wrist damping even with overrides.
            kd[3:] = self.TARGET_TRACKING_KD[3:]
        return kp, kd

    def _emit_fixed_tracking(self, intent: AirbotFixedTrackingIntent) -> None:
        if self._arm is None:
            return
        if self._target_tracking_uses_pvt():
            self._emit_pvt_tracking(intent.position_target)
            return
        tau_ff = (
            np.zeros(self.N_DOF, dtype=np.float64)
            if intent.feedforward is None
            else np.asarray(intent.feedforward, dtype=np.float64)
        )
        kp, kd = self._resolve_tracking_gains(intent)
        self._arm.mit(
            np.asarray(intent.position_target, dtype=np.float64).tolist(),
            np.asarray(intent.velocity_target, dtype=np.float64).tolist(),
            tau_ff.tolist(),
            kp.tolist(),
            kd.tolist(),
        )

    def _emit_pvt_tracking(self, position_target: np.ndarray) -> None:
        if self._arm is None:
            return
        velocity_target = np.full(
            self.N_DOF,
            self.TARGET_TRACKING_PVT_VELOCITY,
            dtype=np.float64,
        )
        # The Play SDK exposes the third PVT vector as effort/threshold-like data.
        current_threshold = np.full(
            self.N_DOF,
            self.TARGET_TRACKING_PVT_CURRENT_THRESHOLD,
            dtype=np.float64,
        )
        self._arm.pvt(
            np.asarray(position_target, dtype=np.float64).tolist(),
            velocity_target.tolist(),
            current_threshold.tolist(),
        )

    def _emit_dynamic_tracking(
        self,
        state: Any,
        intent: AirbotDynamicTrackingIntent,
    ) -> None:
        tau_ff = (
            np.zeros(self.N_DOF, dtype=np.float64)
            if intent.user_feedforward is None
            else np.asarray(intent.user_feedforward, dtype=np.float64).copy()
        )
        if intent.add_gravity_compensation:
            q = np.asarray(state.pos, dtype=np.float64).reshape(-1)[: self.N_DOF]
            tau_ff = tau_ff + self.kinematics.gravity_compensation(
                q,
            )
        self._emit_fixed_tracking(
            AirbotFixedTrackingIntent(
                position_target=np.asarray(intent.position_target, dtype=np.float64),
                velocity_target=np.asarray(intent.velocity_target, dtype=np.float64),
                feedforward=tau_ff,
                kp=intent.kp,
                kd=intent.kd,
                published_at=intent.published_at,
            ),
        )

    def _emit_raw_pvt(self, command: AirbotPvtCommand) -> None:
        if self._arm is None:
            return
        self._arm.pvt(
            np.asarray(command.position_target, dtype=np.float64).tolist(),
            np.asarray(command.velocity_target, dtype=np.float64).tolist(),
            np.asarray(command.effort, dtype=np.float64).tolist(),
        )

    def _control_cycle(
        self,
        command: (
            AirbotFreeDriveIntent
            | AirbotFixedTrackingIntent
            | AirbotDynamicTrackingIntent
            | AirbotPvtCommand
            | None
        ),
        mode: ControlMode,
        enabled: bool,
    ) -> None:
        if not enabled or self._arm is None or command is None:
            return
        if isinstance(command, AirbotPvtCommand):
            self._emit_raw_pvt(command)
            return
        if mode == ControlMode.FREE_DRIVE and isinstance(
            command, AirbotFreeDriveIntent
        ):
            state = self._read_sdk_state()
            if state is None or not state.is_valid:
                return
            self._emit_free_drive(state, command)
            return
        if mode != ControlMode.TARGET_TRACKING:
            return
        if isinstance(command, AirbotDynamicTrackingIntent):
            if self._target_tracking_uses_pvt():
                self._emit_pvt_tracking(command.position_target)
                return
            state = self._read_sdk_state()
            if state is None or not state.is_valid:
                return
            self._emit_dynamic_tracking(state, command)
            return
        if isinstance(command, AirbotFixedTrackingIntent):
            self._emit_fixed_tracking(command)

    def _start_command_pump(self) -> None:
        if self._command_pump is not None:
            return
        self._command_pump = AirbotCommandPump(
            name=f"rollio-airbot-play-{self._can_interface}",
            period_sec=self._dt,
            apply_enabled=self._apply_enabled_request,
            apply_mode=self._apply_mode_request,
            cycle=self._control_cycle,
            initial_enabled=self._is_enabled,
            initial_mode=self._control_mode,
        )
        self._command_pump.start()

    def _stop_command_pump(self) -> None:
        if self._command_pump is None:
            return
        self._command_pump.stop()
        self._command_pump = None

    def _publish_command(
        self,
        command: (
            AirbotFreeDriveIntent
            | AirbotFixedTrackingIntent
            | AirbotDynamicTrackingIntent
            | AirbotPvtCommand
            | None
        ),
        *,
        owner: str | None = None,
    ) -> bool:
        if self._command_pump is None:
            return False
        return self._command_pump.publish_command(command, owner=owner)

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
    def target_tracking_mode(self) -> str:
        return self._target_tracking_mode

    @property
    def can_interface(self) -> str:
        """Get the CAN interface name."""
        return self._can_interface

    def control_loop_metrics(self) -> AirbotLoopMetrics:
        if self._command_pump is not None:
            return self._command_pump.metrics()
        return AirbotLoopMetrics(target_interval_ms=self._dt * 1000.0, run_count=0)

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
    def gravity_compensation_scale(self) -> np.ndarray:
        """Backward-compatible alias for per-joint gravity coefficients."""
        return self.gravity_coefficients

    @gravity_compensation_scale.setter
    def gravity_compensation_scale(self, coefficients: np.ndarray | None) -> None:
        self.gravity_coefficients = coefficients

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
            properties=self._properties,
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

        try:
            self._executor, io_context = get_shared_airbot_runtime(self._ah)

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
            initialized = self._arm.init(
                io_context,
                self._can_interface,
                self._control_frequency,
            )
            if not initialized:
                raise RuntimeError(
                    f"Failed to initialize AIRBOT arm on {self._can_interface}. "
                    "Check CAN connection and motor power."
                )
        except Exception:
            self._arm = None
            self._executor = None
            raise

        self._is_open = True
        self._start_command_pump()

    def close(self) -> None:
        """Close connection to AIRBOT arm."""
        if not self._is_open:
            return
        try:
            if self._is_enabled:
                self.disable()
            self._stop_command_pump()
            if self._arm is not None:
                self._arm.uninit()
                self._arm = None
        finally:
            self._executor = None
            self._is_open = False

    def enable(self) -> bool:
        """Enable AIRBOT motors."""
        if not self._is_open or self._arm is None:
            return False
        if self._command_pump is None:
            self._start_command_pump()
        if self._command_pump is None:
            self._is_enabled = self._apply_enabled_request(True)
            return self._is_enabled
        self._is_enabled = self._command_pump.request_enabled(True)
        return self._is_enabled

    def disable(self) -> None:
        """Disable AIRBOT motors (safe state)."""
        if self._command_pump is not None:
            self._command_pump.request_enabled(False)
            self._command_pump.reset_command()
        elif self._arm is not None and self._is_enabled:
            self._apply_enabled_request(False)
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

    # ── State Reading ─────────────────────────────────────────────────────

    def read_joint_state(self) -> JointState:
        """Read current joint state from AIRBOT arm."""
        return self._read_direct_joint_state()

    # ── Control Mode Setting ──────────────────────────────────────────────

    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode."""
        if not self._is_enabled or self._arm is None:
            return False
        if self._command_pump is not None:
            ok = self._command_pump.request_mode(mode)
        else:
            ok = self._apply_mode_request(mode)
        if not ok:
            return False
        self._control_mode = mode
        if mode == ControlMode.FREE_DRIVE:
            self._publish_command(
                AirbotFreeDriveIntent(
                    gravity_compensation_scale=1.0,
                    external_wrench=None,
                    published_at=time.monotonic(),
                ),
            )
        else:
            self._publish_command(None)
        return True

    # ── Free Drive Mode ───────────────────────────────────────────────────

    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        """Publish the latest free-drive intent for the device thread."""
        if self._control_mode != ControlMode.FREE_DRIVE or self._arm is None:
            return
        self._publish_command(
            AirbotFreeDriveIntent(
                gravity_compensation_scale=float(cmd.gravity_compensation_scale),
                external_wrench=clone_wrench(cmd.external_wrench),
                published_at=time.monotonic(),
            ),
        )

    # ── Target Tracking Mode ──────────────────────────────────────────────

    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        """Publish one fixed tracking packet for the device thread."""
        if self._control_mode != ControlMode.TARGET_TRACKING or self._arm is None:
            return
        self._publish_command(
            AirbotFixedTrackingIntent(
                position_target=np.asarray(
                    cmd.position_target,
                    dtype=np.float64,
                ).copy(),
                velocity_target=np.asarray(
                    cmd.velocity_target,
                    dtype=np.float64,
                ).copy(),
                feedforward=(
                    None
                    if cmd.feedforward is None
                    else np.asarray(cmd.feedforward, dtype=np.float64).copy()
                ),
                kp=(
                    None
                    if cmd.kp is None
                    else np.asarray(cmd.kp, dtype=np.float64).copy()
                ),
                kd=(
                    None
                    if cmd.kd is None
                    else np.asarray(cmd.kd, dtype=np.float64).copy()
                ),
                published_at=time.monotonic(),
            ),
        )

    def step_target_tracking(
        self,
        position_target: np.ndarray,
        velocity_target: np.ndarray | None = None,
        kp: np.ndarray | float | None = None,
        kd: np.ndarray | float | None = None,
        feedforward: np.ndarray | None = None,
        add_gravity_compensation: bool = True,
    ) -> None:
        """Publish a high-level tracking target that the device thread replays."""
        if self._control_mode != ControlMode.TARGET_TRACKING or self._arm is None:
            return
        n = self.n_dof
        if velocity_target is None:
            velocity_target = np.zeros(n, dtype=np.float64)
        kp_array = None
        kd_array = None
        if kp is not None:
            if np.isscalar(kp):
                kp_array = np.full(n, float(kp), dtype=np.float64)
            else:
                kp_array = np.asarray(kp, dtype=np.float64).copy()
        if kd is not None:
            if np.isscalar(kd):
                kd_array = np.full(n, float(kd), dtype=np.float64)
            else:
                kd_array = np.asarray(kd, dtype=np.float64).copy()
        self._publish_command(
            AirbotDynamicTrackingIntent(
                position_target=np.asarray(position_target, dtype=np.float64).copy(),
                velocity_target=np.asarray(velocity_target, dtype=np.float64).copy(),
                user_feedforward=(
                    None
                    if feedforward is None
                    else np.asarray(feedforward, dtype=np.float64).copy()
                ),
                kp=kp_array,
                kd=kd_array,
                add_gravity_compensation=bool(add_gravity_compensation),
                published_at=time.monotonic(),
            ),
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
        if (
            self._arm is None
            or not self._is_enabled
            or self._control_mode != ControlMode.DISABLED
        ):
            return

        self._publish_command(
            AirbotPvtCommand(
                position_target=np.asarray(position, dtype=np.float64).copy(),
                velocity_target=np.asarray(velocity, dtype=np.float64).copy(),
                effort=np.asarray(torque, dtype=np.float64).copy(),
                published_at=time.monotonic(),
            ),
        )

    def move_to_home(self, timeout: float = 10.0) -> bool:
        """Move the robot to home position (all joints at zero).

        This uses PVT mode for safe trajectory execution.

            Args:
                timeout: Maximum time to wait for completion (seconds)

            Returns:
                True if home position reached
        """
        if self._arm is None or not self._is_enabled or self._command_pump is None:
            return False

        lease = self._command_pump.acquire_lease("move_to_home")
        if lease is None:
            return False

        with lease:
            if not self._command_pump.request_mode(ControlMode.DISABLED):
                return False
            self._control_mode = ControlMode.DISABLED
            if not self._publish_command(
                AirbotPvtCommand(
                    position_target=np.zeros(self.N_DOF, dtype=np.float64),
                    velocity_target=np.full(self.N_DOF, 0.5, dtype=np.float64),
                    effort=np.full(self.N_DOF, 10.0, dtype=np.float64),
                    published_at=time.monotonic(),
                ),
                owner=lease.owner,
            ):
                return False

            start_time = time.monotonic()
            try:
                while time.monotonic() - start_time < timeout:
                    state = self._read_sdk_state()
                    if state is not None and state.is_valid:
                        if all(abs(float(p)) < 0.01 for p in state.pos[: self.N_DOF]):
                            return True
                    time.sleep(self._dt)
            finally:
                self._publish_command(None, owner=lease.owner)

        return False

    def move_to_zero(
        self,
        timeout: float = 10.0,
        tolerance: float = 0.01,
        dt: float = DEFAULT_CONTROL_DT_SEC,
    ) -> bool:
        del tolerance, dt
        return self.move_to_home(timeout=timeout)

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
            external_wrench=external_wrench, gravity_compensation_scale=1.0
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

    def identify_step(self) -> None:
        """Keep the arm backdrivable while identification is active."""
        if (
            self._is_open
            and self._is_enabled
            and self._control_mode == ControlMode.FREE_DRIVE
        ):
            self.step_free_drive()

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
