"""Standalone AIRBOT EEF drivers exposed as 1-DOF robots."""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from rollio.defaults import DEFAULT_CONTROL_DT_SEC, DEFAULT_CONTROL_HZ
from rollio.robot.airbot.control_loop import (
    AirbotCommandPump,
    AirbotFixedTrackingIntent,
    AirbotFreeDriveIntent,
    AirbotLoopMetrics,
)
from rollio.robot.airbot.shared import (
    AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR,
    AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF,
    AIRBOT_ROBOT_TYPE_TO_SDK_EEF,
    get_shared_airbot_runtime,
    is_airbot_available,
    normalize_airbot_eef_type,
    scan_airbot_detected_robots,
)
from rollio.robot.base import (
    ControlMode,
    FeedbackCapability,
    FreeDriveCommand,
    JointState,
    KinematicsModel,
    Pose,
    RobotArm,
    RobotInfo,
    TargetTrackingCommand,
)
from rollio.robot.airbot.can import query_airbot_end_effector, set_airbot_led
from rollio.robot.scanner import DetectedRobot
from rollio.utils.time import monotonic_sec


class AIRBOTEEFLinearKinematics(KinematicsModel):
    """Minimal 1-DOF task-space model for standalone AIRBOT EEFs."""

    def __init__(self, frame_name: str = "frame") -> None:
        self._frame_names = [frame_name]

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def frame_names(self) -> list[str]:
        return self._frame_names

    def forward_kinematics(
        self,
        q: np.ndarray,
        frame: str | None = None,
    ) -> Pose:
        del frame
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        x = float(q[0]) if q.size else 0.0
        return Pose(
            position=np.array([x, 0.0, 0.0], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: np.ndarray | None = None,
        frame: str | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        del q_init, frame, max_iterations, tolerance
        return np.array([float(target_pose.position[0])], dtype=np.float64), True

    def jacobian(
        self,
        q: np.ndarray,
        frame: str | None = None,
    ) -> np.ndarray:
        del q, frame
        jacobian = np.zeros((6, 1), dtype=np.float64)
        jacobian[0, 0] = 1.0
        return jacobian

    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        del q, qd, qdd
        return np.zeros(1, dtype=np.float64)


class _AIRBOTStandaloneEEFCommon:
    """Thin shared helper for standalone AIRBOT EEF SDK plumbing only."""

    ROBOT_TYPE: ClassVar[str] = "airbot_eef"

    def __init__(
        self,
        can_interface: str = "can0",
        control_frequency: int = DEFAULT_CONTROL_HZ,
        motor_type: str | None = None,
        *,
        frame_name: str = "frame",
    ) -> None:
        try:
            import airbot_hardware_py as ah
        except ImportError as exc:
            raise ImportError(
                "airbot_hardware_py is required for AIRBOT end-effector support. "
                "Install the AIRBOT SDK."
            ) from exc

        self._ah = ah
        self._can_interface = can_interface
        self._control_frequency = control_frequency
        allowed_motor_type = AIRBOT_ROBOT_TYPE_TO_DEFAULT_MOTOR[self.ROBOT_TYPE]
        requested_motor_type = (
            str(motor_type).strip().upper()
            if motor_type is not None
            else allowed_motor_type
        )
        if requested_motor_type != allowed_motor_type:
            raise ValueError(
                f"{self.ROBOT_TYPE} only supports the standalone AIRBOT EEF "
                f"combination ({AIRBOT_ROBOT_TYPE_TO_SDK_EEF[self.ROBOT_TYPE]}, "
                f"{allowed_motor_type})."
            )
        self._motor_type_name = requested_motor_type
        self._dt = 1.0 / max(control_frequency, 1)
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
        self._eef = None
        self._executor = None
        self._command_pump: AirbotCommandPump | None = None
        self._identify_started_at: float | None = None
        self._last_command_type: str | None = None
        self._last_command_args: dict[str, tuple[float, ...]] = {}
        self._kinematics = AIRBOTEEFLinearKinematics(frame_name=frame_name)
        self._properties: dict[str, Any] = {
            "can_interface": can_interface,
            "control_frequency": control_frequency,
            "end_effector_type": AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF[self.ROBOT_TYPE],
            "eef_motor_type": self._motor_type_name,
        }
        self._info = RobotInfo(
            name=f"{self.ROBOT_TYPE}_{can_interface}",
            robot_type=self.ROBOT_TYPE,
            n_dof=1,
            feedback_capabilities={
                FeedbackCapability.POSITION,
                FeedbackCapability.VELOCITY,
                FeedbackCapability.EFFORT,
            },
            properties=self._properties,
        )

    @classmethod
    def scan(cls) -> list[DetectedRobot]:
        return [
            robot
            for robot in scan_airbot_detected_robots()
            if robot.robot_type == cls.ROBOT_TYPE
        ]

    @classmethod
    def probe(cls, device_id: int | str) -> bool:
        if not is_airbot_available():
            return False
        eef = query_airbot_end_effector(str(device_id), timeout=0.5)
        if not eef:
            return False
        return (
            normalize_airbot_eef_type(eef.get("type_name"))
            == AIRBOT_ROBOT_TYPE_TO_DETECTED_EEF[cls.ROBOT_TYPE]
        )

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties.copy()

    @property
    def kinematics(self) -> KinematicsModel:
        return self._kinematics

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    def control_loop_metrics(self) -> AirbotLoopMetrics:
        if self._command_pump is not None:
            return self._command_pump.metrics()
        return AirbotLoopMetrics(target_interval_ms=self._dt * 1000.0, run_count=0)

    @staticmethod
    def _sdk_mutator_succeeded(result: Any) -> bool:
        if result is None:
            return True
        return bool(result)

    def _resolve_eef_type_enum(self) -> Any:
        sdk_eef_type = AIRBOT_ROBOT_TYPE_TO_SDK_EEF[self.ROBOT_TYPE]
        if sdk_eef_type == "E2":
            return self._ah.EEFType.E2
        if sdk_eef_type == "G2":
            return self._ah.EEFType.G2
        raise ValueError(f"Unsupported AIRBOT EEF type {sdk_eef_type!r}")

    def _resolve_motor_type_enum(self) -> Any:
        if self._motor_type_name == "OD":
            return self._ah.MotorType.OD
        if self._motor_type_name == "DM":
            return self._ah.MotorType.DM
        if self._motor_type_name == "NA":
            return self._ah.MotorType.NA
        raise ValueError(f"Unsupported AIRBOT motor type {self._motor_type_name!r}")

    def _create_eef_handle(self):
        eef_type = self._resolve_eef_type_enum()
        motor_type = self._resolve_motor_type_enum()

        for factory_name in ("EEF1", "EEF"):
            factory = getattr(self._ah, factory_name, None)
            if factory is None:
                continue
            handle = factory.create(eef_type, motor_type)
            if handle is not None:
                return handle

        raise RuntimeError(
            f"AIRBOT SDK does not expose a standalone EEF handle for "
            f"({AIRBOT_ROBOT_TYPE_TO_SDK_EEF[self.ROBOT_TYPE]}, "
            f"{self._motor_type_name})."
        )

    def _set_param(self, param_name: str, value: Any) -> bool:
        if self._eef is None:
            return False
        try:
            result = self._eef.set_param(param_name, value)
        except (OSError, RuntimeError, ValueError, TypeError):
            return False
        return self._sdk_mutator_succeeded(result)

    def _assign_vector_field(
        self, target: Any, field_name: str, values: np.ndarray
    ) -> None:
        data = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]
        if field_name not in {
            "pos",
            "vel",
            "eff",
            "mit_kp",
            "mit_kd",
            "current_threshold",
        }:
            raise ValueError(f"Unsupported command field {field_name!r}")
        setattr(target, field_name, data)

    def _create_command_instance(self) -> Any | None:
        for class_name in ("EEFCommand1", "EEFCommand", "EEFState1", "EEFState"):
            command_cls = getattr(self._ah, class_name, None)
            if command_cls is not None:
                return command_cls()
        return None

    @staticmethod
    def _state_is_valid(state: Any | None) -> bool:
        return bool(state is not None and state.is_valid)

    @staticmethod
    def _extract_joint_measurements(
        state: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = np.array(state.pos, dtype=np.float32).reshape(-1)[:1]
        velocity = np.array(state.vel, dtype=np.float32).reshape(-1)[:1]
        effort = np.array(state.eff, dtype=np.float32).reshape(-1)[:1]
        return position, velocity, effort

    def _build_command_payload(
        self,
        position_target: np.ndarray,
        velocity_target: np.ndarray,
        *,
        effort: np.ndarray | None = None,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        current_threshold: np.ndarray | None = None,
    ) -> Any:
        command = self._create_command_instance()
        if command is None:
            return None
        self._assign_vector_field(command, "pos", position_target)
        self._assign_vector_field(command, "vel", velocity_target)
        self._assign_vector_field(
            command,
            "eff",
            np.zeros(1, dtype=np.float64) if effort is None else effort,
        )
        self._assign_vector_field(
            command,
            "mit_kp",
            np.zeros(1, dtype=np.float64) if kp is None else kp,
        )
        self._assign_vector_field(
            command,
            "mit_kd",
            np.zeros(1, dtype=np.float64) if kd is None else kd,
        )
        self._assign_vector_field(
            command,
            "current_threshold",
            (
                np.zeros(1, dtype=np.float64)
                if current_threshold is None
                else current_threshold
            ),
        )
        return command

    def _record_command_debug(
        self,
        command_type: str,
        *,
        position_target: np.ndarray,
        velocity_target: np.ndarray,
        effort: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        current_threshold: np.ndarray,
    ) -> None:
        self._last_command_type = command_type
        self._last_command_args = {
            "pos": tuple(
                float(v)
                for v in np.asarray(position_target, dtype=np.float64).reshape(-1)
            ),
            "vel": tuple(
                float(v)
                for v in np.asarray(velocity_target, dtype=np.float64).reshape(-1)
            ),
            "eff": tuple(
                float(v) for v in np.asarray(effort, dtype=np.float64).reshape(-1)
            ),
            "mit_kp": tuple(
                float(v) for v in np.asarray(kp, dtype=np.float64).reshape(-1)
            ),
            "mit_kd": tuple(
                float(v) for v in np.asarray(kd, dtype=np.float64).reshape(-1)
            ),
            "current_threshold": tuple(
                float(v)
                for v in np.asarray(current_threshold, dtype=np.float64).reshape(-1)
            ),
        }

    def latest_command_debug(self) -> tuple[str, str] | None:
        if self._last_command_type is None:
            return None
        parts: list[str] = []
        for key in ("pos", "vel", "eff", "mit_kp", "mit_kd", "current_threshold"):
            values = self._last_command_args.get(key)
            if values is None:
                continue
            formatted = "[" + ", ".join(f"{v:7.4f}" for v in values) + "]"
            parts.append(f"{key}={formatted}")
        return self._last_command_type, " ".join(parts)

    def open(self) -> None:
        if self._is_open:
            return
        try:
            self._executor, io_context = get_shared_airbot_runtime(self._ah)
            handle = self._create_eef_handle()
            initialized = handle.init(
                io_context,
                self._can_interface,
                250,
            )
            if not initialized:
                raise RuntimeError(
                    f"Failed to initialize AIRBOT EEF '{self.ROBOT_TYPE}' on "
                    f"{self._can_interface} with "
                    f"({AIRBOT_ROBOT_TYPE_TO_SDK_EEF[self.ROBOT_TYPE]}, "
                    f"{self._motor_type_name})."
                )
            self._eef = handle
            self._is_open = True
            self._start_command_pump()
        except (OSError, RuntimeError, ValueError, TypeError):
            self._executor = None
            raise

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            if self._is_enabled:
                self.disable()
            self._stop_command_pump()
            if self._eef is not None:
                self._eef.uninit()
        finally:
            self._eef = None
            self._executor = None
            self._is_open = False

    def enable(self) -> bool:
        if not self._is_open or self._eef is None:
            return False
        if self._command_pump is None:
            self._start_command_pump()
        if self._command_pump is None:
            self._is_enabled = self._apply_enabled_request(True)
            return self._is_enabled
        self._is_enabled = self._command_pump.request_enabled(True)
        return self._is_enabled

    def disable(self) -> None:
        if self._command_pump is not None:
            self._command_pump.request_enabled(False)
            self._command_pump.reset_command()
        elif self._eef is not None and self._is_enabled:
            self._apply_enabled_request(False)
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

    def _read_sdk_state(self) -> Any | None:
        if self._eef is None or not self._is_open:
            return None
        try:
            return self._eef.state()
        except (OSError, RuntimeError, ValueError, TypeError):
            return None

    def _read_direct_joint_state(self) -> JointState:
        ts = monotonic_sec()
        state = self._read_sdk_state()
        if not self._state_is_valid(state):
            return JointState(
                timestamp=ts,
                position=None,
                velocity=None,
                effort=None,
                is_valid=False,
            )

        position, velocity, effort = self._extract_joint_measurements(state)
        return JointState(
            timestamp=ts,
            position=position,
            velocity=velocity,
            effort=effort,
            is_valid=True,
        )

    def read_joint_state(self) -> JointState:
        return self._read_direct_joint_state()

    def _apply_enabled_request(self, enabled: bool) -> bool:
        if self._eef is None or not self._is_open:
            return False
        if enabled:
            try:
                result = self._eef.enable()
            except (OSError, RuntimeError, ValueError, TypeError):
                return False
            return self._sdk_mutator_succeeded(result)
        self._eef.disable()
        return True

    def _start_command_pump(self) -> None:
        if self._command_pump is not None:
            return
        self._command_pump = AirbotCommandPump(
            name=f"rollio-{self.ROBOT_TYPE}-{self._can_interface}",
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
        command: AirbotFreeDriveIntent | AirbotFixedTrackingIntent | None,
    ) -> bool:
        if self._command_pump is None:
            return False
        return self._command_pump.publish_command(command)

    def _apply_mode_request(self, mode: ControlMode) -> bool:
        del mode
        return False

    def _control_cycle(
        self,
        command: AirbotFreeDriveIntent | AirbotFixedTrackingIntent | None,
        mode: ControlMode,
        enabled: bool,
    ) -> None:
        del command, mode, enabled


# Vendor-specific: E2B position readout scale factor (hardcoded per vendor specs)
_E2B_POSITION_READ_SCALE = 1.5


class AIRBOTE2B(_AIRBOTStandaloneEEFCommon, RobotArm):
    """AIRBOT E2B exposed as a 1-DOF feedback/keepalive robot."""

    ROBOT_TYPE = "airbot_e2b"

    def read_joint_state(self) -> JointState:
        state = super().read_joint_state()
        self._publish_plotjuggler_joint_state(state)
        return state

    def _read_direct_joint_state(self) -> JointState:
        state = super()._read_direct_joint_state()
        if state.is_valid and state.position is not None:
            state = JointState(
                timestamp=state.timestamp,
                position=np.asarray(state.position, dtype=np.float64)
                * _E2B_POSITION_READ_SCALE,
                velocity=state.velocity,
                effort=state.effort,
                is_valid=state.is_valid,
            )
        return state

    def __init__(
        self,
        can_interface: str = "can0",
        control_frequency: int = DEFAULT_CONTROL_HZ,
        motor_type: str | None = None,
    ) -> None:
        super().__init__(
            can_interface=can_interface,
            control_frequency=control_frequency,
            motor_type=motor_type,
        )

    @classmethod
    def default_direct_map_allowlist(
        cls,
        robot_type: str | None = None,
        role: str | None = None,
    ) -> tuple[str, ...]:
        del cls, robot_type
        normalized_role = str(role).strip().lower() if role is not None else ""
        if normalized_role == "leader":
            return ("airbot_g2",)
        if normalized_role == "follower":
            return ()
        return ("airbot_g2",)

    @classmethod
    def default_preview_control_mode(
        cls, role: str | None = None
    ) -> ControlMode | None:
        del cls, role
        return ControlMode.FREE_DRIVE

    def _configure_feedback_mode(self) -> bool:
        param_type = getattr(self._ah, "ParamType", None)
        param_value_cls = getattr(self._ah, "ParamValue", None)
        if param_type is None or param_value_cls is None or self._eef is None:
            return False
        value = param_value_cls(param_type.UINT32_LE, 4)  # pylint: disable=not-callable
        try:
            # Match the vendor E2 example: apply the mode and ignore the SDK
            # return value unless it raises.
            self._eef.set_param("eef.e2.mode", value)
        except (OSError, RuntimeError, ValueError, TypeError):
            return False
        return True

    def _apply_mode_request(self, mode: ControlMode) -> bool:
        if not self._is_enabled or self._eef is None:
            return False
        if mode == ControlMode.DISABLED:
            return True
        if mode != ControlMode.FREE_DRIVE:
            return False
        return self._configure_feedback_mode()

    def set_control_mode(self, mode: ControlMode) -> bool:
        if not self._is_enabled or self._eef is None:
            return False
        ok = (
            self._command_pump.request_mode(mode)
            if self._command_pump is not None
            else self._apply_mode_request(mode)
        )
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

    def _send_feedback_keepalive(self) -> None:
        if self._eef is None:
            return
        payload = self._create_command_instance()
        if payload is None:
            return
        zeros = np.zeros(1, dtype=np.float64)
        self._record_command_debug(
            "MIT",
            position_target=zeros,
            velocity_target=zeros,
            effort=zeros,
            kp=zeros,
            kd=zeros,
            current_threshold=zeros,
        )
        self._eef.mit(payload)

    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        del cmd
        if self._control_mode != ControlMode.FREE_DRIVE or self._eef is None:
            return
        self._publish_command(
            AirbotFreeDriveIntent(
                gravity_compensation_scale=1.0,
                external_wrench=None,
                published_at=time.monotonic(),
            ),
        )

    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        del cmd

    def _control_cycle(
        self,
        command: AirbotFreeDriveIntent | AirbotFixedTrackingIntent | None,
        mode: ControlMode,
        enabled: bool,
    ) -> None:
        if (
            not enabled
            or mode != ControlMode.FREE_DRIVE
            or not isinstance(command, AirbotFreeDriveIntent)
        ):
            return
        del command
        if self._eef is not None:
            self._send_feedback_keepalive()

    def move_to_zero(
        self,
        timeout: float = 10.0,
        tolerance: float = 0.01,
        dt: float = DEFAULT_CONTROL_DT_SEC,
    ) -> bool:
        del timeout, tolerance, dt
        return False

    def identify_start(self) -> bool:
        led_ok = set_airbot_led(self._can_interface, blink_orange=True)
        try:
            if not self._is_open:
                self.open()
            if not self._is_enabled:
                self.enable()
            self.enter_free_drive()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        return led_ok

    def identify_step(self) -> None:
        if not self._is_open or not self._is_enabled:
            return
        self.step_free_drive()

    def identify_stop(self, disable_robot: bool = True) -> bool:
        led_ok = set_airbot_led(self._can_interface, blink_orange=False)
        if self._is_open and disable_robot:
            try:
                if self._is_enabled:
                    self.disable()
                self.close()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
        return led_ok


class AIRBOTG2(_AIRBOTStandaloneEEFCommon, RobotArm):
    """AIRBOT G2 gripper exposed as a 1-DOF target-tracking robot."""

    ROBOT_TYPE = "airbot_g2"
    TARGET_TRACKING_MODE_MIT = "mit"
    TARGET_TRACKING_MODE_PVT = "pvt"
    TARGET_TRACKING_MODE_CHOICES = (
        TARGET_TRACKING_MODE_MIT,
        TARGET_TRACKING_MODE_PVT,
    )
    DEFAULT_TARGET_TRACKING_MODE = TARGET_TRACKING_MODE_MIT
    DEFAULT_PVT_VELOCITY = 200.0
    PVT_CURRENT_THRESHOLD = 200.0
    TARGET_TRACKING_MIT_KP = 20.0
    TARGET_TRACKING_MIT_KD = 1.0
    IDENTIFY_OSCILLATION_MIN = 0.0
    IDENTIFY_OSCILLATION_MAX = 0.07
    IDENTIFY_OSCILLATION_HZ = 0.25

    def __init__(
        self,
        can_interface: str = "can0",
        control_frequency: int = DEFAULT_CONTROL_HZ,
        motor_type: str | None = None,
        target_tracking_mode: str = DEFAULT_TARGET_TRACKING_MODE,
        pvt_velocity: float = DEFAULT_PVT_VELOCITY,
    ) -> None:
        super().__init__(
            can_interface=can_interface,
            control_frequency=control_frequency,
            motor_type=motor_type,
        )
        self._target_tracking_mode = self._normalize_target_tracking_mode(
            target_tracking_mode
        )
        self._pvt_velocity = max(0.1, float(pvt_velocity))
        self._properties["pvt_velocity"] = self._pvt_velocity
        self._properties["target_tracking_mode"] = self._target_tracking_mode

    def read_joint_state(self) -> JointState:
        state = super().read_joint_state()
        self._publish_plotjuggler_joint_state(state)
        return state

    @classmethod
    def default_direct_map_allowlist(
        cls,
        robot_type: str | None = None,
        role: str | None = None,
    ) -> tuple[str, ...]:
        del cls, robot_type
        normalized_role = str(role).strip().lower() if role is not None else ""
        if normalized_role == "leader":
            return ()
        return ("airbot_e2b",)

    @classmethod
    def default_preview_control_mode(
        cls, role: str | None = None
    ) -> ControlMode | None:
        del cls, role
        return ControlMode.TARGET_TRACKING

    @classmethod
    def default_preview_keepalive(cls, role: str | None = None) -> bool:
        del cls, role
        return True

    @classmethod
    def _normalize_target_tracking_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized in cls.TARGET_TRACKING_MODE_CHOICES:
            return normalized
        choices = ", ".join(cls.TARGET_TRACKING_MODE_CHOICES)
        raise ValueError(
            f"Unsupported AIRBOT G2 target_tracking_mode {mode!r}. "
            f"Expected one of: {choices}."
        )

    def _target_tracking_uses_pvt(self) -> bool:
        return self._target_tracking_mode == self.TARGET_TRACKING_MODE_PVT

    def _mit_tracking_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([self.TARGET_TRACKING_MIT_KP], dtype=np.float64),
            np.array([self.TARGET_TRACKING_MIT_KD], dtype=np.float64),
        )

    @property
    def target_tracking_mode(self) -> str:
        return self._target_tracking_mode

    def _set_mit_mode(self) -> bool:
        return self._set_param("control_mode", self._ah.MotorControlMode.MIT)

    def _set_pvt_mode(self) -> bool:
        return self._set_param("control_mode", self._ah.MotorControlMode.PVT)

    def _apply_mode_request(self, mode: ControlMode) -> bool:
        if not self._is_enabled or self._eef is None:
            return False
        if mode == ControlMode.DISABLED:
            return True
        if mode != ControlMode.TARGET_TRACKING:
            return False
        if self._target_tracking_uses_pvt():
            return self._set_pvt_mode()
        return self._set_mit_mode()

    def set_control_mode(self, mode: ControlMode) -> bool:
        if not self._is_enabled or self._eef is None:
            return False
        ok = (
            self._command_pump.request_mode(mode)
            if self._command_pump is not None
            else self._apply_mode_request(mode)
        )
        if not ok:
            return False
        self._control_mode = mode
        if mode == ControlMode.DISABLED:
            self._publish_command(None)
        return True

    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        del cmd

    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        if self._control_mode != ControlMode.TARGET_TRACKING or self._eef is None:
            return
        position_target = np.asarray(cmd.position_target, dtype=np.float64).reshape(-1)
        if position_target.size == 0:
            return
        velocity_target = np.asarray(cmd.velocity_target, dtype=np.float64).reshape(-1)
        if velocity_target.size == 0:
            velocity_target = np.zeros(1, dtype=np.float64)
        feedforward = (
            np.zeros(1, dtype=np.float64)
            if cmd.feedforward is None
            else np.asarray(cmd.feedforward, dtype=np.float64).reshape(-1)
        )
        if feedforward.size == 0:
            feedforward = np.zeros(1, dtype=np.float64)
        if self._target_tracking_uses_pvt():
            velocity_target = np.array([self._pvt_velocity], dtype=np.float64)
            feedforward = np.zeros(1, dtype=np.float64)
            kp = np.zeros(1, dtype=np.float64)
            kd = np.zeros(1, dtype=np.float64)
        else:
            velocity_target = np.zeros(1, dtype=np.float64)
            kp, kd = self._mit_tracking_gains()
        self._publish_command(
            AirbotFixedTrackingIntent(
                position_target=position_target[:1].copy(),
                velocity_target=velocity_target[:1].copy(),
                feedforward=feedforward[:1].copy(),
                kp=kp,
                kd=kd,
                published_at=time.monotonic(),
            ),
        )

    def _control_cycle(
        self,
        command: AirbotFreeDriveIntent | AirbotFixedTrackingIntent | None,
        mode: ControlMode,
        enabled: bool,
    ) -> None:
        if (
            not enabled
            or mode != ControlMode.TARGET_TRACKING
            or self._eef is None
            or not isinstance(command, AirbotFixedTrackingIntent)
        ):
            return
        intent = command
        position_target = np.asarray(intent.position_target, dtype=np.float64).reshape(
            -1
        )
        if position_target.size == 0:
            return
        velocity_target = np.asarray(intent.velocity_target, dtype=np.float64).reshape(
            -1
        )
        if velocity_target.size == 0:
            velocity_target = np.zeros(1, dtype=np.float64)
        effort = (
            np.zeros(1, dtype=np.float64)
            if intent.feedforward is None
            else np.asarray(intent.feedforward, dtype=np.float64).reshape(-1)
        )
        if effort.size == 0:
            effort = np.zeros(1, dtype=np.float64)
        if self._target_tracking_uses_pvt():
            zeros = np.zeros(1, dtype=np.float64)
            current_threshold = np.array([self.PVT_CURRENT_THRESHOLD], dtype=np.float64)
            payload = self._build_command_payload(
                position_target[:1],
                velocity_target[:1],
                effort=zeros,
                kp=zeros,
                kd=zeros,
                current_threshold=current_threshold,
            )
            if payload is None:
                return
            self._record_command_debug(
                "PVT",
                position_target=position_target[:1],
                velocity_target=velocity_target[:1],
                effort=zeros,
                kp=zeros,
                kd=zeros,
                current_threshold=current_threshold,
            )
            self._eef.pvt(payload)
            return
        velocity_target = np.zeros(1, dtype=np.float64)
        kp = (
            np.asarray(intent.kp, dtype=np.float64).reshape(-1)
            if intent.kp is not None
            else np.zeros(0, dtype=np.float64)
        )
        kd = (
            np.asarray(intent.kd, dtype=np.float64).reshape(-1)
            if intent.kd is not None
            else np.zeros(0, dtype=np.float64)
        )
        if kp.size == 0 or kd.size == 0:
            kp, kd = self._mit_tracking_gains()
        current_threshold = np.zeros(1, dtype=np.float64)
        payload = self._build_command_payload(
            position_target[:1],
            velocity_target[:1],
            effort=effort[:1],
            kp=kp[:1],
            kd=kd[:1],
            current_threshold=current_threshold,
        )
        if payload is None:
            return
        self._record_command_debug(
            "MIT",
            position_target=position_target[:1],
            velocity_target=velocity_target[:1],
            effort=effort[:1],
            kp=kp[:1],
            kd=kd[:1],
            current_threshold=current_threshold,
        )
        self._eef.mit(payload)

    def move_to_zero(
        self,
        timeout: float = 5.0,
        tolerance: float = 0.002,
        dt: float | None = None,
    ) -> bool:
        return self.move_to_position(
            np.array([0.0], dtype=np.float64),
            tolerance=tolerance,
            timeout=timeout,
            dt=self._dt if dt is None else dt,
        )

    def _identify_target_position(self, now: float | None = None) -> np.ndarray:
        if now is None:
            now = time.monotonic()
        if self._identify_started_at is None:
            self._identify_started_at = now
        t = now - self._identify_started_at
        lo = float(self.IDENTIFY_OSCILLATION_MIN)
        hi = float(self.IDENTIFY_OSCILLATION_MAX)
        midpoint = 0.5 * (lo + hi)
        amplitude = 0.5 * (hi - lo)
        target = midpoint + amplitude * np.sin(
            2.0 * np.pi * self.IDENTIFY_OSCILLATION_HZ * t
        )
        return np.array([target], dtype=np.float64)

    def identify_start(self) -> bool:
        led_ok = set_airbot_led(self._can_interface, blink_orange=True)
        try:
            if not self._is_open:
                self.open()
            if not self._is_enabled:
                self.enable()
            if self.enter_target_tracking():
                self._identify_started_at = time.monotonic()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        return led_ok

    def identify_step(self) -> None:
        if not self._is_open or not self._is_enabled:
            return
        self.step_target_tracking(
            position_target=self._identify_target_position(),
            velocity_target=np.zeros(1, dtype=np.float64),
            add_gravity_compensation=False,
        )

    def identify_stop(
        self,
        disable_robot: bool = True,
        return_to_zero: bool = True,
        zero_timeout: float = 5.0,
    ) -> bool:
        led_ok = set_airbot_led(self._can_interface, blink_orange=False)
        self._identify_started_at = None
        if self._is_open and disable_robot:
            try:
                if return_to_zero and self._is_enabled:
                    self.move_to_zero(timeout=zero_timeout)
                if self._is_enabled:
                    self.disable()
                self.close()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
        return led_ok


__all__ = [
    "AIRBOTEEFLinearKinematics",
    "AIRBOTE2B",
    "AIRBOTG2",
]
