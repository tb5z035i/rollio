"""Pseudo robot arm — simulated 6-DOF robot for testing.

This module provides a fully functional pseudo robot arm that simulates
all the interfaces of a real robot without requiring hardware.
"""

from __future__ import annotations

import numpy as np

from rollio.defaults import DEFAULT_CONTROL_HZ
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
from rollio.robot.scanner import DetectedRobot
from rollio.utils.time import monotonic_sec

# ═══════════════════════════════════════════════════════════════════════════════
# Pseudo Kinematics Model
# ═══════════════════════════════════════════════════════════════════════════════


class PseudoKinematicsModel(KinematicsModel):
    """Simplified kinematics model for a 6-DOF pseudo robot.

    This is a simplified model that produces plausible but not physically
    accurate results. It's suitable for testing the API but not for
    real robot control.

    The model assumes a generic 6-DOF arm with:
    - Joints 0-2: shoulder/base (larger link lengths)
    - Joints 3-5: wrist (smaller link lengths)
    """

    # Simplified DH-like parameters (just link lengths)
    LINK_LENGTHS = np.array([0.0, 0.3, 0.25, 0.0, 0.0, 0.08])  # meters

    # Gravity vector (pointing down in world frame)
    GRAVITY = np.array([0.0, 0.0, -9.81])

    # Approximate link masses (for gravity compensation)
    LINK_MASSES = np.array([1.0, 2.0, 1.5, 0.5, 0.3, 0.2])  # kg

    def __init__(self, n_dof: int = 6) -> None:
        self._n_dof = n_dof
        self._frame_names = ["frame"]

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def frame_names(self) -> list[str]:
        return self._frame_names

    def forward_kinematics(self, q: np.ndarray, frame: str | None = None) -> Pose:
        """Compute simplified forward kinematics.

        This uses a simplified model that produces reasonable-looking
        frame poses but is not physically accurate.
        """
        del frame
        q = np.asarray(q, dtype=np.float64)
        # Pad to 6-DOF for formulas when used as gripper (n_dof=1)
        q_pad = np.zeros(6)
        q_pad[: min(6, len(q))] = q[:6]

        # Simple FK model: accumulate rotations and translations
        # This is a simplified approximation, not accurate DH

        # Base position
        x, y, z = 0.0, 0.0, 0.1  # Base height

        # Joint 0 rotates around Z (base rotation)
        c0, s0 = np.cos(q_pad[0]), np.sin(q_pad[0])

        # Joint 1 rotates around Y (shoulder pitch)
        c1, s1 = np.cos(q_pad[1]), np.sin(q_pad[1])

        # Joint 2 rotates around Y (elbow pitch)
        _c2, _s2 = np.cos(q_pad[2]), np.sin(q_pad[2])  # reserve for future IK

        # Simplified position calculation
        # Shoulder to elbow
        L1 = self.LINK_LENGTHS[1]
        # Elbow to wrist
        L2 = self.LINK_LENGTHS[2]
        # Wrist to end
        L3 = self.LINK_LENGTHS[5]

        # Forward reach in the plane
        r = (
            L1 * c1
            + L2 * np.cos(q_pad[1] + q_pad[2])
            + L3 * np.cos(q_pad[1] + q_pad[2] + q_pad[3])
        )
        # Height
        h = (
            L1 * s1
            + L2 * np.sin(q_pad[1] + q_pad[2])
            + L3 * np.sin(q_pad[1] + q_pad[2] + q_pad[3])
        )

        # Apply base rotation
        x = r * c0
        y = r * s0
        z = 0.1 + h  # Add base height

        position = np.array([x, y, z])

        # Simplified orientation (accumulated rotations)
        # For simplicity, use axis-angle -> quaternion
        angle = q_pad[0] + q_pad[4]  # Simplified roll
        axis = np.array([0.0, 0.0, 1.0])

        # Convert axis-angle to quaternion
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        quaternion = np.array([w, xyz[0], xyz[1], xyz[2]])

        return Pose(position=position, quaternion=quaternion)

    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: np.ndarray | None = None,
        frame: str | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        """Simplified inverse kinematics using Jacobian pseudo-inverse.

        This is a basic iterative IK solver that may not always converge.
        """
        del frame
        if q_init is None:
            q = np.zeros(self._n_dof)
        else:
            q = np.array(q_init, dtype=np.float64)

        target_pos = target_pose.position

        for _ in range(max_iterations):
            current_pose = self.forward_kinematics(q)

            # Position error
            pos_error = target_pos - current_pose.position

            # Check convergence (position only for simplicity)
            if np.linalg.norm(pos_error) < tolerance:
                return q, True

            # Jacobian
            J = self.jacobian(q)
            J_pos = J[:3, :]  # Position part only

            # Pseudo-inverse
            J_pinv = np.linalg.pinv(J_pos)

            # Update
            dq = J_pinv @ pos_error
            q = q + 0.5 * dq  # Damped update

            # Clamp joint angles
            q = np.clip(q, -np.pi, np.pi)

        return q, False

    def jacobian(self, q: np.ndarray, frame: str | None = None) -> np.ndarray:
        """Compute numerical Jacobian via finite differences.

        This is a simple numerical approximation.
        """
        del frame
        q = np.asarray(q, dtype=np.float64)
        eps = 1e-6

        J = np.zeros((6, self._n_dof))

        pose_0 = self.forward_kinematics(q)
        pos_0 = pose_0.position
        quat_0 = pose_0.quaternion

        for i in range(self._n_dof):
            q_plus = q.copy()
            q_plus[i] += eps

            pose_plus = self.forward_kinematics(q_plus)

            # Linear part
            J[:3, i] = (pose_plus.position - pos_0) / eps

            # Angular part (simplified using quaternion difference)
            dquat = pose_plus.quaternion - quat_0
            J[3:6, i] = dquat[1:4] / eps * 2  # Approximate

        return J

    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        """Simplified inverse dynamics for gravity compensation.

        This provides a rough approximation of gravity torques.
        Real implementations should use Pinocchio or KDL.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)

        tau = np.zeros(self._n_dof)
        # Pad q for formulas when n_dof < 6 (e.g. gripper)
        q_pad = np.zeros(6)
        q_pad[: min(6, len(q))] = q[:6]

        # Simplified gravity compensation model
        # Joints 1 and 2 (shoulder and elbow pitch) bear most gravity load

        g = 9.81

        if self._n_dof >= 2:
            # Joint 1: shoulder pitch - affected by links 1, 2, and EE
            m_total = self.LINK_MASSES[1:].sum()
            L_eff = (self.LINK_LENGTHS[1] + self.LINK_LENGTHS[2]) / 2
            tau[1] = m_total * g * L_eff * np.cos(q_pad[1])

        if self._n_dof >= 3:
            # Joint 2: elbow pitch - affected by links 2 and EE
            m_distal = self.LINK_MASSES[2:].sum()
            L_eff2 = self.LINK_LENGTHS[2] / 2
            tau[2] = m_distal * g * L_eff2 * np.cos(q_pad[1] + q_pad[2])

        if self._n_dof >= 4:
            # Joint 3: wrist pitch
            m_wrist = self.LINK_MASSES[3:].sum()
            tau[3] = m_wrist * g * 0.05 * np.cos(q_pad[1] + q_pad[2] + q_pad[3])

        # Add inertia effects if accelerations are non-zero
        if np.any(qdd != 0):
            # Simplified inertia (diagonal approximation)
            inertias = np.array([0.1, 0.5, 0.3, 0.05, 0.02, 0.01])
            tau += inertias[: self._n_dof] * qdd

        # Add velocity-dependent friction
        if np.any(qd != 0):
            friction = 0.1 * qd  # Viscous friction
            tau += friction

        return tau


# ═══════════════════════════════════════════════════════════════════════════════
# Pseudo Robot Arm
# ═══════════════════════════════════════════════════════════════════════════════


class PseudoRobotArm(RobotArm):
    """Simulated 6-DOF robot arm for testing.

    This robot simulates:
    - Joint state feedback with configurable noise
    - All control modes (free drive, target tracking)
    - Kinematics computations via PseudoKinematicsModel
    - Realistic dynamics simulation
    """

    ROBOT_TYPE = "pseudo"
    DIRECT_MAP_ALLOWLIST: tuple[str, ...] = ()

    # ── Class methods ─────────────────────────────────────────────────────

    @classmethod
    def scan(cls) -> list[DetectedRobot]:
        """Return a single pseudo robot."""
        return [
            DetectedRobot(
                robot_type=cls.ROBOT_TYPE,
                device_id=0,
                label="Pseudo Robot Arm (6-DOF simulation)",
                n_dof=6,
                properties={"type": "pseudo", "simulated": True},
            )
        ]

    @classmethod
    def probe(cls, _: int | str) -> bool:
        """Pseudo robot is always available."""
        return True

    # ── Instance initialization ───────────────────────────────────────────

    def __init__(
        self,
        name: str = "pseudo_arm",
        n_dof: int = 6,
        noise_level: float = 0.001,
        control_frequency: float = DEFAULT_CONTROL_HZ,
    ) -> None:
        """Initialize the pseudo robot arm.

        Args:
            name: Robot name
            n_dof: Number of degrees of freedom
            noise_level: Standard deviation of measurement noise (rad)
            control_frequency: Simulated control frequency (Hz)
        """
        self._name = name
        self._n_dof = n_dof
        self._noise_level = noise_level
        self._dt = 1.0 / control_frequency

        # State
        self._is_open = False
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

        # Simulated joint state
        self._q = np.zeros(n_dof)  # Position
        self._qd = np.zeros(n_dof)  # Velocity
        self._tau = np.zeros(n_dof)  # Torque

        # Kinematics model
        self._kinematics = PseudoKinematicsModel(n_dof)

        # Dynamics parameters (tuned for stable, responsive simulation)
        self._inertia = np.array([0.1, 0.2, 0.15, 0.05, 0.03, 0.02])[:n_dof]
        self._damping = np.array([2.0, 4.0, 3.0, 1.0, 0.5, 0.3])[:n_dof]
        self._max_qdd = 100.0  # Max acceleration for stability

        # Random state for noise
        self._rng = np.random.default_rng()

        # Info
        self._info = RobotInfo(
            name=name,
            robot_type=self.ROBOT_TYPE,
            n_dof=n_dof,
            feedback_capabilities={
                FeedbackCapability.POSITION,
                FeedbackCapability.VELOCITY,
                FeedbackCapability.EFFORT,
                FeedbackCapability.FRAME_POSE,
                FeedbackCapability.FRAME_TWIST,
            },
            properties={
                "simulated": True,
                "noise_level": noise_level,
                "control_frequency": control_frequency,
            },
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def kinematics(self) -> KinematicsModel:
        return self._kinematics

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Initialize the pseudo robot."""
        self._is_open = True
        self._q = np.zeros(self._n_dof)
        self._qd = np.zeros(self._n_dof)
        self._tau = np.zeros(self._n_dof)

    def close(self) -> None:
        """Close the pseudo robot."""
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED
        self._is_open = False

    def enable(self) -> bool:
        """Enable the pseudo robot."""
        if not self._is_open:
            return False
        self._is_enabled = True
        return True

    def disable(self) -> None:
        """Disable the pseudo robot."""
        self._is_enabled = False
        self._control_mode = ControlMode.DISABLED

    # ── State Reading ─────────────────────────────────────────────────────

    def read_joint_state(self) -> JointState:
        """Read current joint state with optional noise."""
        ts = monotonic_sec()

        # Add measurement noise
        noise_pos = self._rng.normal(0, self._noise_level, self._n_dof)
        noise_vel = self._rng.normal(0, self._noise_level * 10, self._n_dof)

        return JointState(
            timestamp=ts,
            position=self._q.copy() + noise_pos,
            velocity=self._qd.copy() + noise_vel,
            effort=self._tau.copy(),
            is_valid=self._is_open,
        )

    # ── Control Mode Setting ──────────────────────────────────────────────

    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode."""
        if not self._is_enabled:
            return False
        self._control_mode = mode
        return True

    # ── Free Drive Mode ───────────────────────────────────────────────────

    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        """Execute a free drive command.

        In free drive mode, we simulate:
        1. Gravity compensation
        2. External wrench (if provided)
        3. Integration of the resulting motion
        """
        if self._control_mode != ControlMode.FREE_DRIVE:
            return

        # Compute gravity compensation
        tau_gravity = self._kinematics.gravity_compensation(self._q)
        tau_gravity *= cmd.gravity_compensation_scale

        # Compute external wrench contribution
        tau_ext = np.zeros(self._n_dof)
        if cmd.external_wrench is not None:
            tau_ext = self._kinematics.wrench_to_joint_torques(
                self._q, cmd.external_wrench
            )

        # Total control torque (gravity comp + external)
        tau_control = tau_gravity + tau_ext

        # Simulate dynamics: M*qdd = tau_ext - D*qd
        # (gravity is compensated, so we only feel external forces)
        qdd = (tau_ext - self._damping * self._qd) / self._inertia

        # Clamp acceleration for numerical stability
        qdd = np.clip(qdd, -self._max_qdd, self._max_qdd)

        # Integrate
        self._qd += qdd * self._dt
        self._q += self._qd * self._dt

        # Clamp joint angles
        self._q = np.clip(self._q, -np.pi, np.pi)

        # Store applied torque
        self._tau = tau_control

    # ── Target Tracking Mode ──────────────────────────────────────────────

    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        """Execute a target tracking command.

        MIT control law: tau = kp*(q_target - q) + kd*(qd_target - qd) + ff

        The simulation assumes feedforward already compensates for gravity,
        so dynamics are: M*qdd = tau_pd + (ff - g(q)) - D*qd
        If ff = g(q), then M*qdd = tau_pd - D*qd (ideal tracking)
        """
        if self._control_mode != ControlMode.TARGET_TRACKING:
            return

        # Compute PD control torques
        pos_error = cmd.position_target - self._q
        vel_error = cmd.velocity_target - self._qd

        tau_pd = cmd.kp * pos_error + cmd.kd * vel_error

        # Add feedforward (should include gravity compensation)
        tau_ff = np.zeros(self._n_dof)
        if cmd.feedforward is not None:
            tau_ff = cmd.feedforward

        tau_control = tau_pd + tau_ff

        # Simulate dynamics: M*qdd = tau_control - g(q) - D*qd
        # If feedforward contains gravity comp, net gravity effect is small
        tau_gravity = self._kinematics.gravity_compensation(self._q)

        # Net torque after accounting for gravity
        tau_net = tau_control - tau_gravity - self._damping * self._qd
        qdd = tau_net / self._inertia

        # Clamp acceleration for numerical stability
        qdd = np.clip(qdd, -self._max_qdd, self._max_qdd)

        # Integrate with velocity damping for stability
        self._qd += qdd * self._dt
        self._qd *= 0.995  # Small velocity decay for stability
        self._q += self._qd * self._dt

        # Clamp joint angles
        self._q = np.clip(self._q, -np.pi, np.pi)

        # Store applied torque
        self._tau = tau_control

    # ── Convenience methods for testing ───────────────────────────────────

    def set_joint_position(self, q: np.ndarray) -> None:
        """Directly set joint positions (for testing)."""
        self._q = np.asarray(q, dtype=np.float64)
        self._qd = np.zeros(self._n_dof)

    def get_raw_position(self) -> np.ndarray:
        """Get raw joint positions without noise (for testing)."""
        return self._q.copy()

    def get_raw_velocity(self) -> np.ndarray:
        """Get raw joint velocities without noise (for testing)."""
        return self._qd.copy()
