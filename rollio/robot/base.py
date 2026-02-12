"""Abstract base classes for robot arms.

This module defines the core abstractions for robot arm control, including:
- State representations (joint states, end-effector states)
- Control modes (free drive, target tracking)
- Kinematics interfaces (FK, IK, Jacobian, inverse dynamics)
- Robot arm interface with feedback and control capabilities
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from rollio.robot.scanner import DetectedRobot


# ═══════════════════════════════════════════════════════════════════════════════
# Enums and Constants
# ═══════════════════════════════════════════════════════════════════════════════


class ControlMode(Enum):
    """Supported robot control modes."""
    DISABLED = auto()       # Robot is disabled, no control
    FREE_DRIVE = auto()     # Gravity-compensated free drive (admittance control)
    TARGET_TRACKING = auto()  # MIT-style PD control with feedforward


class FeedbackCapability(Enum):
    """Types of feedback a robot may provide."""
    POSITION = auto()
    VELOCITY = auto()
    EFFORT = auto()
    END_POSE = auto()
    END_TWIST = auto()
    END_WRENCH = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# State Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JointState:
    """State of all joints at a given timestamp.
    
    All arrays have shape (n_dof,).
    """
    timestamp: float
    position: np.ndarray | None = None   # rad
    velocity: np.ndarray | None = None   # rad/s
    effort: np.ndarray | None = None     # Nm
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Ensure arrays are float32 for efficiency."""
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=np.float32)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.effort is not None:
            self.effort = np.asarray(self.effort, dtype=np.float32)


@dataclass
class Pose:
    """6-DOF pose (position + orientation).
    
    Position is (x, y, z) in meters.
    Orientation is stored as quaternion in scalar-first (w, x, y, z) format.
    
    Uses scipy.spatial.transform.Rotation for all orientation operations.
    """
    position: np.ndarray        # (3,) xyz in meters
    quaternion: np.ndarray      # (4,) wxyz quaternion (scalar-first)
    
    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64)
        # Normalize quaternion using scipy
        # Convert wxyz -> xyzw for scipy, normalize, convert back
        quat_xyzw = self.quaternion[[1, 2, 3, 0]]
        rot = Rotation.from_quat(quat_xyzw)
        quat_xyzw_normalized = rot.as_quat()
        self.quaternion = quat_xyzw_normalized[[3, 0, 1, 2]]  # Back to wxyz

    @property
    def rotation(self) -> Rotation:
        """Get scipy Rotation object."""
        # Convert wxyz -> xyzw for scipy
        quat_xyzw = self.quaternion[[1, 2, 3, 0]]
        return Rotation.from_quat(quat_xyzw)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix using scipy."""
        return self.rotation.as_matrix()

    @property
    def euler_xyz(self) -> np.ndarray:
        """Get Euler angles in XYZ (roll-pitch-yaw) convention."""
        return self.rotation.as_euler('xyz')

    @property
    def euler_zyx(self) -> np.ndarray:
        """Get Euler angles in ZYX (yaw-pitch-roll) convention."""
        return self.rotation.as_euler('zyx')

    @classmethod
    def from_matrix(cls, position: np.ndarray, rotation: np.ndarray) -> "Pose":
        """Create Pose from position and 3x3 rotation matrix."""
        rot = Rotation.from_matrix(rotation)
        quat_xyzw = rot.as_quat()
        # Convert xyzw -> wxyz
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return cls(position=position, quaternion=quat_wxyz)

    @classmethod
    def from_rotation(cls, position: np.ndarray, rotation: Rotation) -> "Pose":
        """Create Pose from position and scipy Rotation."""
        quat_xyzw = rotation.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return cls(position=position, quaternion=quat_wxyz)

    @classmethod
    def from_euler(
        cls, 
        position: np.ndarray, 
        angles: np.ndarray, 
        seq: str = 'xyz'
    ) -> "Pose":
        """Create Pose from position and Euler angles.
        
        Args:
            position: (3,) position in meters
            angles: (3,) Euler angles in radians
            seq: Euler angle sequence (e.g., 'xyz', 'zyx')
        """
        rot = Rotation.from_euler(seq, angles)
        return cls.from_rotation(position, rot)

    @classmethod
    def identity(cls, position: np.ndarray | None = None) -> "Pose":
        """Create Pose with identity orientation."""
        if position is None:
            position = np.zeros(3)
        return cls(position=position, quaternion=np.array([1.0, 0.0, 0.0, 0.0]))

    def as_homogeneous(self) -> np.ndarray:
        """Return 4x4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T

    def inverse(self) -> "Pose":
        """Compute the inverse transformation."""
        rot_inv = self.rotation.inv()
        pos_inv = -rot_inv.apply(self.position)
        return Pose.from_rotation(pos_inv, rot_inv)

    def __matmul__(self, other: "Pose") -> "Pose":
        """Compose two poses: self @ other."""
        rot_composed = self.rotation * other.rotation
        pos_composed = self.position + self.rotation.apply(other.position)
        return Pose.from_rotation(pos_composed, rot_composed)


@dataclass
class Twist:
    """6-DOF spatial velocity (linear + angular).
    
    Linear velocity is (vx, vy, vz) in m/s.
    Angular velocity is (wx, wy, wz) in rad/s.
    """
    linear: np.ndarray     # (3,) m/s
    angular: np.ndarray    # (3,) rad/s

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=np.float64)
        self.angular = np.asarray(self.angular, dtype=np.float64)

    def as_vector(self) -> np.ndarray:
        """Return as 6D vector [linear, angular]."""
        return np.concatenate([self.linear, self.angular])


@dataclass
class Wrench:
    """6-DOF spatial force (force + torque).
    
    Force is (fx, fy, fz) in N.
    Torque is (tx, ty, tz) in Nm.
    """
    force: np.ndarray      # (3,) N
    torque: np.ndarray     # (3,) Nm

    def __post_init__(self) -> None:
        self.force = np.asarray(self.force, dtype=np.float64)
        self.torque = np.asarray(self.torque, dtype=np.float64)

    def as_vector(self) -> np.ndarray:
        """Return as 6D vector [force, torque]."""
        return np.concatenate([self.force, self.torque])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Wrench":
        """Create Wrench from 6D vector."""
        return cls(force=vec[:3], torque=vec[3:])

    @classmethod
    def zero(cls) -> "Wrench":
        """Create zero wrench."""
        return cls(force=np.zeros(3), torque=np.zeros(3))


@dataclass
class EndEffectorState:
    """State of an end-effector at a given timestamp."""
    name: str                           # e.g., "gripper", "left_hand"
    timestamp: float
    pose: Pose | None = None            # Current pose
    twist: Twist | None = None          # Current velocity
    wrench: Wrench | None = None        # Current force/torque (if F/T sensor available)


@dataclass
class RobotState:
    """Complete robot state at a given timestamp."""
    timestamp: float
    joint_state: JointState
    end_effectors: list[EndEffectorState] = field(default_factory=list)
    control_mode: ControlMode = ControlMode.DISABLED
    is_valid: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Control Command Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TargetTrackingCommand:
    """Command for target tracking (MIT) mode.
    
    MIT control law: tau = kp * (q_target - q) + kd * (qd_target - qd) + ff
    
    All arrays have shape (n_dof,).
    """
    position_target: np.ndarray      # rad - target joint positions
    velocity_target: np.ndarray      # rad/s - target joint velocities
    kp: np.ndarray                   # Position gain per joint
    kd: np.ndarray                   # Velocity gain per joint
    feedforward: np.ndarray | None = None  # Additional feedforward torque (e.g., gravity comp)

    def __post_init__(self) -> None:
        self.position_target = np.asarray(self.position_target, dtype=np.float64)
        self.velocity_target = np.asarray(self.velocity_target, dtype=np.float64)
        self.kp = np.asarray(self.kp, dtype=np.float64)
        self.kd = np.asarray(self.kd, dtype=np.float64)
        if self.feedforward is not None:
            self.feedforward = np.asarray(self.feedforward, dtype=np.float64)


@dataclass
class FreeDriveCommand:
    """Command for free drive mode.
    
    In free drive mode, the robot compensates for gravity and optionally
    applies an external wrench transformed to joint torques via Jacobian.
    """
    external_wrench: Wrench | None = None  # External wrench at end-effector
    gravity_compensation_scale: float = 1.0  # Scale factor for gravity comp


# ═══════════════════════════════════════════════════════════════════════════════
# Robot Info
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RobotInfo:
    """Metadata describing a robot arm."""
    name: str
    robot_type: str                    # "pseudo", "airbot_play", etc.
    n_dof: int                         # Degrees of freedom
    feedback_capabilities: set[FeedbackCapability] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)

    def has_feedback(self, capability: FeedbackCapability) -> bool:
        """Check if robot provides specific feedback type."""
        return capability in self.feedback_capabilities


# ═══════════════════════════════════════════════════════════════════════════════
# Kinematics Model Abstract Class
# ═══════════════════════════════════════════════════════════════════════════════


class KinematicsModel(ABC):
    """Abstract interface for robot kinematics computations.
    
    Provides forward/inverse kinematics, Jacobian computation, and
    inverse dynamics for gravity compensation.
    
    Implementations may use Pinocchio, KDL, or custom solutions.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        ...

    @property
    @abstractmethod
    def end_effector_names(self) -> list[str]:
        """Names of all end-effectors."""
        ...

    # ── Forward Kinematics ────────────────────────────────────────────────

    @abstractmethod
    def forward_kinematics(
        self, 
        q: np.ndarray, 
        end_effector: str | None = None
    ) -> Pose:
        """Compute end-effector pose given joint positions.
        
        Args:
            q: Joint positions (n_dof,)
            end_effector: Name of end-effector (None for default/first)
            
        Returns:
            Pose of the end-effector
        """
        ...

    def forward_kinematics_all(self, q: np.ndarray) -> dict[str, Pose]:
        """Compute poses for all end-effectors.
        
        Args:
            q: Joint positions (n_dof,)
            
        Returns:
            Dict mapping end-effector names to their poses
        """
        return {
            name: self.forward_kinematics(q, name) 
            for name in self.end_effector_names
        }

    # ── Inverse Kinematics ────────────────────────────────────────────────

    @abstractmethod
    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: np.ndarray | None = None,
        end_effector: str | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        """Compute joint positions for desired end-effector pose.
        
        Args:
            target_pose: Desired end-effector pose
            q_init: Initial guess for joint positions
            end_effector: Name of end-effector
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            
        Returns:
            (joint_positions, success) - positions may be None if failed
        """
        ...

    # ── Jacobian ──────────────────────────────────────────────────────────

    @abstractmethod
    def jacobian(
        self, 
        q: np.ndarray, 
        end_effector: str | None = None
    ) -> np.ndarray:
        """Compute the geometric Jacobian at given configuration.
        
        The Jacobian maps joint velocities to end-effector twist:
            v_ee = J @ q_dot
        
        Args:
            q: Joint positions (n_dof,)
            end_effector: Name of end-effector
            
        Returns:
            Jacobian matrix (6, n_dof)
        """
        ...

    def jacobian_transpose(
        self, 
        q: np.ndarray, 
        end_effector: str | None = None
    ) -> np.ndarray:
        """Compute transpose of Jacobian (for force transformation).
        
        Maps end-effector wrench to joint torques:
            tau = J^T @ F
        """
        return self.jacobian(q, end_effector).T

    # ── Inverse Dynamics ──────────────────────────────────────────────────

    @abstractmethod
    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        """Compute joint torques via inverse dynamics.
        
        tau = M(q) @ qdd + C(q, qd) @ qd + g(q)
        
        For gravity compensation, pass zero velocities and accelerations.
        
        Args:
            q: Joint positions (n_dof,)
            qd: Joint velocities (n_dof,)
            qdd: Joint accelerations (n_dof,)
            
        Returns:
            Joint torques (n_dof,)
        """
        ...

    def gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques.
        
        Equivalent to inverse_dynamics(q, zeros, zeros).
        """
        zeros = np.zeros(self.n_dof)
        return self.inverse_dynamics(q, zeros, zeros)

    # ── Wrench to Joint Torques ───────────────────────────────────────────

    def wrench_to_joint_torques(
        self,
        q: np.ndarray,
        wrench: Wrench,
        end_effector: str | None = None,
    ) -> np.ndarray:
        """Transform end-effector wrench to joint torques via Jacobian transpose.
        
        tau = J^T @ F
        
        Args:
            q: Joint positions (n_dof,)
            wrench: External wrench at end-effector
            end_effector: Name of end-effector
            
        Returns:
            Joint torques (n_dof,)
        """
        J_T = self.jacobian_transpose(q, end_effector)
        return J_T @ wrench.as_vector()


# ═══════════════════════════════════════════════════════════════════════════════
# Robot Arm Abstract Base Class
# ═══════════════════════════════════════════════════════════════════════════════


class RobotArm(ABC):
    """Abstract interface for robot arm control and feedback.
    
    A robot arm provides:
    - Joint state feedback (position, velocity, effort)
    - End-effector state feedback (pose, twist, wrench)
    - Multiple control modes (free drive, target tracking)
    - Kinematics computations via an associated KinematicsModel
    
    Subclasses should implement class methods for device scanning:
    - scan(): Detect available robots of this type
    - probe(): Check if a specific device exists
    """

    # ── Class-level type identifier ───────────────────────────────────────

    ROBOT_TYPE: ClassVar[str] = "unknown"  # Override in subclasses

    # ── Factory / scanning class methods ──────────────────────────────────

    @classmethod
    def scan(cls) -> list["DetectedRobot"]:
        """Scan for available robots of this type.
        
        Returns a list of DetectedRobot objects.
        Subclasses should override this method.
        """
        return []

    @classmethod
    def probe(cls, device_id: int | str) -> bool:
        """Check if a specific robot device exists.
        
        Args:
            device_id: Device identifier (CAN interface, serial port, etc.)
            
        Returns:
            True if device exists and is accessible
        """
        return False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (joints)."""
        ...

    @property
    @abstractmethod
    def info(self) -> RobotInfo:
        """Get robot metadata and capabilities."""
        ...

    @property
    def properties(self) -> dict[str, Any]:
        """Unstructured properties dictionary for robot-specific data.
        
        This provides a flexible way to store and access robot-specific
        properties that may not be shared across all robot types.
        
        Common properties might include:
        - "serial_number": Robot serial number
        - "end_effector_type": Type of attached end effector
        - "firmware_version": Robot firmware version
        - "hardware_version": Hardware revision
        
        Subclasses should override this to provide actual values.
        
        Returns:
            Dictionary of property name -> value
        """
        return {}

    def query_properties(self) -> dict[str, Any]:
        """Query and update robot properties from hardware.
        
        This method actively queries the robot hardware for current
        property values. Subclasses should override this to implement
        hardware-specific queries.
        
        Returns:
            Updated properties dictionary
        """
        return self.properties

    @property
    @abstractmethod
    def kinematics(self) -> KinematicsModel:
        """Get the kinematics model for this robot."""
        ...

    @property
    @abstractmethod
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        ...

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the robot is enabled and ready for commands."""
        ...

    # ── Feedback Availability ─────────────────────────────────────────────

    @property
    def has_position_feedback(self) -> bool:
        """Whether joint position feedback is available."""
        return FeedbackCapability.POSITION in self.info.feedback_capabilities

    @property
    def has_velocity_feedback(self) -> bool:
        """Whether joint velocity feedback is available."""
        return FeedbackCapability.VELOCITY in self.info.feedback_capabilities

    @property
    def has_effort_feedback(self) -> bool:
        """Whether joint effort (torque) feedback is available."""
        return FeedbackCapability.EFFORT in self.info.feedback_capabilities

    @property
    def has_end_pose_feedback(self) -> bool:
        """Whether end-effector pose feedback is available."""
        return FeedbackCapability.END_POSE in self.info.feedback_capabilities

    @property
    def has_end_twist_feedback(self) -> bool:
        """Whether end-effector twist feedback is available."""
        return FeedbackCapability.END_TWIST in self.info.feedback_capabilities

    @property
    def has_end_wrench_feedback(self) -> bool:
        """Whether end-effector wrench feedback is available (F/T sensor)."""
        return FeedbackCapability.END_WRENCH in self.info.feedback_capabilities

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None:
        """Initialize the robot connection."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the robot connection and release resources."""
        ...

    @abstractmethod
    def enable(self) -> bool:
        """Enable the robot motors.
        
        Returns:
            True if successfully enabled
        """
        ...

    @abstractmethod
    def disable(self) -> None:
        """Disable the robot motors (safe state)."""
        ...

    # ── State Reading ─────────────────────────────────────────────────────

    @abstractmethod
    def read_joint_state(self) -> JointState:
        """Read current joint state (position, velocity, effort)."""
        ...

    def read_end_effector_state(
        self, 
        end_effector: str | None = None
    ) -> EndEffectorState:
        """Read current end-effector state.
        
        Default implementation computes pose/twist from joint state via FK.
        Override for robots with direct end-effector sensing (e.g., F/T sensor).
        """
        joint_state = self.read_joint_state()
        
        # Get end-effector name
        ee_names = self.kinematics.end_effector_names
        if end_effector is None:
            end_effector = ee_names[0] if ee_names else "end_effector"
        
        pose = None
        twist = None
        
        if joint_state.position is not None:
            pose = self.kinematics.forward_kinematics(
                joint_state.position, end_effector
            )
            
            if joint_state.velocity is not None:
                J = self.kinematics.jacobian(joint_state.position, end_effector)
                twist_vec = J @ joint_state.velocity
                twist = Twist(
                    linear=twist_vec[:3],
                    angular=twist_vec[3:]
                )
        
        return EndEffectorState(
            name=end_effector,
            timestamp=joint_state.timestamp,
            pose=pose,
            twist=twist,
            wrench=None  # Override in subclass if F/T sensor available
        )

    def read_state(self) -> RobotState:
        """Read complete robot state."""
        joint_state = self.read_joint_state()
        
        # Compute end-effector states for all end-effectors
        end_effector_states = []
        for ee_name in self.kinematics.end_effector_names:
            ee_state = self.read_end_effector_state(ee_name)
            end_effector_states.append(ee_state)
        
        return RobotState(
            timestamp=joint_state.timestamp,
            joint_state=joint_state,
            end_effectors=end_effector_states,
            control_mode=self.control_mode,
            is_valid=joint_state.is_valid
        )

    # ── Control Mode Setting ──────────────────────────────────────────────

    @abstractmethod
    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode.
        
        Args:
            mode: Desired control mode
            
        Returns:
            True if mode was successfully set
        """
        ...

    # ── Free Drive Mode ───────────────────────────────────────────────────

    def enter_free_drive(self) -> bool:
        """Enter free drive (gravity-compensated) mode.
        
        In this mode, the robot compensates for gravity so the user
        can manually guide the arm.
        
        Returns:
            True if successfully entered free drive mode
        """
        return self.set_control_mode(ControlMode.FREE_DRIVE)

    @abstractmethod
    def command_free_drive(self, cmd: FreeDriveCommand) -> None:
        """Send a free drive command.
        
        The robot will:
        1. Compute gravity compensation torques via inverse dynamics
        2. If external wrench is provided, transform it to joint torques via Jacobian
        3. Apply the combined torques
        
        Args:
            cmd: Free drive command with optional external wrench
        """
        ...

    def step_free_drive(
        self,
        external_wrench: Wrench | None = None,
        gravity_scale: float = 1.0,
    ) -> None:
        """Convenience method for free drive step.
        
        Args:
            external_wrench: Optional external wrench at end-effector
            gravity_scale: Scale factor for gravity compensation
        """
        cmd = FreeDriveCommand(
            external_wrench=external_wrench,
            gravity_compensation_scale=gravity_scale
        )
        self.command_free_drive(cmd)

    # ── Target Tracking Mode (MIT Mode) ───────────────────────────────────

    def enter_target_tracking(self) -> bool:
        """Enter target tracking (MIT) mode.
        
        In this mode, the robot tracks position/velocity targets using
        PD control with feedforward: tau = kp*(q_target - q) + kd*(qd_target - qd) + ff
        
        Returns:
            True if successfully entered target tracking mode
        """
        return self.set_control_mode(ControlMode.TARGET_TRACKING)

    @abstractmethod
    def command_target_tracking(self, cmd: TargetTrackingCommand) -> None:
        """Send a target tracking command.
        
        The robot will compute: tau = kp*(q_target - q) + kd*(qd_target - qd) + ff
        
        Args:
            cmd: Target tracking command with targets and gains
        """
        ...

    def step_target_tracking(
        self,
        position_target: np.ndarray,
        velocity_target: np.ndarray | None = None,
        kp: np.ndarray | float = 10.0,
        kd: np.ndarray | float = 1.0,
        feedforward: np.ndarray | None = None,
        add_gravity_compensation: bool = True,
    ) -> None:
        """Convenience method for target tracking step.
        
        Args:
            position_target: Target joint positions (n_dof,)
            velocity_target: Target joint velocities (n_dof,), defaults to zeros
            kp: Position gains (scalar or per-joint)
            kd: Velocity gains (scalar or per-joint)
            feedforward: Additional feedforward torques
            add_gravity_compensation: If True, add gravity compensation to feedforward
        """
        n = self.n_dof
        
        if velocity_target is None:
            velocity_target = np.zeros(n)
        
        # Convert scalar gains to arrays
        if np.isscalar(kp):
            kp = np.full(n, kp)
        if np.isscalar(kd):
            kd = np.full(n, kd)
        
        # Compute feedforward with optional gravity compensation
        if feedforward is None:
            feedforward = np.zeros(n)
        
        if add_gravity_compensation:
            joint_state = self.read_joint_state()
            if joint_state.position is not None:
                gravity_torques = self.kinematics.gravity_compensation(
                    joint_state.position
                )
                feedforward = feedforward + gravity_torques
        
        cmd = TargetTrackingCommand(
            position_target=position_target,
            velocity_target=velocity_target,
            kp=kp,
            kd=kd,
            feedforward=feedforward
        )
        self.command_target_tracking(cmd)

    # ── Move to Target (Blocking) ─────────────────────────────────────────

    def move_to_position(
        self,
        target_position: np.ndarray,
        kp: np.ndarray | float = 10.0,
        kd: np.ndarray | float = 2.0,
        tolerance: float = 0.01,
        timeout: float = 10.0,
        dt: float = 0.004,
    ) -> bool:
        """Move to target position (blocking).
        
        Uses target tracking mode to move to the target position.
        
        Args:
            target_position: Target joint positions (n_dof,)
            kp: Position gains
            kd: Velocity gains
            tolerance: Position error tolerance (rad)
            timeout: Maximum time to wait (s)
            dt: Control loop period (s)
            
        Returns:
            True if target reached within tolerance
        """
        import time
        
        if not self.enter_target_tracking():
            return False
        
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < timeout:
            self.step_target_tracking(
                position_target=target_position,
                velocity_target=np.zeros(self.n_dof),
                kp=kp,
                kd=kd,
                add_gravity_compensation=True
            )
            
            joint_state = self.read_joint_state()
            if joint_state.position is not None:
                error = np.abs(target_position - joint_state.position).max()
                if error < tolerance:
                    return True
            
            time.sleep(dt)
        
        return False

    # ── Identification ────────────────────────────────────────────────────

    def identify_start(self) -> bool:
        """Start visual identification of this robot.
        
        This method triggers a visual indicator (e.g., LED blinking) so the
        user can identify which physical robot this instance corresponds to.
        
        Returns:
            True if identification started successfully
        """
        # Default implementation does nothing
        return False

    def identify_stop(self) -> bool:
        """Stop visual identification of this robot.
        
        Returns the visual indicator to its normal state.
        
        Returns:
            True if identification stopped successfully
        """
        # Default implementation does nothing
        return False

    # ── Context Manager Support ───────────────────────────────────────────

    def __enter__(self) -> "RobotArm":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable()
        self.close()
