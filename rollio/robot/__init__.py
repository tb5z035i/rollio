"""Robot arm module for rollio.

This module provides abstractions for robot arm control, including:
- State representations (joint states, end-effector states)
- Control modes (free drive, target tracking)
- Kinematics interfaces (FK, IK, Jacobian, inverse dynamics)
"""
from rollio.robot.base import (
    # Enums
    ControlMode,
    FeedbackCapability,
    # State dataclasses
    EndEffectorState,
    JointState,
    Pose,
    RobotInfo,
    RobotState,
    Twist,
    Wrench,
    # Command dataclasses
    FreeDriveCommand,
    TargetTrackingCommand,
    # Abstract classes
    KinematicsModel,
    RobotArm,
)
from rollio.robot.pseudo_robot import PseudoKinematicsModel, PseudoRobotArm
from rollio.robot.scanner import DetectedRobot, scan_robots

# Optional imports for hardware-specific implementations
try:
    from rollio.robot.airbot_play import AIRBOTPlay, is_airbot_available
except ImportError:
    AIRBOTPlay = None  # type: ignore
    is_airbot_available = lambda: False  # type: ignore

try:
    from rollio.robot.pinocchio_kinematics import (
        PinocchioKinematicsModel,
        is_pinocchio_available,
    )
except ImportError:
    PinocchioKinematicsModel = None  # type: ignore
    is_pinocchio_available = lambda: False  # type: ignore

# CAN utilities
from rollio.robot.can_utils import (
    is_python_can_available,
    scan_can_interfaces,
)

__all__ = [
    # Enums
    "ControlMode",
    "FeedbackCapability",
    # State dataclasses
    "EndEffectorState",
    "JointState",
    "Pose",
    "RobotInfo",
    "RobotState",
    "Twist",
    "Wrench",
    # Command dataclasses
    "FreeDriveCommand",
    "TargetTrackingCommand",
    # Abstract classes
    "KinematicsModel",
    "RobotArm",
    # Implementations
    "PseudoKinematicsModel",
    "PseudoRobotArm",
    "AIRBOTPlay",
    "PinocchioKinematicsModel",
    # Availability checks
    "is_airbot_available",
    "is_pinocchio_available",
    "is_python_can_available",
    # Scanner
    "DetectedRobot",
    "scan_robots",
    "scan_can_interfaces",
]
