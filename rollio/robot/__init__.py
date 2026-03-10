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
    FrameState,
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
    from rollio.robot.airbot.shared import is_airbot_available
except ImportError:
    is_airbot_available = lambda: False  # type: ignore

try:
    from rollio.robot.airbot.play import AIRBOTPlay
except ImportError:
    AIRBOTPlay = None  # type: ignore

try:
    from rollio.robot.airbot.eef import AIRBOTE2B, AIRBOTG2
except ImportError:
    AIRBOTE2B = None  # type: ignore
    AIRBOTG2 = None  # type: ignore

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


def robot_class_for_type(robot_type: str) -> type[RobotArm] | None:
    """Resolve a registered built-in robot class by type name."""
    key = str(robot_type).strip()
    candidates = (PseudoRobotArm, AIRBOTPlay, AIRBOTE2B, AIRBOTG2)
    for cls in candidates:
        if cls is not None and getattr(cls, "ROBOT_TYPE", None) == key:
            return cls
    return None

__all__ = [
    # Enums
    "ControlMode",
    "FeedbackCapability",
    # State dataclasses
    "FrameState",
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
    "AIRBOTE2B",
    "AIRBOTG2",
    "PinocchioKinematicsModel",
    # Availability checks
    "is_airbot_available",
    "is_pinocchio_available",
    "is_python_can_available",
    # Scanner
    "DetectedRobot",
    "scan_robots",
    "scan_can_interfaces",
    "robot_class_for_type",
]
