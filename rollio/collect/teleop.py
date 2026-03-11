"""Tele-operation mapping strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rollio.robot import JointState, Pose, RobotArm

MapperMode = Literal["auto", "joint_direct", "pose_fk_ik"]
ResolvedMapperMode = Literal["joint_direct", "pose_fk_ik", "noop"]


def _direct_map_allowlist(robot: RobotArm) -> set[str]:
    return {
        str(item).strip() for item in robot.direct_map_allowlist if str(item).strip()
    }


def supports_joint_direct_runtime(leader: RobotArm, follower: RobotArm) -> bool:
    """Return True when runtime metadata allows direct mapping."""
    return (
        leader.has_position_feedback
        and follower.has_position_feedback
        and leader.n_dof == follower.n_dof
        and follower.info.robot_type in _direct_map_allowlist(leader)
        and leader.info.robot_type in _direct_map_allowlist(follower)
    )


def supports_pose_fkik_runtime(leader: RobotArm, follower: RobotArm) -> bool:
    """Return True when a runtime pair should use pose FK/IK."""
    return leader.has_frame_pose_feedback and leader.n_dof > 1 and follower.n_dof > 1


@dataclass
class TeleopCommand:
    """A joint-space command generated from a tele-operation mapping."""

    mode: ResolvedMapperMode
    position_target: np.ndarray | None
    velocity_target: np.ndarray | None
    leader_pose: Pose | None = None
    leader_joint_state: JointState | None = None

    @classmethod
    def noop(cls) -> "TeleopCommand":
        return cls(
            mode="noop",
            position_target=None,
            velocity_target=None,
            leader_pose=None,
            leader_joint_state=None,
        )


class TeleopMapper:
    """Base class for tele-operation mapping strategies."""

    mode_name: MapperMode = "auto"

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        raise NotImplementedError


class JointSpaceDirectMapper(TeleopMapper):
    """Maps leader joints directly to follower joints."""

    mode_name: MapperMode = "joint_direct"

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        del previous_target
        if not supports_joint_direct_runtime(leader, follower):
            return TeleopCommand.noop()
        leader_state = leader.read_joint_state()
        if leader_state.position is None:
            return TeleopCommand.noop()
        if follower.n_dof != len(leader_state.position):
            return TeleopCommand.noop()

        velocity = leader_state.velocity
        if velocity is None:
            velocity = np.zeros_like(leader_state.position)

        return TeleopCommand(
            mode="joint_direct",
            position_target=np.asarray(leader_state.position, dtype=np.float64),
            velocity_target=np.asarray(velocity, dtype=np.float64),
            leader_pose=None,
            leader_joint_state=leader_state,
        )


class PoseSpaceFkIkMapper(TeleopMapper):
    """Maps leader pose to follower joints via FK/IK."""

    mode_name: MapperMode = "pose_fk_ik"

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        if not supports_pose_fkik_runtime(leader, follower):
            return TeleopCommand.noop()
        leader_joint_state = leader.read_joint_state()
        leader_frame = leader.read_frame_state()
        if leader_frame.pose is None:
            return TeleopCommand.noop()

        if previous_target is None:
            if (
                leader_joint_state.position is not None
                and len(leader_joint_state.position) == follower.n_dof
            ):
                previous_target = np.asarray(
                    leader_joint_state.position,
                    dtype=np.float64,
                )
            else:
                follower_state = follower.read_joint_state()
                previous_target = (
                    np.asarray(follower_state.position, dtype=np.float64)
                    if follower_state.position is not None
                    else np.zeros(follower.n_dof, dtype=np.float64)
                )

        try:
            q_target, success = follower.kinematics.inverse_kinematics(
                leader_frame.pose,
                q_init=np.asarray(previous_target, dtype=np.float64),
            )
        except Exception:
            return TeleopCommand.noop()
        if not success or q_target is None:
            if (
                leader_joint_state.position is not None
                and len(leader_joint_state.position) == follower.n_dof
            ):
                q_target = np.asarray(leader_joint_state.position, dtype=np.float64)
            else:
                q_target = np.asarray(previous_target, dtype=np.float64)

        return TeleopCommand(
            mode="pose_fk_ik",
            position_target=np.asarray(q_target, dtype=np.float64),
            velocity_target=np.zeros(follower.n_dof, dtype=np.float64),
            leader_pose=leader_frame.pose,
            leader_joint_state=leader_joint_state,
        )


class AutoMapper(TeleopMapper):
    """Resolves to direct mapping when possible, otherwise FK/IK."""

    mode_name: MapperMode = "auto"

    def __init__(self) -> None:
        self._direct = JointSpaceDirectMapper()
        self._pose = PoseSpaceFkIkMapper()
        self._noop = NoOpMapper()

    def resolve(self, leader: RobotArm, follower: RobotArm) -> TeleopMapper:
        if supports_joint_direct_runtime(leader, follower):
            return self._direct
        if supports_pose_fkik_runtime(leader, follower):
            return self._pose
        return self._noop

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        mapper = self.resolve(leader, follower)
        return mapper.map_command(leader, follower, previous_target)


class NoOpMapper(TeleopMapper):
    """Mapper used for pairs that should not issue follower commands."""

    mode_name: MapperMode = "auto"

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        del leader, follower, previous_target
        return TeleopCommand.noop()


def build_mapper(mode: MapperMode) -> TeleopMapper:
    """Create a mapper instance from configuration."""

    if mode == "joint_direct":
        return JointSpaceDirectMapper()
    if mode == "pose_fk_ik":
        return PoseSpaceFkIkMapper()
    return AutoMapper()
