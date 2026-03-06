"""Tele-operation mapping strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rollio.robot import Pose, RobotArm

MapperMode = Literal["auto", "joint_direct", "pose_fk_ik"]


@dataclass
class TeleopCommand:
    """A joint-space command generated from a tele-operation mapping."""

    mode: Literal["joint_direct", "pose_fk_ik"]
    position_target: np.ndarray
    velocity_target: np.ndarray
    leader_pose: Pose | None = None


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
        leader_state = leader.read_joint_state()
        if leader_state.position is None:
            raise ValueError("Leader does not expose joint positions")
        if follower.n_dof != len(leader_state.position):
            raise ValueError("Leader/follower DOF mismatch for direct mapping")

        velocity = leader_state.velocity
        if velocity is None:
            velocity = np.zeros_like(leader_state.position)

        return TeleopCommand(
            mode="joint_direct",
            position_target=np.asarray(leader_state.position, dtype=np.float64),
            velocity_target=np.asarray(velocity, dtype=np.float64),
            leader_pose=None,
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
        leader_joint_state = leader.read_joint_state()
        leader_ee = leader.read_end_effector_state()
        if leader_ee.pose is None:
            raise ValueError("Leader does not expose an end-effector pose")

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

        q_target, success = follower.kinematics.inverse_kinematics(
            leader_ee.pose,
            q_init=np.asarray(previous_target, dtype=np.float64),
        )
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
            leader_pose=leader_ee.pose,
        )


class AutoMapper(TeleopMapper):
    """Resolves to direct mapping when possible, otherwise FK/IK."""

    mode_name: MapperMode = "auto"

    def __init__(self) -> None:
        self._direct = JointSpaceDirectMapper()
        self._pose = PoseSpaceFkIkMapper()

    def resolve(self, leader: RobotArm, follower: RobotArm) -> TeleopMapper:
        if (
            leader.has_position_feedback
            and leader.info.robot_type == follower.info.robot_type
            and leader.n_dof == follower.n_dof
        ):
            return self._direct
        return self._pose

    def map_command(
        self,
        leader: RobotArm,
        follower: RobotArm,
        previous_target: np.ndarray | None = None,
    ) -> TeleopCommand:
        mapper = self.resolve(leader, follower)
        return mapper.map_command(leader, follower, previous_target)


def build_mapper(mode: MapperMode) -> TeleopMapper:
    """Create a mapper instance from configuration."""

    if mode == "joint_direct":
        return JointSpaceDirectMapper()
    if mode == "pose_fk_ik":
        return PoseSpaceFkIkMapper()
    return AutoMapper()
