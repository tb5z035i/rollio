"""Helpers for explicit tele-operation pair configuration."""
from __future__ import annotations

from collections import Counter

from rollio.config.schema import RobotConfig, TeleopPairConfig


def supports_joint_direct_mapping(leader: RobotConfig, follower: RobotConfig) -> bool:
    """Return True when a leader/follower pair can use direct joint mapping."""
    return (
        leader.type == follower.type
        and leader.num_joints == follower.num_joints
    )


def default_mapper_for_pair(leader: RobotConfig, follower: RobotConfig) -> str:
    """Choose the default mapper for one explicit tele-op pair."""
    if supports_joint_direct_mapping(leader, follower):
        return "joint_direct"
    return "pose_fk_ik"


def suggest_teleop_pairs(robots: list[RobotConfig]) -> list[TeleopPairConfig]:
    """Build deterministic default tele-op pairs from robot roles."""
    leaders = [robot for robot in robots if robot.role == "leader"]
    followers = [robot for robot in robots if robot.role == "follower"]
    pairs: list[TeleopPairConfig] = []
    for idx, (leader, follower) in enumerate(zip(leaders, followers)):
        pairs.append(TeleopPairConfig(
            name=f"pair_{idx}",
            leader=leader.name,
            follower=follower.name,
            mapper=default_mapper_for_pair(leader, follower),
        ))
    return pairs


def validate_teleop_pairs(
    robots: list[RobotConfig],
    pairs: list[TeleopPairConfig],
) -> None:
    """Raise ValueError when tele-op pair definitions are inconsistent."""
    robots_by_name = {robot.name: robot for robot in robots}
    leaders = {robot.name for robot in robots if robot.role == "leader"}
    followers = {robot.name for robot in robots if robot.role == "follower"}

    if not leaders:
        raise ValueError("Tele-op mode requires at least one leader robot")
    if not followers:
        raise ValueError("Tele-op mode requires at least one follower robot")

    pair_names = [pair.name for pair in pairs]
    duplicate_pair_names = [name for name, count in Counter(pair_names).items() if count > 1]
    if duplicate_pair_names:
        raise ValueError(
            "Tele-op pair names must be unique: "
            + ", ".join(sorted(duplicate_pair_names))
        )

    duplicate_leaders = [
        name for name, count in Counter(pair.leader for pair in pairs).items() if count > 1
    ]
    if duplicate_leaders:
        raise ValueError(
            "Each leader can only be assigned once: "
            + ", ".join(sorted(duplicate_leaders))
        )

    duplicate_followers = [
        name for name, count in Counter(pair.follower for pair in pairs).items() if count > 1
    ]
    if duplicate_followers:
        raise ValueError(
            "Each follower can only be assigned once: "
            + ", ".join(sorted(duplicate_followers))
        )

    for pair in pairs:
        if pair.leader not in robots_by_name:
            raise ValueError(f"Unknown leader robot in pair '{pair.name}': {pair.leader}")
        if pair.follower not in robots_by_name:
            raise ValueError(f"Unknown follower robot in pair '{pair.name}': {pair.follower}")
        if pair.leader not in leaders:
            raise ValueError(f"Robot '{pair.leader}' in pair '{pair.name}' is not marked as a leader")
        if pair.follower not in followers:
            raise ValueError(f"Robot '{pair.follower}' in pair '{pair.name}' is not marked as a follower")
