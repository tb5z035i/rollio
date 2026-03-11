"""Shared policy helpers for tele-operation pair selection."""

from __future__ import annotations

from collections.abc import Iterable


def supports_joint_direct_mapping(
    *,
    leader_type: str,
    leader_n_dof: int,
    leader_allowlist: Iterable[str],
    follower_type: str,
    follower_n_dof: int,
    follower_allowlist: Iterable[str],
) -> bool:
    """Return True when a pair should prefer direct joint mapping."""

    normalized_leader_allowlist = {
        str(item).strip() for item in leader_allowlist if str(item).strip()
    }
    normalized_follower_allowlist = {
        str(item).strip() for item in follower_allowlist if str(item).strip()
    }
    return (
        int(leader_n_dof) == int(follower_n_dof)
        and str(follower_type).strip() in normalized_leader_allowlist
        and str(leader_type).strip() in normalized_follower_allowlist
    )


def supports_pose_fkik_mapping(*, leader_n_dof: int, follower_n_dof: int) -> bool:
    """Return True when a pair should prefer pose FK/IK mapping."""

    return int(leader_n_dof) > 1 and int(follower_n_dof) > 1


def default_mapper_name(
    *,
    leader_type: str,
    leader_n_dof: int,
    leader_allowlist: Iterable[str],
    follower_type: str,
    follower_n_dof: int,
    follower_allowlist: Iterable[str],
) -> str:
    """Choose the default mapper name for one leader/follower pair."""

    if supports_joint_direct_mapping(
        leader_type=leader_type,
        leader_n_dof=leader_n_dof,
        leader_allowlist=leader_allowlist,
        follower_type=follower_type,
        follower_n_dof=follower_n_dof,
        follower_allowlist=follower_allowlist,
    ):
        return "joint_direct"
    if supports_pose_fkik_mapping(
        leader_n_dof=leader_n_dof,
        follower_n_dof=follower_n_dof,
    ):
        return "pose_fk_ik"
    return "pose_fk_ik"
