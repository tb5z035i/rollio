"""Worker bootstrap helpers used by runtime-service tests."""

from __future__ import annotations

from rollio.collect import register_camera_factory, register_robot_factory
from rollio.robot import PseudoRobotArm
from rollio.sensors import PseudoCamera


def register() -> None:
    """Register custom factories so a spawned worker can import them."""

    register_camera_factory(
        "worker_test_camera",
        lambda cfg: PseudoCamera(
            name=f"{cfg.name}-bootstrapped",
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        ),
        replace=True,
    )
    register_robot_factory(
        "worker_test_robot",
        lambda cfg: PseudoRobotArm(
            name=f"{cfg.name}-bootstrapped",
            n_dof=cfg.num_joints,
        ),
        replace=True,
    )
