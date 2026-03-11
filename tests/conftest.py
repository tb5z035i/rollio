"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def pseudo_robot():
    """Create a pseudo robot arm for testing."""
    from rollio.robot import PseudoRobotArm

    robot = PseudoRobotArm(name="test_robot", noise_level=0.0)
    robot.open()
    yield robot
    robot.close()


@pytest.fixture
def enabled_robot(pseudo_robot):
    """Create an enabled pseudo robot."""
    pseudo_robot.enable()
    return pseudo_robot


@pytest.fixture
def kinematics_model():
    """Create a pseudo kinematics model for testing."""
    from rollio.robot import PseudoKinematicsModel

    return PseudoKinematicsModel(n_dof=6)
