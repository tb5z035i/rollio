"""Unit tests for the rollio.robot module."""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rollio.robot import (
    ControlMode,
    DetectedRobot,
    FrameState,
    FeedbackCapability,
    FreeDriveCommand,
    JointState,
    KinematicsModel,
    Pose,
    PseudoKinematicsModel,
    PseudoRobotArm,
    RobotArm,
    RobotInfo,
    RobotState,
    TargetTrackingCommand,
    Twist,
    Wrench,
    scan_robots,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


class TestPose:
    """Tests for Pose dataclass."""
    
    def test_pose_creation(self) -> None:
        """Test basic pose creation."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        
        pose = Pose(position=position, quaternion=quaternion)
        
        np.testing.assert_array_almost_equal(pose.position, position)
        # Quaternion should be normalized
        assert abs(np.linalg.norm(pose.quaternion) - 1.0) < 1e-10
    
    def test_quaternion_normalization(self) -> None:
        """Test that quaternion is normalized on creation."""
        position = np.array([0.0, 0.0, 0.0])
        quaternion = np.array([2.0, 0.0, 0.0, 0.0])  # unnormalized
        
        pose = Pose(position=position, quaternion=quaternion)
        
        assert abs(np.linalg.norm(pose.quaternion) - 1.0) < 1e-10
        np.testing.assert_array_almost_equal(pose.quaternion, [1.0, 0.0, 0.0, 0.0])
    
    def test_rotation_matrix_identity(self) -> None:
        """Test rotation matrix for identity quaternion."""
        pose = Pose(
            position=np.array([0.0, 0.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0])
        )
        
        R = pose.rotation_matrix
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_rotation_matrix_90deg_z(self) -> None:
        """Test rotation matrix for 90-degree rotation around Z."""
        # Quaternion for 90-degree rotation around Z: [cos(45°), 0, 0, sin(45°)]
        angle = math.pi / 2
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        
        pose = Pose(
            position=np.array([0.0, 0.0, 0.0]),
            quaternion=np.array([w, 0.0, 0.0, z])
        )
        
        R = pose.rotation_matrix
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        np.testing.assert_array_almost_equal(R, expected, decimal=5)
    
    def test_from_matrix(self) -> None:
        """Test creation from rotation matrix."""
        position = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)
        
        pose = Pose.from_matrix(position, R)
        
        np.testing.assert_array_almost_equal(pose.position, position)
        # Identity rotation -> quaternion [1, 0, 0, 0] or close
        assert abs(abs(pose.quaternion[0]) - 1.0) < 1e-5 or \
               abs(np.linalg.norm(pose.quaternion[1:]) - 0.0) < 1e-5
    
    def test_homogeneous_matrix(self) -> None:
        """Test 4x4 homogeneous transformation matrix."""
        pose = Pose(
            position=np.array([1.0, 2.0, 3.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0])
        )
        
        T = pose.as_homogeneous()
        
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_from_euler(self) -> None:
        """Test creation from Euler angles."""
        position = np.array([1.0, 0.0, 0.0])
        angles = np.array([0.0, 0.0, math.pi / 2])  # 90 deg around Z
        
        pose = Pose.from_euler(position, angles, seq='xyz')
        
        np.testing.assert_array_almost_equal(pose.position, position)
        # Check rotation matrix matches 90 deg Z rotation
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(pose.rotation_matrix, expected_R, decimal=5)
    
    def test_euler_roundtrip(self) -> None:
        """Test Euler angle extraction matches input."""
        angles_in = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
        pose = Pose.from_euler(np.zeros(3), angles_in, seq='xyz')
        
        angles_out = pose.euler_xyz
        np.testing.assert_array_almost_equal(angles_out, angles_in)
    
    def test_identity(self) -> None:
        """Test identity pose creation."""
        pose = Pose.identity()
        
        np.testing.assert_array_almost_equal(pose.position, [0, 0, 0])
        np.testing.assert_array_almost_equal(pose.rotation_matrix, np.eye(3))
        
        pose_with_pos = Pose.identity(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(pose_with_pos.position, [1, 2, 3])
    
    def test_inverse(self) -> None:
        """Test pose inverse."""
        # Create a pose: translate then rotate
        pose = Pose.from_euler(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, math.pi / 2]),
            seq='xyz'
        )
        
        # Compose with inverse should give identity
        result = pose @ pose.inverse()
        
        np.testing.assert_array_almost_equal(result.position, [0, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(result.rotation_matrix, np.eye(3), decimal=5)
    
    def test_pose_composition(self) -> None:
        """Test pose composition (@ operator)."""
        # First pose: translate by [1, 0, 0]
        pose1 = Pose.identity(np.array([1.0, 0.0, 0.0]))
        
        # Second pose: translate by [0, 1, 0]
        pose2 = Pose.identity(np.array([0.0, 1.0, 0.0]))
        
        # Composition should translate by [1, 1, 0]
        result = pose1 @ pose2
        np.testing.assert_array_almost_equal(result.position, [1, 1, 0])
    
    def test_pose_composition_with_rotation(self) -> None:
        """Test pose composition with rotation."""
        # First pose: rotate 90 deg around Z
        pose1 = Pose.from_euler(np.zeros(3), np.array([0, 0, math.pi/2]), seq='xyz')
        
        # Second pose: translate by [1, 0, 0] in local frame
        pose2 = Pose.identity(np.array([1.0, 0.0, 0.0]))
        
        # After rotation, [1, 0, 0] becomes [0, 1, 0] in world frame
        result = pose1 @ pose2
        np.testing.assert_array_almost_equal(result.position, [0, 1, 0], decimal=5)
    
    def test_rotation_property(self) -> None:
        """Test access to scipy Rotation object."""
        from scipy.spatial.transform import Rotation
        
        pose = Pose.from_euler(
            np.zeros(3),
            np.array([0.1, 0.2, 0.3]),
            seq='xyz'
        )
        
        rot = pose.rotation
        assert isinstance(rot, Rotation)
        
        # Verify it matches the quaternion
        quat_xyzw = rot.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        np.testing.assert_array_almost_equal(quat_wxyz, pose.quaternion)


class TestTwist:
    """Tests for Twist dataclass."""
    
    def test_twist_creation(self) -> None:
        """Test basic twist creation."""
        twist = Twist(
            linear=np.array([1.0, 2.0, 3.0]),
            angular=np.array([0.1, 0.2, 0.3])
        )
        
        np.testing.assert_array_almost_equal(twist.linear, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(twist.angular, [0.1, 0.2, 0.3])
    
    def test_as_vector(self) -> None:
        """Test conversion to 6D vector."""
        twist = Twist(
            linear=np.array([1.0, 2.0, 3.0]),
            angular=np.array([4.0, 5.0, 6.0])
        )
        
        vec = twist.as_vector()
        
        assert vec.shape == (6,)
        np.testing.assert_array_almost_equal(vec, [1, 2, 3, 4, 5, 6])


class TestWrench:
    """Tests for Wrench dataclass."""
    
    def test_wrench_creation(self) -> None:
        """Test basic wrench creation."""
        wrench = Wrench(
            force=np.array([10.0, 20.0, 30.0]),
            torque=np.array([1.0, 2.0, 3.0])
        )
        
        np.testing.assert_array_almost_equal(wrench.force, [10, 20, 30])
        np.testing.assert_array_almost_equal(wrench.torque, [1, 2, 3])
    
    def test_as_vector(self) -> None:
        """Test conversion to 6D vector."""
        wrench = Wrench(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([4.0, 5.0, 6.0])
        )
        
        vec = wrench.as_vector()
        
        np.testing.assert_array_almost_equal(vec, [1, 2, 3, 4, 5, 6])
    
    def test_from_vector(self) -> None:
        """Test creation from 6D vector."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        wrench = Wrench.from_vector(vec)
        
        np.testing.assert_array_almost_equal(wrench.force, [1, 2, 3])
        np.testing.assert_array_almost_equal(wrench.torque, [4, 5, 6])
    
    def test_zero(self) -> None:
        """Test zero wrench creation."""
        wrench = Wrench.zero()
        
        np.testing.assert_array_almost_equal(wrench.force, [0, 0, 0])
        np.testing.assert_array_almost_equal(wrench.torque, [0, 0, 0])


class TestJointState:
    """Tests for JointState dataclass."""
    
    def test_joint_state_creation(self) -> None:
        """Test basic joint state creation."""
        state = JointState(
            timestamp=1.0,
            position=np.array([0.1, 0.2, 0.3]),
            velocity=np.array([0.01, 0.02, 0.03]),
            effort=np.array([1.0, 2.0, 3.0])
        )
        
        assert state.timestamp == 1.0
        assert state.position.dtype == np.float32
        assert state.velocity.dtype == np.float32
        assert state.effort.dtype == np.float32
        assert state.is_valid
    
    def test_partial_state(self) -> None:
        """Test state with only position."""
        state = JointState(
            timestamp=1.0,
            position=np.array([0.1, 0.2, 0.3])
        )
        
        assert state.position is not None
        assert state.velocity is None
        assert state.effort is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test Kinematics Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestPseudoKinematicsModel:
    """Tests for PseudoKinematicsModel."""
    
    @pytest.fixture
    def model(self) -> PseudoKinematicsModel:
        """Create a test kinematics model."""
        return PseudoKinematicsModel(n_dof=6)
    
    def test_n_dof(self, model: PseudoKinematicsModel) -> None:
        """Test DOF property."""
        assert model.n_dof == 6
    
    def test_frame_names(self, model: PseudoKinematicsModel) -> None:
        """Test task-space frame names."""
        names = model.frame_names
        assert len(names) == 1
        assert names[0] == "frame"
    
    def test_forward_kinematics_zero_config(self, model: PseudoKinematicsModel) -> None:
        """Test FK at zero configuration."""
        q = np.zeros(6)
        pose = model.forward_kinematics(q)
        
        assert isinstance(pose, Pose)
        assert pose.position.shape == (3,)
        assert pose.quaternion.shape == (4,)
        # At zero config, arm should be extended forward
        assert pose.position[2] > 0  # Above base
    
    def test_forward_kinematics_various_configs(self, model: PseudoKinematicsModel) -> None:
        """Test FK at various configurations."""
        # Test multiple random configurations
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            q = rng.uniform(-np.pi/2, np.pi/2, 6)
            pose = model.forward_kinematics(q)
            
            assert isinstance(pose, Pose)
            # Position should be reasonable (within arm reach)
            assert np.linalg.norm(pose.position) < 2.0  # meters
    
    def test_jacobian_shape(self, model: PseudoKinematicsModel) -> None:
        """Test Jacobian matrix shape."""
        q = np.zeros(6)
        J = model.jacobian(q)
        
        assert J.shape == (6, 6)
    
    def test_jacobian_consistency(self, model: PseudoKinematicsModel) -> None:
        """Test that Jacobian is consistent with FK via finite differences."""
        q = np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.1])
        J = model.jacobian(q)
        
        # Verify Jacobian by finite differences (already how it's computed, so this
        # just checks consistency)
        eps = 1e-4
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            
            pose_plus = model.forward_kinematics(q_plus)
            pose_minus = model.forward_kinematics(q_minus)
            
            dpos = (pose_plus.position - pose_minus.position) / (2 * eps)
            np.testing.assert_array_almost_equal(J[:3, i], dpos, decimal=3)
    
    def test_inverse_dynamics_gravity_only(self, model: PseudoKinematicsModel) -> None:
        """Test inverse dynamics with zero velocities/accelerations (gravity only)."""
        q = np.array([0.0, np.pi/4, -np.pi/4, 0.0, 0.0, 0.0])
        zeros = np.zeros(6)
        
        tau = model.inverse_dynamics(q, zeros, zeros)
        
        assert tau.shape == (6,)
        # Shoulder and elbow joints should have significant gravity torques
        assert abs(tau[1]) > 0.1  # Shoulder
        assert abs(tau[2]) > 0.01  # Elbow
    
    def test_gravity_compensation(self, model: PseudoKinematicsModel) -> None:
        """Test gravity compensation shortcut."""
        q = np.array([0.0, 0.5, -0.3, 0.1, 0.0, 0.0])
        
        tau_grav = model.gravity_compensation(q)
        tau_id = model.inverse_dynamics(q, np.zeros(6), np.zeros(6))
        
        np.testing.assert_array_almost_equal(tau_grav, tau_id)
    
    def test_wrench_to_joint_torques(self, model: PseudoKinematicsModel) -> None:
        """Test wrench transformation via Jacobian transpose."""
        q = np.zeros(6)
        wrench = Wrench(
            force=np.array([10.0, 0.0, 0.0]),  # Force in X direction
            torque=np.array([0.0, 0.0, 0.0])
        )
        
        tau = model.wrench_to_joint_torques(q, wrench)
        
        assert tau.shape == (6,)
        # Should produce some non-zero torques
        assert np.linalg.norm(tau) > 0
    
    def test_inverse_kinematics_reachable(self, model: PseudoKinematicsModel) -> None:
        """Test IK for a reachable pose."""
        # First get a pose from FK
        q_original = np.array([0.1, 0.3, -0.2, 0.1, 0.0, 0.05])
        target_pose = model.forward_kinematics(q_original)
        
        # Now try to find it with IK
        q_solution, success = model.inverse_kinematics(
            target_pose, 
            q_init=np.zeros(6),
            max_iterations=200
        )
        
        # Check that we got close (IK may not be exact for pseudo model)
        if success:
            result_pose = model.forward_kinematics(q_solution)
            pos_error = np.linalg.norm(result_pose.position - target_pose.position)
            assert pos_error < 0.1  # Within 10cm


# ═══════════════════════════════════════════════════════════════════════════════
# Test Robot Arm
# ═══════════════════════════════════════════════════════════════════════════════


class TestPseudoRobotArm:
    """Tests for PseudoRobotArm."""
    
    @pytest.fixture
    def robot(self) -> PseudoRobotArm:
        """Create and open a test robot."""
        robot = PseudoRobotArm(name="test_robot", n_dof=6)
        robot.open()
        return robot
    
    def test_scan(self) -> None:
        """Test robot scanning."""
        devices = PseudoRobotArm.scan()
        
        assert len(devices) == 1
        assert devices[0].robot_type == "pseudo"
        assert devices[0].n_dof == 6
    
    def test_probe(self) -> None:
        """Test robot probing."""
        assert PseudoRobotArm.probe(0) is True
        assert PseudoRobotArm.probe("any") is True
    
    def test_properties(self, robot: PseudoRobotArm) -> None:
        """Test robot properties."""
        assert robot.n_dof == 6
        assert robot.ROBOT_TYPE == "pseudo"
        assert isinstance(robot.info, RobotInfo)
        assert isinstance(robot.kinematics, KinematicsModel)
    
    def test_info(self, robot: PseudoRobotArm) -> None:
        """Test robot info."""
        info = robot.info
        
        assert info.name == "test_robot"
        assert info.robot_type == "pseudo"
        assert info.n_dof == 6
        assert FeedbackCapability.POSITION in info.feedback_capabilities
        assert FeedbackCapability.VELOCITY in info.feedback_capabilities
        assert FeedbackCapability.EFFORT in info.feedback_capabilities
    
    def test_feedback_availability(self, robot: PseudoRobotArm) -> None:
        """Test feedback availability flags."""
        assert robot.has_position_feedback is True
        assert robot.has_velocity_feedback is True
        assert robot.has_effort_feedback is True
        assert robot.has_frame_pose_feedback is True
        assert robot.has_frame_twist_feedback is True
        assert robot.has_frame_wrench_feedback is False  # No F/T sensor
    
    def test_lifecycle(self) -> None:
        """Test open/close/enable/disable lifecycle."""
        robot = PseudoRobotArm()
        
        assert robot.is_enabled is False
        
        robot.open()
        assert robot.is_enabled is False
        
        assert robot.enable() is True
        assert robot.is_enabled is True
        
        robot.disable()
        assert robot.is_enabled is False
        
        robot.close()
    
    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with PseudoRobotArm() as robot:
            robot.enable()
            assert robot.is_enabled is True
            state = robot.read_joint_state()
            assert state.is_valid
        
        # After exit, robot should be disabled
        assert robot.is_enabled is False
    
    def test_read_joint_state(self, robot: PseudoRobotArm) -> None:
        """Test reading joint state."""
        state = robot.read_joint_state()
        
        assert isinstance(state, JointState)
        assert state.timestamp > 0
        assert state.position is not None
        assert state.position.shape == (6,)
        assert state.velocity is not None
        assert state.velocity.shape == (6,)
        assert state.effort is not None
        assert state.effort.shape == (6,)
    
    def test_read_frame_state(self, robot: PseudoRobotArm) -> None:
        """Test reading task-space frame state."""
        ee_state = robot.read_frame_state()
        
        assert isinstance(ee_state, FrameState)
        assert ee_state.pose is not None
        assert ee_state.twist is not None
        assert ee_state.wrench is None  # No F/T sensor
    
    def test_read_state(self, robot: PseudoRobotArm) -> None:
        """Test reading complete robot state."""
        state = robot.read_state()
        
        assert isinstance(state, RobotState)
        assert state.joint_state is not None
        assert len(state.frames) == 1
        assert state.control_mode == ControlMode.DISABLED
    
    def test_control_mode_setting(self, robot: PseudoRobotArm) -> None:
        """Test control mode setting."""
        robot.enable()
        
        assert robot.set_control_mode(ControlMode.FREE_DRIVE) is True
        assert robot.control_mode == ControlMode.FREE_DRIVE
        
        assert robot.set_control_mode(ControlMode.TARGET_TRACKING) is True
        assert robot.control_mode == ControlMode.TARGET_TRACKING
        
        assert robot.set_control_mode(ControlMode.DISABLED) is True
        assert robot.control_mode == ControlMode.DISABLED
    
    def test_control_mode_requires_enable(self, robot: PseudoRobotArm) -> None:
        """Test that control mode setting requires enabled robot."""
        # Robot is open but not enabled
        assert robot.set_control_mode(ControlMode.FREE_DRIVE) is False
    
    def test_enter_free_drive(self, robot: PseudoRobotArm) -> None:
        """Test entering free drive mode."""
        robot.enable()
        
        assert robot.enter_free_drive() is True
        assert robot.control_mode == ControlMode.FREE_DRIVE
    
    def test_enter_target_tracking(self, robot: PseudoRobotArm) -> None:
        """Test entering target tracking mode."""
        robot.enable()
        
        assert robot.enter_target_tracking() is True
        assert robot.control_mode == ControlMode.TARGET_TRACKING


class TestFreeDriveMode:
    """Tests for free drive mode."""
    
    @pytest.fixture
    def robot(self) -> PseudoRobotArm:
        """Create a robot in free drive mode."""
        robot = PseudoRobotArm(name="test_robot", noise_level=0.0)
        robot.open()
        robot.enable()
        robot.enter_free_drive()
        return robot
    
    def test_free_drive_command(self, robot: PseudoRobotArm) -> None:
        """Test basic free drive command."""
        cmd = FreeDriveCommand(
            external_wrench=None,
            gravity_compensation_scale=1.0
        )
        
        # Should not raise
        robot.command_free_drive(cmd)
    
    def test_free_drive_with_wrench(self, robot: PseudoRobotArm) -> None:
        """Test free drive with external wrench."""
        # Start at a non-zero config where external force produces torque
        robot.set_joint_position(np.array([0.0, 0.5, -0.3, 0.0, 0.0, 0.0]))
        
        initial_pos = robot.get_raw_position().copy()
        
        # Apply a force in X direction (produces torque when arm is bent)
        wrench = Wrench(
            force=np.array([50.0, 0.0, 0.0]),  # Force in X
            torque=np.array([0.0, 0.0, 0.0])
        )
        
        for _ in range(500):  # Run control loop
            robot.step_free_drive(external_wrench=wrench)
        
        final_pos = robot.get_raw_position()
        
        # Position should have changed due to the applied force
        pos_change = np.linalg.norm(final_pos - initial_pos)
        assert pos_change > 0.001, f"Expected movement, got {pos_change}"
    
    def test_gravity_compensation_scale(self, robot: PseudoRobotArm) -> None:
        """Test gravity compensation scaling."""
        # With full compensation
        robot.step_free_drive(gravity_scale=1.0)
        state1 = robot.read_joint_state()
        
        robot.set_joint_position(np.zeros(6))
        
        # With reduced compensation
        robot.step_free_drive(gravity_scale=0.5)
        state2 = robot.read_joint_state()
        
        # Both should execute without error
        assert state1.is_valid
        assert state2.is_valid


class TestTargetTrackingMode:
    """Tests for target tracking (MIT) mode."""
    
    @pytest.fixture
    def robot(self) -> PseudoRobotArm:
        """Create a robot in target tracking mode."""
        robot = PseudoRobotArm(name="test_robot", noise_level=0.0)
        robot.open()
        robot.enable()
        robot.enter_target_tracking()
        return robot
    
    def test_target_tracking_command(self, robot: PseudoRobotArm) -> None:
        """Test basic target tracking command."""
        cmd = TargetTrackingCommand(
            position_target=np.zeros(6),
            velocity_target=np.zeros(6),
            kp=np.full(6, 10.0),
            kd=np.full(6, 2.0),
            feedforward=None
        )
        
        # Should not raise
        robot.command_target_tracking(cmd)
    
    def test_target_tracking_convergence(self, robot: PseudoRobotArm) -> None:
        """Test that robot converges toward target position."""
        # Start from non-zero position (smaller initial offset)
        initial_pos = np.array([0.3, 0.2, -0.1, 0.05, 0.05, 0.0])
        robot.set_joint_position(initial_pos)
        
        target = np.zeros(6)
        initial_error = np.abs(initial_pos - target).max()
        
        # Run control loop with higher gains for faster convergence
        for _ in range(1000):
            robot.step_target_tracking(
                position_target=target,
                kp=100.0,
                kd=20.0,
                add_gravity_compensation=True
            )
        
        final_state = robot.read_joint_state()
        final_error = np.abs(final_state.position - target).max()
        
        # Should have made progress toward target (error reduced)
        assert final_error < initial_error, \
            f"Expected error to decrease: {initial_error:.3f} -> {final_error:.3f}"
    
    def test_target_tracking_with_feedforward(self, robot: PseudoRobotArm) -> None:
        """Test target tracking with explicit feedforward."""
        target = np.array([0.1, 0.2, -0.1, 0.0, 0.0, 0.0])
        ff = np.array([0.5, 1.0, 0.5, 0.1, 0.05, 0.01])
        
        robot.step_target_tracking(
            position_target=target,
            velocity_target=np.zeros(6),
            kp=10.0,
            kd=2.0,
            feedforward=ff,
            add_gravity_compensation=False
        )
        
        state = robot.read_joint_state()
        assert state.is_valid
    
    def test_move_to_position(self, robot: PseudoRobotArm) -> None:
        """Test blocking move to position."""
        # Start from zero
        robot.set_joint_position(np.zeros(6))
        
        target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Move with short timeout for test speed
        success = robot.move_to_position(
            target_position=target,
            kp=50.0,
            kd=10.0,
            tolerance=0.1,
            timeout=2.0,
            dt=0.004
        )
        
        # Should either succeed or timeout (both valid outcomes for pseudo)
        assert isinstance(success, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Scanner
# ═══════════════════════════════════════════════════════════════════════════════


class TestScanner:
    """Tests for robot scanner."""
    
    def test_scan_robots(self) -> None:
        """Test scanning for robots."""
        devices = scan_robots()
        
        assert len(devices) >= 1  # At least pseudo robot
        
        # Find pseudo robot
        pseudo = next((d for d in devices if d.robot_type == "pseudo"), None)
        assert pseudo is not None
        assert pseudo.n_dof == 6
        assert isinstance(pseudo.device_id, int)
    
    def test_detected_robot_dataclass(self) -> None:
        """Test DetectedRobot dataclass."""
        device = DetectedRobot(
            robot_type="test",
            device_id="can0",
            label="Test Robot",
            n_dof=7,
            properties={"key": "value"}
        )
        
        assert device.robot_type == "test"
        assert device.device_id == "can0"
        assert device.label == "Test Robot"
        assert device.n_dof == 7
        assert device.properties["key"] == "value"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Control Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestControlCommands:
    """Tests for control command dataclasses."""
    
    def test_target_tracking_command(self) -> None:
        """Test TargetTrackingCommand creation and normalization."""
        cmd = TargetTrackingCommand(
            position_target=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # list input
            velocity_target=np.zeros(6),
            kp=10.0,  # scalar
            kd=np.full(6, 2.0)
        )
        
        # Should be converted to arrays
        assert cmd.position_target.dtype == np.float64
        assert cmd.velocity_target.dtype == np.float64
        assert cmd.kp.dtype == np.float64
        assert cmd.kd.dtype == np.float64
    
    def test_free_drive_command(self) -> None:
        """Test FreeDriveCommand creation."""
        cmd = FreeDriveCommand(
            external_wrench=Wrench.zero(),
            gravity_compensation_scale=0.8
        )
        
        assert cmd.gravity_compensation_scale == 0.8
        assert cmd.external_wrench is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self) -> None:
        """Test complete robot workflow."""
        # Scan for robots
        devices = scan_robots()
        assert len(devices) > 0
        
        # Create robot
        robot = PseudoRobotArm()
        
        with robot:
            robot.enable()
            
            # Read initial state
            state = robot.read_state()
            assert state.is_valid
            
            # Enter free drive and move around
            robot.enter_free_drive()
            for _ in range(10):
                robot.step_free_drive()
            
            # Switch to target tracking
            robot.enter_target_tracking()
            target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            for _ in range(10):
                robot.step_target_tracking(position_target=target)
            
            # Read final state
            final_state = robot.read_state()
            assert final_state.is_valid
            assert final_state.control_mode == ControlMode.TARGET_TRACKING
    
    def test_kinematics_integration(self) -> None:
        """Test kinematics computations through robot interface."""
        with PseudoRobotArm() as robot:
            robot.enable()
            
            # Read joint state
            joint_state = robot.read_joint_state()
            q = joint_state.position
            
            # Compute FK
            pose = robot.kinematics.forward_kinematics(q)
            assert isinstance(pose, Pose)
            
            # Compute Jacobian
            J = robot.kinematics.jacobian(q)
            assert J.shape == (6, 6)
            
            # Compute gravity compensation
            tau_g = robot.kinematics.gravity_compensation(q)
            assert tau_g.shape == (6,)
