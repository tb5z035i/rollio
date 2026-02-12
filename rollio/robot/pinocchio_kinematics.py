"""Pinocchio-based kinematics model for robot arms.

This module provides a kinematics model implementation using the Pinocchio
rigid body dynamics library. It supports loading URDF models and computing:
- Forward kinematics
- Inverse kinematics (numerical)
- Jacobian matrices
- Inverse dynamics (for gravity compensation)
"""
from __future__ import annotations

from pathlib import Path
from importlib import resources

import numpy as np

from rollio.robot.base import KinematicsModel, Pose, Wrench

# Lazy import pinocchio to avoid hard dependency
_pin = None
_PIN_AVAILABLE = None


def _import_pinocchio():
    """Lazy import pinocchio."""
    global _pin, _PIN_AVAILABLE
    if _PIN_AVAILABLE is None:
        try:
            import pinocchio as pin
            _pin = pin
            _PIN_AVAILABLE = True
        except ImportError:
            _PIN_AVAILABLE = False
    return _pin, _PIN_AVAILABLE


def is_pinocchio_available() -> bool:
    """Check if pinocchio is available."""
    _, available = _import_pinocchio()
    return available


def get_bundled_urdf(name: str) -> Path | None:
    """Get path to a bundled URDF file.
    
    Args:
        name: URDF name (e.g., "play_e2" for AIRBOT Play E2)
        
    Returns:
        Path to the URDF file, or None if not found
    """
    try:
        # Try importlib.resources (Python 3.9+)
        try:
            from importlib.resources import files
            urdf_dir = files("rollio.robot") / "urdf"
            urdf_path = urdf_dir / f"{name}.urdf"
            if urdf_path.is_file():
                return Path(str(urdf_path))
        except (ImportError, TypeError):
            pass
        
        # Fallback: check relative to this module
        module_dir = Path(__file__).parent
        urdf_path = module_dir / "urdf" / f"{name}.urdf"
        if urdf_path.exists():
            return urdf_path
        
        return None
    except Exception:
        return None


class PinocchioKinematicsModel(KinematicsModel):
    """Kinematics model using Pinocchio for URDF-based robots.
    
    This implementation uses Pinocchio for accurate kinematics and dynamics
    computations based on URDF robot descriptions.
    
    Supports locking joints (e.g., gripper joints) to reduce the model to
    only the arm joints for control.
    
    Args:
        urdf_path: Path to URDF file
        end_effector_frame: Name of the end-effector frame in the URDF
        mesh_dir: Optional directory containing mesh files (for visualization)
        arm_joints: List of joint names to include (others will be locked).
                   If None, all joints are included.
    """
    
    def __init__(
        self,
        urdf_path: str | Path,
        end_effector_frame: str | None = None,
        mesh_dir: str | Path | None = None,
        arm_joints: list[str] | None = None,
    ) -> None:
        pin, available = _import_pinocchio()
        if not available:
            raise ImportError(
                "Pinocchio is required for PinocchioKinematicsModel. "
                "Install with: pip install pin"
            )
        
        self._pin = pin
        self._urdf_path = Path(urdf_path)
        
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Load full URDF model
        if mesh_dir:
            full_model = pin.buildModelFromUrdf(
                str(self._urdf_path), 
                str(mesh_dir)
            )
        else:
            full_model = pin.buildModelFromUrdf(str(self._urdf_path))
        
        # Lock non-arm joints if arm_joints is specified
        if arm_joints is not None:
            locked_joint_ids = []
            for joint_id in range(1, full_model.njoints):  # Skip universe joint
                joint_name = full_model.names[joint_id]
                if joint_name not in arm_joints:
                    locked_joint_ids.append(joint_id)
            
            if locked_joint_ids:
                # Lock joints at neutral position
                q_neutral = pin.neutral(full_model)
                self._model = pin.buildReducedModel(
                    full_model, locked_joint_ids, q_neutral
                )
            else:
                self._model = full_model
        else:
            self._model = full_model
        
        self._data = self._model.createData()
        self._n_dof = self._model.nq
        
        # Find end-effector frame
        if end_effector_frame:
            if not self._model.existFrame(end_effector_frame):
                available_frames = [
                    self._model.frames[i].name 
                    for i in range(self._model.nframes)
                ]
                raise ValueError(
                    f"Frame '{end_effector_frame}' not found in URDF. "
                    f"Available frames: {available_frames}"
                )
            self._ee_frame_id = self._model.getFrameId(end_effector_frame)
            self._ee_frame_name = end_effector_frame
        else:
            # Try common end-effector frame names
            for frame_name in ["end_link", "link6", "eef_connect_base_link"]:
                if self._model.existFrame(frame_name):
                    self._ee_frame_id = self._model.getFrameId(frame_name)
                    self._ee_frame_name = frame_name
                    break
            else:
                # Use the last frame as end-effector
                self._ee_frame_id = self._model.nframes - 1
                self._ee_frame_name = self._model.frames[self._ee_frame_id].name
        
        self._end_effector_names = [self._ee_frame_name]
    
    @property
    def n_dof(self) -> int:
        return self._n_dof
    
    @property
    def end_effector_names(self) -> list[str]:
        return self._end_effector_names
    
    @property
    def model(self):
        """Access the underlying Pinocchio model."""
        return self._model
    
    @property
    def data(self):
        """Access the underlying Pinocchio data."""
        return self._data
    
    @property
    def ee_frame_id(self) -> int:
        """Get the end-effector frame ID."""
        return self._ee_frame_id
    
    def _get_frame_id(self, end_effector: str | None) -> int:
        """Get frame ID for the given end-effector name."""
        if end_effector is None:
            return self._ee_frame_id
        if not self._model.existFrame(end_effector):
            raise ValueError(f"Frame '{end_effector}' not found")
        return self._model.getFrameId(end_effector)
    
    def forward_kinematics(
        self, 
        q: np.ndarray, 
        end_effector: str | None = None
    ) -> Pose:
        """Compute forward kinematics using Pinocchio."""
        q = np.asarray(q, dtype=np.float64)
        
        # Run forward kinematics
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        
        frame_id = self._get_frame_id(end_effector)
        oMf = self._data.oMf[frame_id]
        
        # Extract position and rotation matrix
        position = oMf.translation.copy()
        rotation = oMf.rotation.copy()
        
        return Pose.from_matrix(position, rotation)
    
    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: np.ndarray | None = None,
        end_effector: str | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        """Compute inverse kinematics using iterative Jacobian method."""
        frame_id = self._get_frame_id(end_effector)
        
        if q_init is None:
            q = self._pin.neutral(self._model)
        else:
            q = np.array(q_init, dtype=np.float64)
        
        # Target SE3
        target_se3 = self._pin.SE3(
            target_pose.rotation_matrix,
            target_pose.position
        )
        
        eps = tolerance
        dt = 1.0  # Integration step
        damp = 1e-6  # Damping for pseudo-inverse
        
        for _ in range(max_iterations):
            # Forward kinematics
            self._pin.forwardKinematics(self._model, self._data, q)
            self._pin.updateFramePlacements(self._model, self._data)
            
            # Current frame placement
            oMf = self._data.oMf[frame_id]
            
            # Compute error in SE3
            dMf = target_se3.actInv(oMf)
            err = self._pin.log(dMf).vector
            
            if np.linalg.norm(err) < eps:
                return q, True
            
            # Compute Jacobian
            J = self._pin.computeFrameJacobian(
                self._model, self._data, q, frame_id,
                self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # Damped pseudo-inverse
            JJt = J @ J.T
            JJt_reg = JJt + damp * np.eye(6)
            v = -J.T @ np.linalg.solve(JJt_reg, err)
            
            # Integrate
            q = self._pin.integrate(self._model, q, v * dt)
        
        return q, False
    
    def jacobian(
        self, 
        q: np.ndarray, 
        end_effector: str | None = None
    ) -> np.ndarray:
        """Compute the geometric Jacobian using Pinocchio."""
        q = np.asarray(q, dtype=np.float64)
        frame_id = self._get_frame_id(end_effector)
        
        # Need to run forward kinematics first
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        
        # Compute frame Jacobian in world-aligned local frame
        J = self._pin.computeFrameJacobian(
            self._model, self._data, q, frame_id,
            self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        return J.copy()
    
    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse dynamics using RNEA algorithm.
        
        tau = M(q) * qdd + C(q, qd) * qd + g(q)
        
        For gravity compensation only, pass zero velocities and accelerations.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        
        # Recursive Newton-Euler Algorithm
        tau = self._pin.rnea(self._model, self._data, q, qd, qdd)
        
        return tau.copy()
    
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the joint-space mass matrix M(q)."""
        q = np.asarray(q, dtype=np.float64)
        
        self._pin.crba(self._model, self._data, q)
        M = self._data.M.copy()
        
        # Make symmetric (CRBA only computes upper triangle)
        M = np.triu(M) + np.triu(M, 1).T
        
        return M
    
    def coriolis_matrix(
        self, 
        q: np.ndarray, 
        qd: np.ndarray
    ) -> np.ndarray:
        """Compute the Coriolis matrix C(q, qd)."""
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        
        self._pin.computeCoriolisMatrix(self._model, self._data, q, qd)
        
        return self._data.C.copy()
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute the gravity vector g(q)."""
        q = np.asarray(q, dtype=np.float64)
        
        self._pin.computeGeneralizedGravity(self._model, self._data, q)
        
        return self._data.g.copy()
