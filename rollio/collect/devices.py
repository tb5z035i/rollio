"""Pluggable device factories for cameras and robots."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rollio.config.schema import CameraConfig, RobotConfig, RollioConfig
from rollio.defaults import DEFAULT_CONTROL_HZ
from rollio.robot import AIRBOTE2B, AIRBOTG2, AIRBOTPlay, RobotArm
from rollio.robot.pseudo_robot import PseudoRobotArm
from rollio.sensors import ImageSensor, PseudoCamera, RealSenseCamera, V4L2Camera

CameraFactory = Callable[[CameraConfig], ImageSensor]
RobotFactory = Callable[[RobotConfig], RobotArm]

_CAMERA_FACTORIES: dict[str, CameraFactory] = {}
_ROBOT_FACTORIES: dict[str, RobotFactory] = {}
_DEFAULT_FACTORIES_REGISTERED = False


def register_camera_factory(
    sensor_type: str,
    factory: CameraFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one camera factory."""
    key = str(sensor_type).strip()
    if not key:
        raise ValueError("Camera factory type must be a non-empty string")
    if key in _CAMERA_FACTORIES and not replace:
        raise ValueError(f"Camera factory already registered for type '{key}'")
    _CAMERA_FACTORIES[key] = factory


def register_robot_factory(
    robot_type: str,
    factory: RobotFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one robot factory."""
    key = str(robot_type).strip()
    if not key:
        raise ValueError("Robot factory type must be a non-empty string")
    if key in _ROBOT_FACTORIES and not replace:
        raise ValueError(f"Robot factory already registered for type '{key}'")
    _ROBOT_FACTORIES[key] = factory


def registered_camera_types() -> tuple[str, ...]:
    ensure_default_device_factories()
    return tuple(sorted(_CAMERA_FACTORIES))


def registered_robot_types() -> tuple[str, ...]:
    ensure_default_device_factories()
    return tuple(sorted(_ROBOT_FACTORIES))


def _build_pseudo_camera(cam_cfg: CameraConfig) -> ImageSensor:
    return PseudoCamera(
        name=cam_cfg.name,
        width=cam_cfg.width,
        height=cam_cfg.height,
        fps=cam_cfg.fps,
    )


def _build_v4l2_camera(cam_cfg: CameraConfig) -> ImageSensor:
    return V4L2Camera(
        name=cam_cfg.name,
        device=cam_cfg.device,
        width=cam_cfg.width,
        height=cam_cfg.height,
        fps=cam_cfg.fps,
        pixel_format=cam_cfg.pixel_format,
    )


def _build_realsense_camera(cam_cfg: CameraConfig) -> ImageSensor:
    channel = cam_cfg.channel or "color"
    kwargs: dict[str, Any] = {
        "name": cam_cfg.name,
        "device": str(cam_cfg.device).split(":")[0],
        "enable_color": channel == "color",
        "enable_depth": channel == "depth",
        "enable_infrared": channel == "infrared",
        "preview_channel": channel,
    }
    if channel == "color":
        kwargs.update(width=cam_cfg.width, height=cam_cfg.height, fps=cam_cfg.fps)
    elif channel == "depth":
        kwargs.update(
            width=cam_cfg.width,
            height=cam_cfg.height,
            fps=cam_cfg.fps,
            depth_width=cam_cfg.width,
            depth_height=cam_cfg.height,
            depth_fps=cam_cfg.fps,
            depth_format=cam_cfg.pixel_format,
        )
    else:
        kwargs.update(
            width=cam_cfg.width,
            height=cam_cfg.height,
            fps=cam_cfg.fps,
            ir_width=cam_cfg.width,
            ir_height=cam_cfg.height,
            ir_fps=cam_cfg.fps,
            ir_format=cam_cfg.pixel_format,
        )
    kwargs.update(cam_cfg.options)
    return RealSenseCamera(**kwargs)


def _build_pseudo_robot(robot_cfg: RobotConfig) -> RobotArm:
    return PseudoRobotArm(
        name=robot_cfg.name,
        n_dof=robot_cfg.num_joints,
        noise_level=float(robot_cfg.options.get("noise_level", 0.0)),
        control_frequency=float(
            robot_cfg.options.get("control_frequency", DEFAULT_CONTROL_HZ)
        ),
    )


def _build_airbot_robot(robot_cfg: RobotConfig) -> RobotArm:
    if AIRBOTPlay is None:
        raise ImportError("AIRBOTPlay support is not available in this environment")
    kwargs = dict(robot_cfg.options)
    kwargs.setdefault("can_interface", robot_cfg.device or "can0")
    return AIRBOTPlay(**kwargs)


def _build_airbot_e2b_robot(robot_cfg: RobotConfig) -> RobotArm:
    if AIRBOTE2B is None:
        raise ImportError("AIRBOTE2B support is not available in this environment")
    kwargs = dict(robot_cfg.options)
    kwargs.setdefault("can_interface", robot_cfg.device or "can0")
    return AIRBOTE2B(**kwargs)


def _build_airbot_g2_robot(robot_cfg: RobotConfig) -> RobotArm:
    if AIRBOTG2 is None:
        raise ImportError("AIRBOTG2 support is not available in this environment")
    kwargs = dict(robot_cfg.options)
    kwargs.setdefault("can_interface", robot_cfg.device or "can0")
    return AIRBOTG2(**kwargs)


def _apply_robot_config_metadata(robot: RobotArm, robot_cfg: RobotConfig) -> RobotArm:
    robot.info.robot_type = robot_cfg.type
    robot.info.n_dof = robot_cfg.num_joints
    robot.info.properties["config_name"] = robot_cfg.name
    robot.info.properties["config_role"] = robot_cfg.role
    robot.info.properties["config_device"] = robot_cfg.device
    return robot


def ensure_default_device_factories() -> None:
    """Register the built-in camera and robot factories once."""
    global _DEFAULT_FACTORIES_REGISTERED
    if _DEFAULT_FACTORIES_REGISTERED:
        return

    register_camera_factory("pseudo", _build_pseudo_camera, replace=True)
    register_camera_factory("v4l2", _build_v4l2_camera, replace=True)
    register_camera_factory("realsense", _build_realsense_camera, replace=True)

    register_robot_factory("pseudo", _build_pseudo_robot, replace=True)
    register_robot_factory("airbot_play", _build_airbot_robot, replace=True)
    register_robot_factory("airbot_e2b", _build_airbot_e2b_robot, replace=True)
    register_robot_factory("airbot_g2", _build_airbot_g2_robot, replace=True)

    _DEFAULT_FACTORIES_REGISTERED = True


def build_camera_from_config(cam_cfg: CameraConfig) -> ImageSensor:
    """Instantiate one camera from config using the registry."""
    ensure_default_device_factories()
    try:
        factory = _CAMERA_FACTORIES[cam_cfg.type]
    except KeyError as exc:
        known = ", ".join(sorted(_CAMERA_FACTORIES))
        raise NotImplementedError(
            f"Unsupported camera type: {cam_cfg.type}. "
            f"Registered camera types: {known or '(none)'}"
        ) from exc
    return factory(cam_cfg)


def build_robot_from_config(robot_cfg: RobotConfig) -> RobotArm:
    """Instantiate one robot from config using the registry."""
    ensure_default_device_factories()
    try:
        factory = _ROBOT_FACTORIES[robot_cfg.type]
    except KeyError as exc:
        known = ", ".join(sorted(_ROBOT_FACTORIES))
        raise NotImplementedError(
            f"Unsupported robot type: {robot_cfg.type}. "
            f"Registered robot types: {known or '(none)'}"
        ) from exc
    return _apply_robot_config_metadata(factory(robot_cfg), robot_cfg)


def build_cameras_from_config(cfg: RollioConfig) -> dict[str, ImageSensor]:
    """Instantiate cameras from config using registered factories."""
    return {cam_cfg.name: build_camera_from_config(cam_cfg) for cam_cfg in cfg.cameras}


def build_robots_from_config(cfg: RollioConfig) -> dict[str, RobotArm]:
    """Instantiate robots from config using registered factories."""
    return {robot_cfg.name: build_robot_from_config(robot_cfg) for robot_cfg in cfg.robots}
