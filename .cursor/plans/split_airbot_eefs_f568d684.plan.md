---
name: split airbot eefs
overview: Refactor AIRBOT E2B and G2 into independent `RobotArm` implementations aligned to the vendor reference flows, while cleaning generic/base APIs so AIRBOT-specific control semantics and end-effector naming no longer leak into shared code.
todos:
  - id: split-airbot-drivers
    content: Refactor `rollio/robot/airbot_eef.py` into independent `AIRBOTE2B` and `AIRBOTG2` `RobotArm` implementations aligned to the vendor examples
    status: completed
  - id: rename-frame-api
    content: Rename the generic `end_effector` state API in `rollio/robot/base.py` and all kinematics/caller references to neutral frame/task-space terminology
    status: completed
  - id: remove-generic-type-branches
    content: Replace AIRBOT-specific branches in runtime/config/pairing with robot-owned capability/default hooks
    status: completed
  - id: update-downstream-callers
    content: Update teleop, wizard, pseudo/pinocchio/AIRBOTPlay integrations, and public exports to the new abstractions
    status: completed
  - id: refresh-tests
    content: Revise AIRBOT, robot-base, runtime, config, and wizard tests to match the split driver semantics and renamed generic API
    status: completed
isProject: false
---

# Split AIRBOT EEF Drivers

## Goal

- Turn [rollio/robot/airbot_eef.py](rollio/robot/airbot_eef.py) into two independent `RobotArm` implementations: `AIRBOTE2B` and `AIRBOTG2`.
- Treat the vendor examples as the source of truth:
  - [external/airbot_play_examples/python/product/e2.py](external/airbot_play_examples/python/product/e2.py): `E2` must set `eef.e2.mode=4` and use empty `mit(EEFCommand1())` keepalive/read semantics.
  - [external/airbot_play_examples/python/product/g2_pos_ctrl.py](external/airbot_play_examples/python/product/g2_pos_ctrl.py): `G2` must use `PVT` with fixed `vel=10.0` and `current_threshold=10.0`.
- Remove AIRBOT-specific branching from generic code, and rename the generic `end_effector` API to neutral frame/task-space terminology.

## Key Refactor Areas

- [rollio/robot/airbot_eef.py](rollio/robot/airbot_eef.py): replace `AIRBOTEEFBase` with either two direct `RobotArm` subclasses plus a thin non-`RobotArm` helper for shared SDK plumbing only, or duplicated logic where the behaviors truly differ. The shared layer should stop owning semantic control behavior.
- [rollio/robot/base.py](rollio/robot/base.py): rename generic task-space API away from `EndEffectorState` / `read_end_effector_state()` / `end_effectors` / `end_effector_names`. Default target naming: `FrameState`, `read_frame_state()`, `frames`, and `frame_names`.
- [rollio/collect/runtime.py](rollio/collect/runtime.py), [rollio/config/schema.py](rollio/config/schema.py), and [rollio/config/pairing.py](rollio/config/pairing.py): remove hard-coded `airbot_e2b` / `airbot_g2` special cases and move robot-specific preview/keepalive/direct-map behavior behind robot-owned hooks or explicit per-class defaults.
- [rollio/collect/teleop.py](rollio/collect/teleop.py), [rollio/robot/pseudo_robot.py](rollio/robot/pinocchio_kinematics.py), [rollio/robot/airbot_play.py](rollio/robot/airbot_play.py), and [rollio/robot/**init**.py](rollio/robot/__init__.py): update all generic task-space references to the renamed base API.

## Concrete Couplings To Remove

```316:326:rollio/collect/runtime.py
def _preview_control_mode(robot: RobotArm) -> ControlMode | None:
    robot_type = robot.info.robot_type
    if robot_type == "airbot_e2b":
        return ControlMode.FREE_DRIVE
    if robot_type == "airbot_g2":
        return ControlMode.TARGET_TRACKING
    return None
```

```20:32:rollio/config/schema.py
def default_direct_map_allowlist(robot_type: str, ... ) -> list[str]:
    ...
    if normalized_type == "airbot_e2b":
        return ["airbot_g2"]
    if normalized_type == "airbot_g2":
        return ["airbot_e2b"]
```

- These type checks should become robot-provided capabilities/defaults instead of generic config/runtime knowledge.
- The current generic state API in [rollio/robot/base.py](rollio/robot/base.py) should be renamed to neutral frame/task-space terminology so a 1-DOF gripper is not treated as a special "end-effector flavored" robot type.

## Implementation Steps

1. Rebuild [rollio/robot/airbot_eef.py](rollio/robot/airbot_eef.py) around two independent classes.
  - `AIRBOTG2`: keep vendor-like lifecycle `init -> enable -> control_mode=PVT -> pvt(payload)` and preserve current public class name / module path / `robot_type`.
  - `AIRBOTE2B`: implement vendor-like lifecycle `init -> enable -> eef.e2.mode=4 -> mit(empty_cmd)` refresh loop, and stop reusing G2-style abstractions such as generic target-tracking or echoed `pos/vel` MIT refresh.
  - Keep only a thin shared helper for SDK import/create/init/close, command allocation, and simple 1-DOF measurement extraction if it still reduces duplication.
2. Rename the generic task-space API in [rollio/robot/base.py](rollio/robot/base.py).
  - Replace `EndEffectorState` with `FrameState`.
  - Replace `read_end_effector_state()` with `read_frame_state()`.
  - Replace `RobotState.end_effectors` with `RobotState.frames`.
  - Replace `KinematicsModel.end_effector_names` with `frame_names` and update default FK/Jacobian helpers accordingly.
3. Move AIRBOT-specific preview/direct-map defaults out of generic code.
  - Add robot-owned hooks/defaults for preview mode / keepalive / direct-map compatibility, then switch [rollio/collect/runtime.py](rollio/collect/runtime.py), [rollio/config/schema.py](rollio/config/schema.py), and [rollio/config/pairing.py](rollio/config/pairing.py) to use those hooks instead of type-string branches.
4. Update downstream consumers.
  - Adjust [rollio/collect/teleop.py](rollio/collect/teleop.py) to use the renamed frame/task-space API.
  - Update [rollio/robot/pseudo_robot.py](rollio/robot/pseudo_robot.py), [rollio/robot/pinocchio_kinematics.py](rollio/robot/pinocchio_kinematics.py), and [rollio/robot/airbot_play.py](rollio/robot/airbot_play.py) to the new naming.
  - Keep AIRBOT public exports stable in [rollio/robot/**init**.py](rollio/robot/__init__.py).
5. Refresh tests to match the new semantics.
  - [tests/test_airbot.py](tests/test_airbot.py): split E2B and G2 expectations by vendor behavior.
  - [tests/test_robot.py](tests/test_robot.py): rename generic task-space API assertions.
  - [tests/test_collect_runtime.py](tests/test_collect_runtime.py), [tests/test_config_helpers.py](tests/test_config_helpers.py), and [tests/test_tui_wizard.py](tests/test_tui_wizard.py): update capability/default assumptions after the AIRBOT-specific branches move out of shared code.

## Validation

- Run the non-hardware test subset that covers robot base abstractions and AIRBOT mocks.
- Verify that mocked `AIRBOTG2` still emits `PVT` payloads matching the vendor script.
- Verify that mocked `AIRBOTE2B` now configures `eef.e2.mode=4` and uses empty `MIT` keepalive refresh like the vendor script.
- Smoke-check any runtime preview codepaths affected by the new robot capability hooks.

