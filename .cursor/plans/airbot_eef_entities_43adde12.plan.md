---
name: AIRBOT EEF Entities
overview: Treat AIRBOT end effectors as ordinary `robots[]` entries backed by new robot-driver types, preserve one flat exported `action` vector, and support AIRBOT arm plus EEF teleop using the existing pairing model.
todos:
  - id: add-airbot-eef-robot-types
    content: Add AIRBOT EEF robot-driver types that reuse the existing robot contract and shared device access.
    status: cancelled
  - id: scan-persist-eefs
    content: Teach discovery, factories, validation, and wizard flow to surface and persist EEFs as `robots[]` entries.
    status: completed
  - id: entity-teleop-policies
    content: Reuse teleop pairing for both arms and EEFs, including E2B->G2 width control and safe handling of unsupported EEF directions.
    status: completed
  - id: per-entity-export
    content: Refactor runtime recording and LeRobot export for per-robot DOF shapes and one flat multi-robot `action` vector.
    status: completed
  - id: eef-test-coverage
    content: Add runtime, AIRBOT, writer, config, and wizard tests for separate EEF robot entries and recorded gripper channels.
    status: completed
isProject: false
---

# AIRBOT EEF Entity Support

## Locked Decisions

- End effectors are ordinary `robots[]` entries, distinguished only by `robots[].type`; there is no new abstract base class and no separate config section.
- AIRBOT discovery should emit an arm robot plus an optional attached EEF robot that reuse the same physical `device` value.
- Shared AIRBOT access must be serialized per physical `device` during probing and runtime control.
- Config validation should reject duplicate `(type, device)` combinations while allowing different types to share one `device` such as `airbot_play@can0` plus `airbot_g2@can0`.
- Setup should let you configure teleop pairs for arms and for EEFs using the same `teleop_pairs` mechanism, and the wizard should support entering multiple leader-follower pairs in sequence rather than a single one-shot pairing pass.
- Each robot config should persist a direct-mapping allowlist of peer robot types.
- Direct mapping is allowed only when both robots have matching `num_joints` and each robot type appears in the other side's allowlist.
- Types that support same-type direct mapping should include their own type in that allowlist, so `airbot_play` should default to allowing `airbot_play`.
- `E2B -> G2` uses absolute opening width and should work only because `airbot_e2b` and `airbot_g2` mutually allowlist each other, not because of any hardcoded special-case mapper.
- Episode export keeps a single flat `action` vector, built by concatenating follower command DOFs in configured teleop-pair order.

## Architecture

```mermaid
flowchart LR
scanAirbot[ScanAIRBOT]
scanAirbot --> discoveredRobots[DiscoveredRobots]
discoveredRobots --> configRobots[Configrobots[]]
configRobots --> robotFactories[RobotFactories]
robotFactories --> sharedDeviceIO[SharedDeviceIO]
sharedDeviceIO --> runtime[AsyncCollectionRuntime]
runtime --> armPairs[ArmTeleopPairs]
runtime --> eefPairs[EEFTeleopPairs]
runtime --> telemetry[RobotTelemetry]
armPairs --> recorder[EpisodeAccumulator]
eefPairs --> recorder
telemetry --> recorder
recorder --> writer[LeRobotV21Writer]
```



## Implementation Steps

- Add AIRBOT EEF robot types on the existing robot contract.
  - Implement AIRBOT-backed EEF drivers in a new module such as `[/home/tb5z035i/workspace/data-collect/rollio/rollio/robot/airbot_eef.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/robot/airbot_eef.py)` with distinct types like `airbot_e2b` and `airbot_g2`.
  - Keep them on the current robot/factory path by conforming to the existing `RobotArm` contract with `n_dof=1`, `position` expressed as absolute opening width, and lightweight no-op/trivial implementations where arm-specific methods are not meaningful.
  - Factor shared AIRBOT transport/session helpers so `airbot_play`, `airbot_e2b`, and `airbot_g2` serialize reads and writes when they share one `canX`.
- Keep configuration flat and make discovery persist EEFs as robots.
  - Reuse `[/home/tb5z035i/workspace/data-collect/rollio/rollio/config/schema.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/config/schema.py)` `RobotConfig`; do not add a new top-level section or a new pairing model, but add a persisted field such as `direct_map_allowlist: list[str]`.
  - Extend AIRBOT discovery in `[/home/tb5z035i/workspace/data-collect/rollio/rollio/robot/scanner.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/robot/scanner.py)` plus AIRBOT helpers so one physical AIRBOT can emit multiple logical robot entries from one device without probing in parallel.
  - Register the new EEF robot types in `[/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/devices.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/devices.py)` and add config validation that rejects duplicate `(type, device)` combinations.
  - Populate sensible default allowlists during discovery/config generation so arm and EEF entries serialize their direct-mapping compatibility into the config file, including self-allowlisting for same-type mappings such as `airbot_play -> airbot_play`.
  - Update `[/home/tb5z035i/workspace/data-collect/rollio/rollio/tui/wizard.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/tui/wizard.py)` to surface EEFs as separate selectable robots, persist them into `robots[]`, and let users add multiple arm and EEF leader-follower pairs in the same teleop flow.
- Reuse teleop pairing for both arms and EEFs.
  - Keep `[/home/tb5z035i/workspace/data-collect/rollio/rollio/config/pairing.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/config/pairing.py)` and `[/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/teleop.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/teleop.py)` as the only pairing/mapping path.
  - Make `joint_direct` capability-driven rather than type-special-cased: it should be selected only when the two robot entries have matching DOF counts and mutually include each other's `type` in `direct_map_allowlist`.
  - Configure AIRBOT EEF types so `airbot_e2b` and `airbot_g2` direct-map through this generic rule while still using absolute opening width as their shared 1-DOF signal.
  - Keep unsupported EEF directions out of auto-suggestions; if explicitly configured, they should safely resolve to no control command.
- Refactor recording and export around logical robots instead of a single arm shape.
  - Update `[/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/runtime.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/collect/runtime.py)` to record each logical robot entry independently, including separate EEF channels.
  - Rework `[/home/tb5z035i/workspace/data-collect/rollio/rollio/episode/writer.py](/home/tb5z035i/workspace/data-collect/rollio/rollio/episode/writer.py)` to track per-robot state shapes, export separate arm/EEF observations, and compose one flat `action` vector from follower pair targets in teleop-pair order.
  - Add metadata describing action slices and per-robot dimensions so downstream readers can decode the flat action vector reliably.
- Add coverage for the robot-style EEF path.
  - Extend `[/home/tb5z035i/workspace/data-collect/rollio/tests/test_airbot.py](/home/tb5z035i/workspace/data-collect/rollio/tests/test_airbot.py)` for AIRBOT EEF detection plus mocked width read/command behavior.
  - Extend `[/home/tb5z035i/workspace/data-collect/rollio/tests/test_collect_runtime.py](/home/tb5z035i/workspace/data-collect/rollio/tests/test_collect_runtime.py)` and `[/home/tb5z035i/workspace/data-collect/rollio/tests/test_collect_devices.py](/home/tb5z035i/workspace/data-collect/rollio/tests/test_collect_devices.py)` for same-device arm+EEF runtime, `E2B -> G2` teleop, and flat multi-robot actions.
  - Extend `[/home/tb5z035i/workspace/data-collect/rollio/tests/test_tui_wizard.py](/home/tb5z035i/workspace/data-collect/rollio/tests/test_tui_wizard.py)` and config-level tests for discovery, duplicate `(type, device)` rejection, persisted allowlists, self-allowlisted same-type mapping, mutual-allowlist direct mapping, pairing defaults, and repeated multi-pair wizard entry.

## Main Risk

- Reusing `RobotArm` keeps the design modular and avoids a second hierarchy, but the contract is arm-centric. The EEF drivers will need lightweight compatibility code for kinematics/free-drive methods, and the runtime must consistently serialize shared-device access so one `canX` does not see conflicting operations.

