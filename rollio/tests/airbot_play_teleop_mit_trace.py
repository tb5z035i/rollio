"""AIRBOT Play leader/follower MIT-tracking trace experiment.

This hardware script is meant to help diagnose whether the follower-arm MIT
tracking gains or follower-G2 MIT tracking are the root cause of visible
trembling during teleoperation.

Setup:
- Leader on one CAN interface: AIRBOT Play + E2B
- Follower on another CAN interface: AIRBOT Play + G2

Runtime behavior:
- Leader arm runs in Rollio free-drive mode with gravity compensation
- Leader E2B stays in feedback mode for manual guidance
- Follower arm runs Rollio MIT target tracking with configurable gains
- Follower G2 follows the leader E2B with Rollio MIT target tracking
- The script publishes live JSON telemetry to PlotJuggler over UDP using a
  background publisher thread so socket/JSON work stays off the control path
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rollio.defaults import DEFAULT_CONTROL_HZ

PLOTJUGGLER_UDP_HOST = "127.0.0.1"
PLOTJUGGLER_UDP_PORT = 9870
ARM_JOINT_LABELS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ARM_STREAM_KEYS = tuple(
    (
        f"leader/{label}",
        f"target/{label}",
        f"follower/{label}",
        f"error/{label}",
    )
    for label in ARM_JOINT_LABELS
)


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    del signum, frame
    print("\n\nInterrupted by user. Stopping...")
    sys.exit(130)


def _parse_axis_values(raw: str, *, expected: int, name: str) -> np.ndarray:
    values = [float(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if len(values) == 1:
        return np.full(expected, values[0], dtype=np.float64)
    if len(values) != expected:
        raise ValueError(
            f"{name} must provide either 1 value or {expected} comma-separated values. "
            f"Got {len(values)} values."
        )
    return np.asarray(values, dtype=np.float64)


def _clamp_g2_target(position: float) -> float:
    return max(0.0, min(float(position), 0.072))


@dataclass
class TraceSample:
    timestamp_sec: float
    leader_position: np.ndarray
    mapped_target_position: np.ndarray
    follower_position: np.ndarray


def _read_position(robot: Any) -> np.ndarray | None:
    state = robot.read_joint_state()
    if not state.is_valid or state.position is None:
        return None
    return np.asarray(state.position, dtype=np.float64).copy()


def _read_position_and_velocity(
    robot: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    state = robot.read_joint_state()
    if not state.is_valid or state.position is None:
        return None, None
    position = np.asarray(state.position, dtype=np.float64).copy()
    if state.velocity is None:
        velocity = np.zeros_like(position)
    else:
        velocity = np.asarray(state.velocity, dtype=np.float64).copy()
    return position, velocity


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):7.3f}" for v in values) + "]"


def _build_plotjuggler_message(sample: TraceSample) -> dict[str, float]:
    payload: dict[str, float] = {
        "timestamp": float(sample.timestamp_sec),
    }
    arm_error = sample.follower_position[:6] - sample.mapped_target_position[:6]

    for idx, (leader_key, target_key, follower_key, error_key) in enumerate(
        ARM_STREAM_KEYS
    ):
        leader_value = float(sample.leader_position[idx])
        target_value = float(sample.mapped_target_position[idx])
        follower_value = float(sample.follower_position[idx])
        payload[leader_key] = leader_value
        payload[target_key] = target_value
        payload[follower_key] = follower_value
        payload[error_key] = follower_value - target_value

    leader_e2b = float(sample.leader_position[6])
    target_g2 = float(sample.mapped_target_position[6])
    follower_g2 = float(sample.follower_position[6])
    payload["leader/e2b"] = leader_e2b
    payload["target/g2"] = target_g2
    payload["follower/g2"] = follower_g2
    payload["error/g2"] = follower_g2 - target_g2
    payload["error/arm_rms"] = float(np.sqrt(np.mean(np.square(arm_error))))
    return payload


def _encode_plotjuggler_message(sample: TraceSample) -> bytes:
    return json.dumps(
        _build_plotjuggler_message(sample),
        separators=(",", ":"),
    ).encode("utf-8")


class _PlotJugglerUdpPublisher:
    """Low-overhead latest-sample UDP JSON publisher for PlotJuggler.

    The control loop only swaps in the newest sample. JSON encoding and UDP
    I/O happen on a background thread so telemetry publishing does not add
    noticeable latency to the follow loop.
    """

    def __init__(self, host: str, port: int, publish_hz: float) -> None:
        self._address = (str(host), int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._period_sec = 1.0 / max(float(publish_hz), 1e-6)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest_sample: TraceSample | None = None
        self._latest_seq = 0
        self._sent_seq = 0
        self._sent_packets = 0

    @property
    def address(self) -> tuple[str, int]:
        return self._address

    @property
    def sent_packets(self) -> int:
        return self._sent_packets

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="plotjuggler-udp-publisher",
            daemon=True,
        )
        self._thread.start()

    def publish_latest(self, sample: TraceSample) -> None:
        with self._lock:
            self._latest_sample = sample
            self._latest_seq += 1

    def _snapshot_latest(self) -> tuple[TraceSample | None, int]:
        with self._lock:
            return self._latest_sample, self._latest_seq

    def _run(self) -> None:
        next_publish = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_publish:
                self._stop_event.wait(next_publish - now)
                continue
            next_publish += self._period_sec

            sample, sample_seq = self._snapshot_latest()
            if sample is None or sample_seq == self._sent_seq:
                continue

            try:
                self._sock.sendto(_encode_plotjuggler_message(sample), self._address)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                continue

            self._sent_seq = sample_seq
            self._sent_packets += 1

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._period_sec * 4.0))
            self._thread = None
        self._sock.close()


def _run_follow_tick(
    *,
    leader_arm: Any,
    leader_eef: Any,
    follower_arm: Any,
    follower_g2: Any,
    eef_scale: float,
    use_leader_velocity: bool,
    timestamp_sec: float | None = None,
) -> TraceSample | None:
    leader_arm.step_free_drive()
    leader_eef.step_free_drive()

    leader_arm_pos, leader_arm_vel = _read_position_and_velocity(leader_arm)
    leader_eef_pos = _read_position(leader_eef)
    if (
        leader_arm_pos is None
        or leader_arm_vel is None
        or leader_eef_pos is None
        or leader_eef_pos.size == 0
    ):
        return None

    arm_velocity_target = (
        leader_arm_vel if use_leader_velocity else np.zeros_like(leader_arm_pos)
    )
    leader_g2_target = _clamp_g2_target(float(leader_eef_pos[0]) * eef_scale)

    follower_arm.step_target_tracking(
        position_target=leader_arm_pos,
        velocity_target=arm_velocity_target,
        add_gravity_compensation=True,
    )
    follower_g2.step_target_tracking(
        position_target=np.array([leader_g2_target], dtype=np.float64),
        velocity_target=np.zeros(1, dtype=np.float64),
        add_gravity_compensation=False,
    )

    follower_arm_pos = _read_position(follower_arm)
    follower_g2_pos = _read_position(follower_g2)
    if follower_arm_pos is None or follower_g2_pos is None or follower_g2_pos.size == 0:
        return None

    leader_position = np.concatenate((leader_arm_pos, leader_eef_pos[:1]))
    mapped_target_position = np.concatenate(
        (leader_arm_pos, np.array([leader_g2_target], dtype=np.float64))
    )
    follower_position = np.concatenate((follower_arm_pos, follower_g2_pos[:1]))

    return TraceSample(
        timestamp_sec=time.time() if timestamp_sec is None else float(timestamp_sec),
        leader_position=leader_position,
        mapped_target_position=mapped_target_position,
        follower_position=follower_position,
    )


def _print_summary(samples: list[TraceSample]) -> None:
    if not samples:
        print("No valid samples were recorded.")
        return

    target = np.stack([sample.mapped_target_position for sample in samples], axis=0)
    follower = np.stack([sample.follower_position for sample in samples], axis=0)
    error = follower - target
    joint_labels = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "g2",
    )

    print()
    print("=" * 72)
    print("Tracking Error Summary")
    print("=" * 72)
    for idx, label in enumerate(joint_labels):
        axis_error = error[:, idx]
        rms = float(np.sqrt(np.mean(np.square(axis_error))))
        mean_abs = float(np.mean(np.abs(axis_error)))
        max_abs = float(np.max(np.abs(axis_error)))
        unit = "m" if idx == 6 else "rad"
        print(
            f"{label:<8} rms={rms:8.5f} {unit}  "
            f"mean_abs={mean_abs:8.5f} {unit}  "
            f"max_abs={max_abs:8.5f} {unit}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leader/follower AIRBOT Play teleoperation trace with leader "
            "gravity compensation and follower MIT tracking, then stream JSON to PlotJuggler."
        )
    )
    parser.add_argument(
        "--leader-can",
        default="can0",
        help="CAN interface for the leader AIRBOT Play + E2B (default: can0)",
    )
    parser.add_argument(
        "--follower-can",
        default="can1",
        help="CAN interface for the follower AIRBOT Play + G2 (default: can1)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=15.0,
        help="Recording duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--control-hz",
        type=int,
        default=DEFAULT_CONTROL_HZ,
        help=f"Control/update frequency in Hz (default: {DEFAULT_CONTROL_HZ})",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=2.0,
        help="Follower settle time before recording starts (default: 2.0)",
    )
    parser.add_argument(
        "--kp",
        default="200,200,200,50,50,50",
        help="Follower arm MIT kp (scalar or 6 comma-separated values)",
    )
    parser.add_argument(
        "--kd",
        default="5,5,5,1,1,1",
        help="Follower arm MIT kd (scalar or 6 comma-separated values)",
    )
    parser.add_argument(
        "--eef-scale",
        type=float,
        default=1.5,
        help="Scale factor from leader E2B opening to follower G2 target (default: 1.5)",
    )
    parser.add_argument(
        "--zero-velocity-target",
        action="store_true",
        help="Use zero velocity targets for the follower arm instead of leader joint velocity",
    )
    parser.add_argument(
        "--udp-host",
        default=PLOTJUGGLER_UDP_HOST,
        help=f"PlotJuggler UDP host (default: {PLOTJUGGLER_UDP_HOST})",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=PLOTJUGGLER_UDP_PORT,
        help=f"PlotJuggler UDP port (default: {PLOTJUGGLER_UDP_PORT})",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=25.0,
        help="Background UDP publish rate to PlotJuggler in Hz (default: 25.0)",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=5.0,
        help="Console status rate while recording (default: 5.0 Hz)",
    )
    parser.add_argument(
        "--no-return-zero",
        action="store_true",
        help="Do not return the follower arm/gripper to zero at shutdown",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        arm_kp = _parse_axis_values(args.kp, expected=6, name="kp")
        arm_kd = _parse_axis_values(args.kd, expected=6, name="kd")
        if args.duration <= 0.0:
            raise ValueError("duration must be positive.")
        if args.control_hz <= 0:
            raise ValueError("control-hz must be positive.")
        if args.warmup_sec < 0.0:
            raise ValueError("warmup-sec must be non-negative.")
        if args.eef_scale <= 0.0:
            raise ValueError("eef-scale must be positive.")
        if args.publish_hz <= 0.0:
            raise ValueError("publish-hz must be positive.")
        if args.udp_port <= 0:
            raise ValueError("udp-port must be positive.")
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    from rollio.robot import AIRBOTE2B, AIRBOTG2, AIRBOTPlay, is_airbot_available
    from rollio.robot.can_utils import is_can_interface_up

    if not is_airbot_available():
        print("Error: airbot_hardware_py is not installed.")
        print("Install the AIRBOT SDK first.")
        return 1

    if not is_can_interface_up(args.leader_can):
        print(
            f"Error: leader CAN interface '{args.leader_can}' is not available or not UP."
        )
        return 1
    if not is_can_interface_up(args.follower_can):
        print(
            f"Error: follower CAN interface '{args.follower_can}' is not available or not UP."
        )
        return 1

    verbose = not args.quiet
    leader_arm = None
    leader_eef = None
    follower_arm = None
    follower_g2 = None
    publisher: _PlotJugglerUdpPublisher | None = None
    samples: list[TraceSample] = []

    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    try:
        publisher = _PlotJugglerUdpPublisher(
            args.udp_host,
            args.udp_port,
            args.publish_hz,
        )
        publisher.start()

        if verbose:
            print()
            print("=" * 72)
            print("AIRBOT Play MIT Tracking Trace")
            print("=" * 72)
            print(f"Leader   : {args.leader_can}  (AIRBOT Play + E2B)")
            print(f"Follower : {args.follower_can}  (AIRBOT Play + G2)")
            print(f"Duration : {args.duration:.1f}s")
            print(f"Control  : {args.control_hz} Hz")
            print(f"kp       : {_format_vector(arm_kp)}")
            print(f"kd       : {_format_vector(arm_kd)}")
            print(f"E2B->G2  : scale={args.eef_scale:.3f}, tracking=mit")
            print(
                f"UDP      : {publisher.address[0]}:{publisher.address[1]} "
                f"@ {args.publish_hz:.1f} Hz"
            )
            print()
            print("PlotJuggler setup:")
            print("- Select `UDP Server` as the streaming source")
            print("- Set message protocol to `JSON`")
            print("- Bind to `0.0.0.0` or leave the default all-interfaces bind")
            print("- Enable `use timestamp if available` with field name `timestamp`")
            print()

        leader_arm = AIRBOTPlay(
            can_interface=args.leader_can,
            control_frequency=args.control_hz,
        )
        leader_eef = AIRBOTE2B(
            can_interface=args.leader_can,
            control_frequency=args.control_hz,
        )
        follower_arm = AIRBOTPlay(
            can_interface=args.follower_can,
            control_frequency=args.control_hz,
            target_tracking_mode="mit",
        )
        follower_g2 = AIRBOTG2(
            can_interface=args.follower_can,
            control_frequency=args.control_hz,
            target_tracking_mode="mit",
        )

        leader_arm.open()
        leader_eef.open()
        follower_arm.open()
        follower_g2.open()

        if not leader_arm.enable():
            raise RuntimeError("Failed to enable leader arm.")
        if not leader_eef.enable():
            raise RuntimeError("Failed to enable leader E2B.")
        if not follower_arm.enable():
            raise RuntimeError("Failed to enable follower arm.")
        if not follower_g2.enable():
            raise RuntimeError("Failed to enable follower G2.")

        follower_arm.TARGET_TRACKING_KP = arm_kp.copy()
        follower_arm.TARGET_TRACKING_KD = arm_kd.copy()

        if not leader_arm.enter_free_drive():
            raise RuntimeError("Failed to enter leader arm free-drive mode.")
        if not leader_eef.enter_free_drive():
            raise RuntimeError("Failed to enter leader E2B feedback mode.")
        if not follower_arm.enter_target_tracking():
            raise RuntimeError("Failed to enter follower arm MIT target-tracking mode.")
        if not follower_g2.enter_target_tracking():
            raise RuntimeError("Failed to enter follower G2 MIT target-tracking mode.")

        dt = 1.0 / max(args.control_hz, 1)
        if verbose:
            print(
                "Hold the leader still while the follower settles, then move the leader by hand."
            )
            print("Press Ctrl+C to stop early.\n")

        warmup_deadline = time.monotonic() + args.warmup_sec
        next_tick = time.monotonic()
        while time.monotonic() < warmup_deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
                continue
            next_tick += dt
            _run_follow_tick(
                leader_arm=leader_arm,
                leader_eef=leader_eef,
                follower_arm=follower_arm,
                follower_g2=follower_g2,
                eef_scale=args.eef_scale,
                use_leader_velocity=not args.zero_velocity_target,
            )

        start = time.monotonic()
        next_tick = start
        last_print = start
        print_period = 1.0 / max(args.print_hz, 1e-6)
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration:
                break
            if now < next_tick:
                time.sleep(next_tick - now)
                continue
            next_tick += dt

            sample = _run_follow_tick(
                leader_arm=leader_arm,
                leader_eef=leader_eef,
                follower_arm=follower_arm,
                follower_g2=follower_g2,
                eef_scale=args.eef_scale,
                use_leader_velocity=not args.zero_velocity_target,
                timestamp_sec=time.time(),
            )
            if sample is None:
                continue
            samples.append(sample)

            if publisher is not None:
                publisher.publish_latest(sample)

            if verbose and (now - last_print >= print_period):
                arm_error = (
                    sample.follower_position[:6] - sample.mapped_target_position[:6]
                )
                eef_error = (
                    sample.follower_position[6] - sample.mapped_target_position[6]
                )
                print(
                    f"\rt={elapsed:6.2f}s  "
                    f"arm_rms={np.sqrt(np.mean(np.square(arm_error))):7.4f} rad  "
                    f"eef_err={eef_error:+7.4f} m  "
                    f"udp={publisher.sent_packets if publisher is not None else 0}",
                    end="",
                    flush=True,
                )
                last_print = now

        if verbose:
            print()
            print(f"\nRecorded {len(samples)} valid samples.")
            print(
                "Published "
                f"{publisher.sent_packets if publisher is not None else 0} "
                "UDP JSON packets."
            )
            if follower_arm is not None:
                metrics = follower_arm.control_loop_metrics()
                print(
                    "Follower arm loop: "
                    f"target={metrics.target_interval_ms:6.3f}ms  "
                    f"avg={metrics.avg_interval_ms or 0.0:6.3f}ms  "
                    f"max={metrics.max_interval_ms or 0.0:6.3f}ms  "
                    f"runs={metrics.run_count}"
                )

        if not samples:
            raise RuntimeError("No valid samples were recorded.")
        _print_summary(samples)
        return 0

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        return 130
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        signal.signal(signal.SIGINT, original_handler)

        if (
            follower_g2 is not None
            and follower_arm is not None
            and not args.no_return_zero
        ):
            try:
                follower_g2.move_to_zero(timeout=5.0)
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            try:
                follower_arm.move_to_home(timeout=15.0)
            except (OSError, RuntimeError, ValueError, TypeError):
                pass

        if publisher is not None:
            try:
                publisher.close()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass

        for robot in (follower_g2, follower_arm, leader_eef, leader_arm):
            if robot is None:
                continue
            try:
                robot.disable()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            try:
                robot.close()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
