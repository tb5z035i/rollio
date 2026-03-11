"""AIRBOT G2 hardware experiments.

This module provides direct-SDK and Rollio-wrapper experiments for the
standalone AIRBOT G2 gripper. The goal is not just to send commands, but to
measure whether the actual position follows the commanded target with a small
allowed latency.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    del signum, frame
    print("\n\nInterrupted by user. Stopping...")
    sys.exit(130)


def _validate_range(center: float, amplitude: float) -> tuple[float, float]:
    amplitude = abs(float(amplitude))
    center = float(center)
    lo = center - amplitude
    hi = center + amplitude
    if lo < 0.0 or hi > 0.072:
        raise ValueError(
            "Target range must stay within [0.0, 0.072] meters. "
            f"Got center={center:.4f}, amplitude={amplitude:.4f} -> "
            f"[{lo:.4f}, {hi:.4f}]"
        )
    return center, amplitude


def _parse_targets(raw: str) -> list[float]:
    targets = []
    for chunk in str(raw).split(","):
        value = chunk.strip()
        if not value:
            continue
        target = float(value)
        if not 0.0 <= target <= 0.072:
            raise ValueError(f"Fixed target {target:.4f} is outside [0.0, 0.072].")
        targets.append(target)
    if not targets:
        raise ValueError("At least one fixed target is required.")
    return targets


@dataclass
class FixedTargetResult:
    target: float
    success: bool
    settle_time_sec: float | None
    final_actual: float | None
    final_error: float | None
    samples: int


@dataclass
class TrackingReport:
    success: bool
    matched_samples: int
    rms_error: float
    max_error: float
    within_tolerance_ratio: float
    mean_lag_ms: float
    max_lag_ms: float


class _PositionController(Protocol):
    name: str
    dt: float

    def open(self) -> None: ...
    def close(self) -> None: ...
    def send_position(self, target: float) -> None: ...
    def read_position(self) -> float | None: ...
    def latest_debug(self) -> str | None: ...


class _DirectSDKG2Controller:
    """Vendor-like direct G2 control using ``airbot_hardware_py`` only."""

    name = "sdk_direct"

    def __init__(self, can_interface: str, control_frequency: int = 250) -> None:
        import airbot_hardware_py as ah

        self._ah = ah
        self._can_interface = can_interface
        self._control_frequency = control_frequency
        self.dt = 1.0 / max(control_frequency, 1)
        self._executor = None
        self._eef = None
        self._last_debug: str | None = None

    def open(self) -> None:
        from rollio.robot.airbot.shared import get_shared_airbot_runtime

        self._executor, io_context = get_shared_airbot_runtime(self._ah)
        self._eef = self._ah.EEF1.create(self._ah.EEFType.G2, self._ah.MotorType.DM)
        if not self._eef.init(io_context, self._can_interface, self._control_frequency):
            raise RuntimeError("The direct G2 initialization failed.")
        self._eef.enable()
        self._eef.set_param("control_mode", self._ah.MotorControlMode.PVT)

    def close(self) -> None:
        if self._eef is not None:
            try:
                self._eef.disable()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            try:
                self._eef.uninit()
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
        self._eef = None
        self._executor = None

    def send_position(self, target: float) -> None:
        if self._eef is None:
            raise RuntimeError("Direct G2 controller is not open.")
        cmd = self._ah.EEFCommand1()
        cmd.pos = [float(target)]
        cmd.vel = [200.0]
        cmd.current_threshold = [200.0]
        self._eef.pvt(cmd)
        self._last_debug = (
            f"PVT pos=[{target:7.4f}] vel=[200.0000] current_threshold=[200.0000]"
        )

    def read_position(self) -> float | None:
        if self._eef is None:
            return None
        state = self._eef.state()
        if not state.is_valid:
            return None
        return float(state.pos[0])

    def latest_debug(self) -> str | None:
        return self._last_debug


class _RollioG2Controller:
    """Rollio-wrapper G2 control using ``rollio.robot.AIRBOTG2``."""

    name = "rollio_wrapper"

    def __init__(self, can_interface: str, control_frequency: int = 250) -> None:
        self._can_interface = can_interface
        self._control_frequency = control_frequency
        self.dt = 1.0 / max(control_frequency, 1)
        self._robot = None

    def open(self) -> None:
        from rollio.robot import AIRBOTG2

        self._robot = AIRBOTG2(
            can_interface=self._can_interface,
            control_frequency=self._control_frequency,
            target_tracking_mode="pvt",
        )
        self._robot.open()
        if not self._robot.enable():
            raise RuntimeError("Failed to enable Rollio G2 controller.")
        if not self._robot.enter_target_tracking():
            raise RuntimeError("Failed to enter Rollio G2 target-tracking mode.")

    def close(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.disable()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            self._robot.close()
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        self._robot = None

    def send_position(self, target: float) -> None:
        if self._robot is None:
            raise RuntimeError("Rollio G2 controller is not open.")
        self._robot.step_target_tracking(
            position_target=np.array([target], dtype=np.float64),
            velocity_target=np.zeros(1, dtype=np.float64),
            add_gravity_compensation=False,
        )

    def read_position(self) -> float | None:
        if self._robot is None:
            return None
        state = self._robot.read_joint_state()
        if state.position is None:
            return None
        return float(state.position[0])

    def latest_debug(self) -> str | None:
        if self._robot is None:
            return None
        debug = self._robot.latest_command_debug()
        if debug is None:
            return None
        return f"{debug[0]} {debug[1]}"


def _print_experiment_header(name: str, *, verbose: bool) -> None:
    if not verbose:
        return
    print()
    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)


def _run_fixed_target(
    controller: _PositionController,
    target: float,
    *,
    tolerance: float,
    timeout: float,
    stable_samples: int,
    print_hz: float,
    verbose: bool,
) -> FixedTargetResult:
    stable = 0
    samples = 0
    final_actual: float | None = None
    final_error: float | None = None
    start = time.monotonic()
    last_print = 0.0
    print_period = 1.0 / max(print_hz, 1e-6)

    while True:
        now = time.monotonic()
        if now - start >= timeout:
            break

        controller.send_position(target)
        actual = controller.read_position()
        samples += 1

        if actual is not None:
            final_actual = actual
            final_error = actual - target
            if abs(final_error) <= tolerance:
                stable += 1
            else:
                stable = 0

        if verbose and (now - last_print >= print_period):
            debug_text = controller.latest_debug() or ""
            actual_text = "nan" if final_actual is None else f"{final_actual:7.4f}"
            error_text = "nan" if final_error is None else f"{final_error:+7.4f}"
            print(
                f"\r[{controller.name}] tgt={target:7.4f} act={actual_text} "
                f"err={error_text} stable={stable:02d}/{stable_samples:02d}  "
                f"{debug_text}",
                end="",
                flush=True,
            )
            last_print = now

        if stable >= stable_samples:
            if verbose:
                print()
            return FixedTargetResult(
                target=target,
                success=True,
                settle_time_sec=now - start,
                final_actual=final_actual,
                final_error=final_error,
                samples=samples,
            )

        time.sleep(controller.dt)

    if verbose:
        print()
    return FixedTargetResult(
        target=target,
        success=False,
        settle_time_sec=None,
        final_actual=final_actual,
        final_error=final_error,
        samples=samples,
    )


def _run_fixed_sequence(
    controller: _PositionController,
    *,
    targets: list[float],
    tolerance: float,
    timeout: float,
    stable_samples: int,
    print_hz: float,
    verbose: bool,
) -> bool:
    _print_experiment_header(
        f"{controller.name}: fixed-target convergence experiment",
        verbose=verbose,
    )
    all_ok = True
    for target in targets:
        result = _run_fixed_target(
            controller,
            target,
            tolerance=tolerance,
            timeout=timeout,
            stable_samples=stable_samples,
            print_hz=print_hz,
            verbose=verbose,
        )
        final_actual = (
            "nan" if result.final_actual is None else f"{result.final_actual:.4f}"
        )
        final_error = (
            "nan" if result.final_error is None else f"{result.final_error:+.4f}"
        )
        settle = (
            "timeout"
            if result.settle_time_sec is None
            else f"{result.settle_time_sec:.2f}s"
        )
        print(
            f"[{controller.name}] fixed target {target:.4f} -> "
            f"{'PASS' if result.success else 'FAIL'}  "
            f"settle={settle}  actual={final_actual}  error={final_error}"
        )
        all_ok &= result.success
        time.sleep(0.25)
    return all_ok


def _analyze_tracking(
    command_times: list[float],
    command_targets: list[float],
    actual_times: list[float],
    actual_positions: list[float],
    *,
    tolerance: float,
    max_lag_sec: float,
    min_match_ratio: float,
) -> TrackingReport:
    matches: list[tuple[float, float]] = []
    for actual_time, actual in zip(actual_times, actual_positions, strict=False):
        best_error: float | None = None
        best_lag: float | None = None
        for command_time, command_target in zip(
            command_times, command_targets, strict=False
        ):
            lag = actual_time - command_time
            if lag < 0.0:
                break
            if lag > max_lag_sec:
                continue
            error = abs(actual - command_target)
            if best_error is None or error < best_error:
                best_error = error
                best_lag = lag
        if best_error is not None and best_lag is not None:
            matches.append((best_error, best_lag))

    if not matches:
        return TrackingReport(
            success=False,
            matched_samples=0,
            rms_error=float("inf"),
            max_error=float("inf"),
            within_tolerance_ratio=0.0,
            mean_lag_ms=float("inf"),
            max_lag_ms=float("inf"),
        )

    errors = np.array([item[0] for item in matches], dtype=np.float64)
    lags = np.array([item[1] for item in matches], dtype=np.float64)
    within_ratio = float(np.mean(errors <= tolerance))
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    mean_lag_ms = float(np.mean(lags) * 1000.0)
    max_lag_ms = float(np.max(lags) * 1000.0)
    success = within_ratio >= min_match_ratio
    return TrackingReport(
        success=success,
        matched_samples=len(matches),
        rms_error=rms_error,
        max_error=max_error,
        within_tolerance_ratio=within_ratio,
        mean_lag_ms=mean_lag_ms,
        max_lag_ms=max_lag_ms,
    )


def _run_sine_tracking(
    controller: _PositionController,
    *,
    center: float,
    amplitude: float,
    period: float,
    duration: float,
    tolerance: float,
    max_lag_sec: float,
    min_match_ratio: float,
    print_hz: float,
    verbose: bool,
) -> TrackingReport:
    _print_experiment_header(
        f"{controller.name}: sine tracking experiment",
        verbose=verbose,
    )

    omega = 2.0 * math.pi / period
    start = time.monotonic()
    last_print = 0.0
    print_period = 1.0 / max(print_hz, 1e-6)
    command_times: list[float] = []
    command_targets: list[float] = []
    actual_times: list[float] = []
    actual_positions: list[float] = []
    last_actual: float | None = None

    while True:
        now = time.monotonic()
        t = now - start
        if t >= duration:
            break

        target = center + amplitude * math.sin(omega * t)
        controller.send_position(target)
        command_times.append(now)
        command_targets.append(target)

        actual = controller.read_position()
        if actual is not None:
            last_actual = actual
            actual_times.append(time.monotonic())
            actual_positions.append(actual)

        if verbose and (now - last_print >= print_period):
            actual_text = "nan" if last_actual is None else f"{last_actual:7.4f}"
            error_text = (
                "nan" if last_actual is None else f"{last_actual - target:+7.4f}"
            )
            debug_text = controller.latest_debug() or ""
            print(
                f"\r[{controller.name}] t={t:6.2f}s tgt={target:7.4f} "
                f"act={actual_text} err={error_text}  {debug_text}",
                end="",
                flush=True,
            )
            last_print = now

        time.sleep(controller.dt)

    if verbose:
        print()

    report = _analyze_tracking(
        command_times,
        command_targets,
        actual_times,
        actual_positions,
        tolerance=tolerance,
        max_lag_sec=max_lag_sec,
        min_match_ratio=min_match_ratio,
    )
    print(
        f"[{controller.name}] sine tracking -> "
        f"{'PASS' if report.success else 'FAIL'}  "
        f"matched={report.matched_samples}  "
        f"within_tol={report.within_tolerance_ratio * 100.0:5.1f}%  "
        f"rms_err={report.rms_error:.4f}  max_err={report.max_error:.4f}  "
        f"mean_lag={report.mean_lag_ms:6.1f}ms  max_lag={report.max_lag_ms:6.1f}ms"
    )
    return report


def _controller_factory(mode: str, can_interface: str):
    if mode.startswith("sdk_"):
        return _DirectSDKG2Controller(can_interface=can_interface)
    return _RollioG2Controller(can_interface=can_interface)


def _run_one_mode(
    mode: str,
    *,
    can_interface: str,
    center: float,
    amplitude: float,
    period: float,
    duration: float,
    fixed_targets: list[float],
    tolerance: float,
    max_lag_sec: float,
    min_match_ratio: float,
    stable_samples: int,
    timeout: float,
    print_hz: float,
    verbose: bool,
    return_to_zero: bool,
) -> bool:
    controller = _controller_factory(mode, can_interface)
    controller.open()
    success = False
    try:
        if mode.endswith("_fixed"):
            success = _run_fixed_sequence(
                controller,
                targets=fixed_targets,
                tolerance=tolerance,
                timeout=timeout,
                stable_samples=stable_samples,
                print_hz=print_hz,
                verbose=verbose,
            )
        elif mode.endswith("_sine"):
            if verbose:
                print(
                    f"[{controller.name}] centering at {center:.4f} before sine tracking"
                )
            _run_fixed_target(
                controller,
                center,
                tolerance=max(tolerance, 0.004),
                timeout=timeout,
                stable_samples=max(3, stable_samples),
                print_hz=print_hz,
                verbose=verbose,
            )
            time.sleep(0.2)
            report = _run_sine_tracking(
                controller,
                center=center,
                amplitude=amplitude,
                period=period,
                duration=duration,
                tolerance=tolerance,
                max_lag_sec=max_lag_sec,
                min_match_ratio=min_match_ratio,
                print_hz=print_hz,
                verbose=verbose,
            )
            success = report.success
        else:
            raise ValueError(f"Unsupported experiment mode: {mode}")

        if return_to_zero:
            _run_fixed_target(
                controller,
                0.0,
                tolerance=max(tolerance, 0.002),
                timeout=timeout,
                stable_samples=max(3, stable_samples // 2),
                print_hz=print_hz,
                verbose=False,
            )
        return success
    finally:
        controller.close()


def test_sine_position(
    can_interface: str = "can0",
    duration: float | None = None,
    verbose: bool = True,
    return_to_zero: bool = True,
    center: float = 0.035,
    amplitude: float = 0.020,
    period: float = 2.0,
    print_hz: float = 5.0,
) -> bool:
    """Run the Rollio-wrapper sine experiment and report lag-aware tracking."""
    from rollio.robot import is_airbot_available
    from rollio.robot.can_utils import is_can_interface_up

    if not is_airbot_available():
        print("Error: airbot_hardware_py is not installed.")
        return False

    if not is_can_interface_up(can_interface):
        print(f"Error: CAN interface '{can_interface}' is not available or not UP.")
        print(f"Run: sudo ip link set {can_interface} up type can bitrate 1000000")
        return False

    if period <= 0.0:
        print("Error: period must be > 0.")
        return False

    if print_hz <= 0.0:
        print("Error: print_hz must be > 0.")
        return False

    try:
        center, amplitude = _validate_range(center, amplitude)
    except ValueError as exc:
        print(f"Error: {exc}")
        return False

    fixed_targets = [
        max(0.0, center - amplitude),
        center,
        min(0.072, center + amplitude),
    ]
    if duration is None:
        duration = max(6.0, 2.0 * period)

    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    try:
        return _run_one_mode(
            "wrapper_sine",
            can_interface=can_interface,
            center=center,
            amplitude=amplitude,
            period=period,
            duration=duration,
            fixed_targets=fixed_targets,
            tolerance=0.006,
            max_lag_sec=0.25,
            min_match_ratio=0.85,
            stable_samples=6,
            timeout=5.0,
            print_hz=print_hz,
            verbose=verbose,
            return_to_zero=return_to_zero,
        )
    except KeyboardInterrupt:
        if verbose:
            print("\n\nStopped by user.")
        return True
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()
        return False
    finally:
        signal.signal(signal.SIGINT, original_handler)


def compare_modes(
    *,
    can_interface: str,
    center: float,
    amplitude: float,
    period: float,
    duration: float,
    fixed_targets: list[float],
    tolerance: float,
    max_lag_sec: float,
    min_match_ratio: float,
    stable_samples: int,
    timeout: float,
    print_hz: float,
    verbose: bool,
    return_to_zero: bool,
) -> bool:
    """Run direct-SDK and Rollio wrapper experiments back to back."""
    modes = ("sdk_fixed", "sdk_sine", "wrapper_fixed", "wrapper_sine")
    results: dict[str, bool] = {}
    for mode in modes:
        results[mode] = _run_one_mode(
            mode,
            can_interface=can_interface,
            center=center,
            amplitude=amplitude,
            period=period,
            duration=duration,
            fixed_targets=fixed_targets,
            tolerance=tolerance,
            max_lag_sec=max_lag_sec,
            min_match_ratio=min_match_ratio,
            stable_samples=stable_samples,
            timeout=timeout,
            print_hz=print_hz,
            verbose=verbose,
            return_to_zero=return_to_zero,
        )
        time.sleep(0.5)

    print()
    print("=" * 72)
    print("  G2 experiment summary")
    print("=" * 72)
    for mode, success in results.items():
        print(f"  {mode:<14} {'PASS' if success else 'FAIL'}")
    return all(results.values())


TESTS = {
    "airbot_g2_sine_position": test_sine_position,
}
TEST_DESCRIPTIONS = {
    "airbot_g2_sine_position": "Run AIRBOT G2 sine target-position test using AIRBOTG2",
}


def run_test(test_name: str, **kwargs) -> bool:
    """Run a named AIRBOT G2 hardware helper."""
    if test_name not in TESTS:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(TESTS.keys())}")
        return False
    return TESTS[test_name](**kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AIRBOT G2 hardware experiments. "
            "Compare direct SDK control with the Rollio wrapper and measure "
            "actual-vs-command tracking with allowed lag."
        )
    )
    parser.add_argument("-i", "--can", default="can0", help="CAN interface name")
    parser.add_argument(
        "--mode",
        choices=(
            "compare",
            "sdk_fixed",
            "sdk_sine",
            "wrapper_fixed",
            "wrapper_sine",
        ),
        default="wrapper_sine",
        help="Experiment mode to run",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=6.0,
        help="Sine experiment duration in seconds",
    )
    parser.add_argument(
        "--center",
        type=float,
        default=0.035,
        help="Center position in meters (default: 0.035)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.020,
        help="Sine amplitude in meters (default: 0.020)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=2.0,
        help="Sine period in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--fixed-targets",
        default="0.015,0.035,0.055",
        help="Comma-separated fixed targets for convergence experiments",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.006,
        help="Tracking tolerance in meters",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=250.0,
        help="Allowed command-following lag in milliseconds",
    )
    parser.add_argument(
        "--pass-ratio",
        type=float,
        default=0.85,
        help="Minimum fraction of matched samples that must be within tolerance",
    )
    parser.add_argument(
        "--stable-samples",
        type=int,
        default=6,
        help="Consecutive in-tolerance samples required for fixed-target success",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout per fixed target in seconds",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=5.0,
        help="Console status rate in Hz",
    )
    parser.add_argument(
        "--no-return-zero",
        action="store_true",
        help="Do not return the gripper to zero before closing",
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
        center, amplitude = _validate_range(args.center, args.amplitude)
        fixed_targets = _parse_targets(args.fixed_targets)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    try:
        common_kwargs = dict(
            can_interface=args.can,
            center=center,
            amplitude=amplitude,
            period=args.period,
            duration=args.duration,
            fixed_targets=fixed_targets,
            tolerance=args.tolerance,
            max_lag_sec=args.max_lag_ms / 1000.0,
            min_match_ratio=args.pass_ratio,
            stable_samples=max(1, args.stable_samples),
            timeout=args.timeout,
            print_hz=args.print_hz,
            verbose=not args.quiet,
            return_to_zero=not args.no_return_zero,
        )
        if args.mode == "compare":
            success = compare_modes(**common_kwargs)
        else:
            success = _run_one_mode(args.mode, **common_kwargs)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        return 130
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()
        return 1
    finally:
        signal.signal(signal.SIGINT, original_handler)


if __name__ == "__main__":
    raise SystemExit(main())
