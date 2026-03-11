"""AIRBOT Play hardware tests.

Tests for verifying AIRBOT Play robot arm functionality.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rollio.robot import AIRBOTPlay


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted by user. Stopping...")
    sys.exit(130)


def test_gravity_compensation(
    can_interface: str = "can0",
    duration: float | None = None,
    verbose: bool = True,
    return_to_zero: bool = True,
) -> bool:
    """Test AIRBOT Play gravity compensation (free drive mode).

    This test puts the robot in free drive mode with gravity compensation,
    allowing manual guidance of the arm. The robot should feel "weightless"
    and stay in place when released.

    Args:
        can_interface: CAN interface name (e.g., "can0")
        duration: Test duration in seconds (None for indefinite, Ctrl+C to stop)
        verbose: Print status messages
        return_to_zero: If True, return arm to zero position when test ends

    Returns:
        True if test completed successfully
    """
    from rollio.robot import AIRBOTPlay, ControlMode, is_airbot_available
    from rollio.robot.can_utils import (
        is_can_interface_up,
        probe_airbot_device,
        scan_can_interfaces,
    )

    # Check dependencies
    if not is_airbot_available():
        print("Error: airbot_hardware_py is not installed.")
        print("Install with: pip install airbot_hardware_py")
        return False

    # First, scan and print available AIRBOT arms
    if verbose:
        print("Scanning for available AIRBOT Play arms...")
        print()

        can_interfaces = scan_can_interfaces()
        if not can_interfaces:
            print("  No CAN interfaces found.")
        else:
            found_any = False
            for iface in can_interfaces:
                is_up = is_can_interface_up(iface)
                status = "UP" if is_up else "DOWN"

                if is_up:
                    is_airbot = probe_airbot_device(iface, timeout=0.5)
                    if is_airbot:
                        print(f"  {iface}: [{status}] AIRBOT Play detected ✓")
                        found_any = True
                    else:
                        print(f"  {iface}: [{status}] No AIRBOT device")
                else:
                    print(f"  {iface}: [{status}]")

            if not found_any:
                print("\n  No AIRBOT Play devices detected on any interface.")

        print()

    # Check CAN interface
    if not is_can_interface_up(can_interface):
        print(f"Error: CAN interface '{can_interface}' is not available or not UP.")
        print(f"Run: sudo ip link set {can_interface} up type can bitrate 1000000")
        return False

    # Set up signal handler for graceful shutdown
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    robot: AIRBOTPlay | None = None

    try:
        if verbose:
            print(f"Connecting to AIRBOT Play on {can_interface}...")

        robot = AIRBOTPlay(can_interface=can_interface)
        robot.open()

        if verbose:
            print("Enabling motors...")

        robot.enable()

        if verbose:
            print("Entering free drive mode with gravity compensation...")
            print("\n" + "=" * 60)
            print("  GRAVITY COMPENSATION TEST")
            print("=" * 60)
            print("\nThe robot should now feel 'weightless'.")
            print(
                "You can manually guide the arm - it should stay in place when released."
            )
            print("\nPress Ctrl+C to stop the test.\n")

        # Enter free drive mode
        if not robot.enter_free_drive():
            print("Error: Failed to enter free drive mode.")
            return False

        start_time = time.monotonic()
        iteration = 0

        # Main control loop
        while True:
            # Check duration
            elapsed = time.monotonic() - start_time
            if duration is not None and elapsed >= duration:
                if verbose:
                    print(f"\nTest duration ({duration}s) completed.")
                break

            # Read current state
            joint_state = robot.read_joint_state()

            if not joint_state.is_valid:
                print("Warning: Invalid joint state received")
                time.sleep(0.01)
                continue

            # Send free drive command (gravity compensation only)
            robot.step_free_drive()

            # Print status periodically
            if verbose and iteration % 250 == 0:  # ~1Hz at 250Hz control rate
                pos = joint_state.position
                if pos is not None:
                    pos_str = " ".join(f"{p:+6.2f}" for p in pos)
                    print(
                        f"\rJoints: [{pos_str}]  Time: {elapsed:.1f}s",
                        end="",
                        flush=True,
                    )

            iteration += 1

            # Control loop timing (~250Hz)
            time.sleep(0.004)

        if verbose:
            print("\n\nTest completed successfully!")

        return True

    except KeyboardInterrupt:
        if verbose:
            print("\n\nTest stopped by user.")
        return True

    except Exception as e:
        print(f"\nError during test: {e}")
        return False

    finally:
        # Restore signal handler
        signal.signal(signal.SIGINT, original_handler)

        # Clean up robot
        if robot is not None:
            try:
                # Return to zero position if requested
                if return_to_zero and robot._is_enabled:
                    if verbose:
                        print("\nReturning to zero position...")

                    success = robot.move_to_home(timeout=15.0)
                    if verbose:
                        if success:
                            print("Returned to zero position.")
                        else:
                            print(
                                "Warning: Could not reach zero position within timeout."
                            )

                if verbose:
                    print("Disabling motors and closing connection...")

                robot.disable()
                robot.close()
            except Exception:
                pass


def test_identify(
    can_interface: str = "can0",
    duration: float = 5.0,
    verbose: bool = True,
) -> bool:
    """Test AIRBOT Play LED identification.

    Blinks the robot's LED orange for identification.

    Args:
        can_interface: CAN interface name
        duration: How long to blink (seconds)
        verbose: Print status messages

    Returns:
        True if test completed successfully
    """
    from rollio.robot.can_utils import (
        is_can_interface_up,
        probe_airbot_device,
        scan_can_interfaces,
        set_airbot_led,
    )

    # First, scan and print available AIRBOT arms
    if verbose:
        print("Scanning for available AIRBOT Play arms...")
        print()

        can_interfaces = scan_can_interfaces()
        if not can_interfaces:
            print("  No CAN interfaces found.")
        else:
            found_any = False
            for iface in can_interfaces:
                is_up = is_can_interface_up(iface)
                status = "UP" if is_up else "DOWN"

                if is_up:
                    # Probe for AIRBOT device
                    is_airbot = probe_airbot_device(iface, timeout=0.5)
                    if is_airbot:
                        print(f"  {iface}: [{status}] AIRBOT Play detected ✓")
                        found_any = True
                    else:
                        print(f"  {iface}: [{status}] No AIRBOT device")
                else:
                    print(f"  {iface}: [{status}]")

            if not found_any:
                print("\n  No AIRBOT Play devices detected on any interface.")

        print()

    # Check if specified interface is available
    if not is_can_interface_up(can_interface):
        print(f"Error: CAN interface '{can_interface}' is not available or not UP.")
        return False

    try:
        if verbose:
            print(f"Blinking LED on {can_interface} for {duration}s...")

        # Start blinking
        if not set_airbot_led(can_interface, blink_orange=True):
            print("Error: Failed to start LED blinking.")
            return False

        time.sleep(duration)

        # Stop blinking
        if not set_airbot_led(can_interface, blink_orange=False):
            print("Warning: Failed to stop LED blinking.")

        if verbose:
            print("LED test completed.")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_sine_swing(
    can_interface: str = "can0",
    duration: float | None = None,
    verbose: bool = True,
    return_to_zero: bool = True,
    joint: int = 0,
    amplitude: float | None = None,
    period: float = 1.0,
    kp: float = 15.0,
    kd: float = 3.0,
) -> bool:
    """Test AIRBOT Play target tracking with a sine-wave sweep on one joint.

    1. Move the arm to the zero position
    2. Swing the specified joint with a sine trajectory while all other
       joints hold at zero.  Gravity compensation is automatically
       included via ``step_target_tracking(add_gravity_compensation=True)``.

    Args:
        can_interface: CAN interface name (e.g., "can0")
        duration: Swing duration in seconds (None → indefinite, Ctrl+C to stop)
        verbose: Print status messages
        return_to_zero: Return to zero position when finished
        joint: Joint index to swing (0-based, default 0 = shoulder yaw)
        amplitude: Swing amplitude in radians (default π/6)
        period: Sine wave period in seconds (default 1.0)
        kp: Position gain
        kd: Velocity gain

    Returns:
        True if test completed successfully
    """
    import math

    from rollio.robot import AIRBOTPlay, is_airbot_available
    from rollio.robot.can_utils import is_can_interface_up

    if amplitude is None:
        amplitude = math.pi / 6

    if not is_airbot_available():
        print("Error: airbot_hardware_py is not installed.")
        return False

    if not is_can_interface_up(can_interface):
        print(f"Error: CAN interface '{can_interface}' is not available or not UP.")
        return False

    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    robot: AIRBOTPlay | None = None

    try:
        if verbose:
            print(f"Connecting to AIRBOT Play on {can_interface}…")

        robot = AIRBOTPlay(can_interface=can_interface)
        robot.open()

        if verbose:
            print("Enabling motors…")
        robot.enable()

        n = robot.N_DOF
        if joint < 0 or joint >= n:
            print(f"Error: joint index {joint} out of range [0, {n-1}]")
            return False

        # ── Phase 1: move to zero ────────────────────────────────
        if verbose:
            print("Moving to zero position…")
        if not robot.move_to_home(timeout=15.0):
            print(
                "Warning: could not reach zero position within timeout, "
                "continuing anyway."
            )
        time.sleep(0.3)

        # ── Phase 2: sine swing ──────────────────────────────────
        if verbose:
            print()
            print("=" * 60)
            print(f"  SINE SWING TEST  (joint {joint})")
            print(
                f"  amplitude={amplitude:.3f} rad  "
                f"period={period:.2f}s  kp={kp}  kd={kd}"
            )
            print("=" * 60)
            print("\nPress Ctrl+C to stop.\n")

        if not robot.enter_target_tracking():
            print("Error: failed to enter target tracking mode.")
            return False

        kp_arr = np.array([200, 200, 200, 50, 50, 50])
        kd_arr = np.array([5, 5, 5, 1, 1, 1])
        start = time.monotonic()
        iteration = 0

        while True:
            t = time.monotonic() - start
            if duration is not None and t >= duration:
                if verbose:
                    print(f"\nDuration ({duration}s) reached.")
                break

            # Target: all joints at 0 except the swing joint
            q_target = np.zeros(n)
            qd_target = np.zeros(n)
            omega = 2.0 * math.pi / period
            q_target[joint] = amplitude * math.sin(omega * t)
            qd_target[joint] = amplitude * omega * math.cos(omega * t)

            robot.step_target_tracking(
                position_target=q_target,
                velocity_target=qd_target,
                kp=kp_arr,
                kd=kd_arr,
                add_gravity_compensation=True,
            )

            # Print status ~2 Hz
            if verbose and iteration % 125 == 0:
                js = robot.read_joint_state()
                if js.position is not None:
                    actual = js.position[joint]
                    target = q_target[joint]
                    err = actual - target
                    pos_str = " ".join(f"{p:+6.2f}" for p in js.position)
                    print(
                        f"\r  t={t:5.1f}s  "
                        f"tgt={target:+6.3f}  "
                        f"act={actual:+6.3f}  "
                        f"err={err:+6.3f}  "
                        f"[{pos_str}]",
                        end="",
                        flush=True,
                    )

            iteration += 1
            time.sleep(robot._dt)

        if verbose:
            print("\n\nSine swing test completed!")
        return True

    except KeyboardInterrupt:
        if verbose:
            print("\n\nStopped by user.")
        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        signal.signal(signal.SIGINT, original_handler)
        if robot is not None:
            try:
                if return_to_zero and robot._is_enabled:
                    if verbose:
                        print("\nReturning to zero position…")
                    robot.move_to_home(timeout=15.0)
                robot.disable()
                robot.close()
            except Exception:
                pass


# Register tests
TESTS = {
    "airbot_play_gravity_compensation": test_gravity_compensation,
    "airbot_play_identify": test_identify,
    "airbot_play_sine_swing": test_sine_swing,
}


def run_test(test_name: str, **kwargs) -> bool:
    """Run a named test.

    Args:
        test_name: Name of the test to run
        **kwargs: Arguments to pass to the test function

    Returns:
        True if test passed
    """
    if test_name not in TESTS:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(TESTS.keys())}")
        return False

    return TESTS[test_name](**kwargs)
