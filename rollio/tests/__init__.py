"""Rollio hardware tests.

This module provides hardware test functions that can be run via
the `rollio test` command.
"""
from __future__ import annotations

# Registry of available tests with descriptions
AVAILABLE_TESTS: dict[str, str] = {
    "airbot_play_gravity_compensation": "Test AIRBOT Play gravity compensation (free drive mode)",
    "airbot_play_identify": "Test AIRBOT Play LED identification (blink orange)",
    "airbot_play_sine_swing": "Move to zero then swing one joint with a sine wave",
    "airbot_g2_sine_position": "Run AIRBOT G2 sine target-position test using AIRBOTG2",
}


def get_available_tests() -> list[str]:
    """Return list of available test names."""
    return list(AVAILABLE_TESTS.keys())


def get_test_description(test_name: str) -> str | None:
    """Return description for a test."""
    return AVAILABLE_TESTS.get(test_name)


def run_test(test_name: str, **kwargs) -> bool:
    """Run a named test.
    
    Args:
        test_name: Name of the test to run
        **kwargs: Arguments to pass to the test function
        
    Returns:
        True if test passed, False otherwise
    """
    if test_name.startswith("airbot_play_"):
        from rollio.tests.airbot_play import run_test as run_airbot_test
        return run_airbot_test(test_name, **kwargs)
    if test_name.startswith("airbot_g2_"):
        from rollio.tests.airbot_g2 import run_test as run_airbot_test
        return run_airbot_test(test_name, **kwargs)
    
    print(f"Unknown test: {test_name}")
    print(f"Available tests: {', '.join(get_available_tests())}")
    return False
