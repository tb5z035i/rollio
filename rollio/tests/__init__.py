"""Packaged hardware/manual helpers used by ``rollio test``."""

from __future__ import annotations

from importlib import import_module

_TEST_MODULES = (
    "rollio.tests.airbot_play",
    "rollio.tests.airbot_g2",
)


def _load_test_registry() -> tuple[dict[str, str], dict[str, object]]:
    descriptions: dict[str, str] = {}
    runners: dict[str, object] = {}
    for module_name in _TEST_MODULES:
        module = import_module(module_name)
        descriptions.update(getattr(module, "TEST_DESCRIPTIONS", {}))
        runners.update(getattr(module, "TESTS", {}))
    return descriptions, runners


def get_available_tests() -> list[str]:
    """Return the hardware helper names exposed via ``rollio test``."""

    descriptions, _ = _load_test_registry()
    return list(descriptions.keys())


def get_test_description(test_name: str) -> str | None:
    """Return the short human description for one hardware helper."""

    descriptions, _ = _load_test_registry()
    return descriptions.get(test_name)


def run_test(test_name: str, **kwargs) -> bool:
    """Run one packaged hardware helper by name."""

    descriptions, runners = _load_test_registry()
    runner = runners.get(test_name)
    if runner is None:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(descriptions)}")
        return False
    return bool(runner(**kwargs))
