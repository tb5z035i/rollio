"""Shared fake terminal/stdout helpers for TUI tests."""

from __future__ import annotations

import io


class CollectionFakeTerm:
    cols = 80
    rows = 24

    def __init__(self, keys: list[str]) -> None:
        self._keys = iter(keys)

    def __enter__(self) -> "CollectionFakeTerm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def key(self) -> str | None:
        return next(self._keys, None)


class WizardFakeTerm:
    cols = 80
    rows = 24

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = iter(keys or ["x"])

    def __enter__(self) -> "WizardFakeTerm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read_key_blocking(self, timeout: float = 0.05) -> str | None:
        del timeout
        return next(self._keys, "x")

    def read_key(self) -> str | None:
        return next(self._keys, None)


class FakeStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None
