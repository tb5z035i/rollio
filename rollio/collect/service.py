"""Reusable worker-backed runtime service layer."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Protocol

from rollio.config.schema import RollioConfig

from .runtime import (
    AsyncCollectionRuntime,
    RecordedEpisodeSummary,
    RuntimeSnapshot,
    summarize_recorded_episode,
)


class CollectionRuntimeService(Protocol):
    """Command surface plus batched snapshot reads."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def snapshot(
        self,
        *,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> RuntimeSnapshot: ...

    def start_episode(self) -> int: ...

    def stop_episode(self) -> RecordedEpisodeSummary: ...

    def keep_episode(self) -> int: ...

    def discard_episode(self) -> None: ...

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]: ...

    def wait_for_exports(self) -> None: ...

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool: ...


@dataclass(frozen=True)
class _WorkerLaunchConfig:
    cfg_data: dict[str, Any]
    export_delay_sec: float
    scheduler_driver: str
    preview_live_feedback: bool
    bootstrap_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class _WorkerRequest:
    command: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _WorkerResponse:
    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None


def _run_bootstrap_entries(entries: tuple[str, ...]) -> None:
    for raw_entry in entries:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        module_name, sep, callable_path = entry.partition(":")
        module = importlib.import_module(module_name)
        if not sep:
            continue
        target: Any = module
        for attr in callable_path.split("."):
            target = getattr(target, attr)
        if not callable(target):
            raise TypeError(f"Worker bootstrap target '{entry}' is not callable.")
        target()


def _handle_worker_request(
    runtime: AsyncCollectionRuntime,
    request: _WorkerRequest,
) -> Any:
    command = request.command
    payload = dict(request.payload)

    if command == "snapshot":
        return runtime.snapshot(
            max_frame_width=payload.get("max_frame_width"),
            max_frame_height=payload.get("max_frame_height"),
        )
    if command == "start_episode":
        return runtime.start_episode()
    if command == "stop_episode":
        return summarize_recorded_episode(runtime.stop_episode())
    if command == "keep_episode":
        return runtime.keep_episode().episode_index
    if command == "discard_episode":
        runtime.discard_episode()
        return None
    if command == "return_robots_to_zero":
        timeout = float(payload.get("timeout", 10.0))
        return runtime.return_robots_to_zero(timeout=timeout)
    if command == "wait_for_exports":
        runtime.wait_for_exports()
        return None
    if command == "wait_for_episode_export":
        return runtime.wait_for_episode_export(
            int(payload["episode_index"]),
            payload.get("timeout"),
        )
    if command == "close":
        return None
    raise ValueError(f"Unsupported worker command: {command}")


def _runtime_worker_main(conn: Connection, launch_config: _WorkerLaunchConfig) -> None:
    runtime: AsyncCollectionRuntime | None = None
    try:
        _run_bootstrap_entries(launch_config.bootstrap_entries)
        cfg = RollioConfig.model_validate(launch_config.cfg_data)
        runtime = AsyncCollectionRuntime.from_config(
            cfg,
            export_delay_sec=launch_config.export_delay_sec,
            scheduler_driver=launch_config.scheduler_driver,
            preview_live_feedback=launch_config.preview_live_feedback,
        )
        runtime.open()
        conn.send(_WorkerResponse(ok=True, result="ready"))
        while True:
            if not conn.poll(0.05):
                continue
            try:
                request = conn.recv()
            except EOFError:
                break
            if not isinstance(request, _WorkerRequest):
                conn.send(
                    _WorkerResponse(
                        ok=False,
                        error_type="TypeError",
                        error_message="Unexpected worker request payload.",
                    )
                )
                continue
            try:
                conn.send(
                    _WorkerResponse(
                        ok=True, result=_handle_worker_request(runtime, request)
                    )
                )
                if request.command == "close":
                    break
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                conn.send(
                    _WorkerResponse(
                        ok=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                )
                if request.command == "close":
                    break
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        try:
            conn.send(
                _WorkerResponse(
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if runtime is not None:
            runtime.close()
        try:
            conn.close()
        except OSError:
            pass


class WorkerCollectionRuntimeService:
    """Proxy that owns one isolated runtime worker process."""

    def __init__(
        self,
        cfg: RollioConfig,
        *,
        export_delay_sec: float = 0.0,
        scheduler_driver: str = "asyncio",
        preview_live_feedback: bool = False,
        bootstrap_entries: tuple[str, ...] = (),
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._launch_config = _WorkerLaunchConfig(
            cfg_data=cfg.model_dump(),
            export_delay_sec=float(export_delay_sec),
            scheduler_driver=str(scheduler_driver),
            preview_live_feedback=bool(preview_live_feedback),
            bootstrap_entries=tuple(str(entry) for entry in bootstrap_entries),
        )
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._opened = False
        self._lock = threading.Lock()

    def open(self) -> None:
        if self._opened:
            return
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_runtime_worker_main,
            args=(child_conn, self._launch_config),
            name="rollio-runtime-worker",
        )
        process.start()
        child_conn.close()
        self._conn = parent_conn
        self._process = process
        try:
            response = self._receive_response(timeout=60.0, context="worker startup")
        except (RuntimeError, TimeoutError, TypeError):
            self._cleanup_process()
            raise
        if response.result != "ready":
            self._cleanup_process()
            raise RuntimeError("Runtime worker did not report a ready state.")
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            self._cleanup_process()
            return
        try:
            self._request("close", response_timeout=15.0)
        except (BrokenPipeError, EOFError, OSError, RuntimeError, TimeoutError):
            pass
        self._cleanup_process()

    def snapshot(
        self,
        *,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> RuntimeSnapshot:
        result = self._request(
            "snapshot",
            max_frame_width=max_frame_width,
            max_frame_height=max_frame_height,
        )
        if not isinstance(result, RuntimeSnapshot):
            raise TypeError("Runtime worker returned an invalid snapshot payload.")
        return result

    def start_episode(self) -> int:
        return int(self._request("start_episode"))

    def stop_episode(self) -> RecordedEpisodeSummary:
        result = self._request("stop_episode")
        if not isinstance(result, RecordedEpisodeSummary):
            raise TypeError("Runtime worker returned an invalid episode summary.")
        return result

    def keep_episode(self) -> int:
        return int(self._request("keep_episode"))

    def discard_episode(self) -> None:
        self._request("discard_episode")

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        result = self._request("return_robots_to_zero", timeout=float(timeout))
        if not isinstance(result, dict):
            raise TypeError("Runtime worker returned an invalid zeroing result.")
        return result

    def wait_for_exports(self) -> None:
        self._request("wait_for_exports", response_timeout=None)

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool:
        result = self._request(
            "wait_for_episode_export",
            response_timeout=(
                None if timeout is None else max(float(timeout) + 5.0, 5.0)
            ),
            episode_index=int(episode_index),
            timeout=timeout,
        )
        return bool(result)

    def _request(
        self,
        command: str,
        response_timeout: float | None = 30.0,
        **payload: Any,
    ) -> Any:
        if not self._opened or self._conn is None:
            raise RuntimeError("Runtime worker is not open.")
        with self._lock:
            self._ensure_process_alive()
            self._conn.send(_WorkerRequest(command=command, payload=dict(payload)))
            response = self._receive_response(
                timeout=response_timeout,
                context=command,
            )
        return response.result

    def _receive_response(
        self,
        *,
        timeout: float | None,
        context: str,
    ) -> _WorkerResponse:
        if self._conn is None:
            raise RuntimeError("Runtime worker connection is not available.")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            poll_timeout = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out while waiting for runtime {context}."
                    )
                poll_timeout = min(poll_timeout, remaining)
            if self._conn.poll(poll_timeout):
                response = self._conn.recv()
                if not isinstance(response, _WorkerResponse):
                    raise TypeError(
                        "Runtime worker returned an invalid response payload."
                    )
                if response.ok:
                    return response
                error_suffix = (
                    f"\n{response.traceback_text}" if response.traceback_text else ""
                )
                raise RuntimeError(
                    f"Runtime worker {response.error_type or 'error'} during "
                    f"{context}: {response.error_message}{error_suffix}"
                )
            self._ensure_process_alive()

    def _ensure_process_alive(self) -> None:
        process = self._process
        if process is not None and not process.is_alive():
            raise RuntimeError(
                f"Runtime worker exited unexpectedly with code {process.exitcode}."
            )

    def _cleanup_process(self) -> None:
        self._opened = False
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            self._process = None


def create_runtime_service(
    cfg: RollioConfig,
    *,
    export_delay_sec: float = 0.0,
    scheduler_driver: str = "asyncio",
    preview_live_feedback: bool = False,
    bootstrap_entries: tuple[str, ...] = (),
) -> CollectionRuntimeService:
    """Build one worker-backed runtime service."""

    resolved_bootstrap_entries = (
        tuple(str(entry) for entry in bootstrap_entries)
        if bootstrap_entries
        else tuple(str(entry) for entry in cfg.async_pipeline.worker_bootstrap)
    )
    return WorkerCollectionRuntimeService(
        cfg,
        export_delay_sec=export_delay_sec,
        scheduler_driver=scheduler_driver,
        preview_live_feedback=preview_live_feedback,
        bootstrap_entries=resolved_bootstrap_entries,
    )
