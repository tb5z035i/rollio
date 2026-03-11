"""Reusable runtime service layer for UI and non-UI clients."""

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
    RecordedEpisode,
    RecordedEpisodeSummary,
    RuntimeSnapshot,
    summarize_recorded_episode,
)


@dataclass(frozen=True)
class EpisodeExportHandle:
    """Serializable handle for one submitted episode export."""

    episode_index: int


class CollectionRuntimeService(Protocol):
    """Common runtime surface shared by in-process and worker clients."""

    video_codec: str
    depth_codec: str
    scheduler_driver: str
    telemetry_hz: int
    control_hz: int

    @property
    def recording(self) -> bool: ...

    @property
    def elapsed(self) -> float: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def snapshot(self) -> RuntimeSnapshot: ...

    def start_episode(self) -> int: ...

    def stop_episode(self) -> RecordedEpisodeSummary: ...

    def keep_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> EpisodeExportHandle: ...

    def discard_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> None: ...

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]: ...

    def export_status(self) -> tuple[int, int]: ...

    def wait_for_exports(self) -> None: ...

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool: ...

    def latest_frames(self) -> dict[str, object]: ...

    def latest_robot_states(self) -> dict[str, dict[str, object]]: ...

    def latest_pair_modes(self) -> dict[str, str]: ...

    def action_layout(self) -> list[dict[str, int | str]]: ...

    def scheduler_metrics(self) -> dict[str, object]: ...

    def timing_diagnostics(self) -> object: ...


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


def _episode_index_from_argument(
    episode: RecordedEpisodeSummary | RecordedEpisode | int | None,
) -> int | None:
    if episode is None:
        return None
    if isinstance(episode, int):
        return int(episode)
    episode_index = getattr(episode, "episode_index", None)
    if episode_index is None:
        raise TypeError("Episode handle must expose an episode_index.")
    return int(episode_index)


def _validate_pending_episode(
    pending_episode: RecordedEpisodeSummary | None,
    episode: RecordedEpisodeSummary | RecordedEpisode | int | None,
) -> int | None:
    requested_index = _episode_index_from_argument(episode)
    pending_index = pending_episode.episode_index if pending_episode is not None else None
    if requested_index is not None and pending_index != requested_index:
        raise RuntimeError(
            f"Pending episode mismatch: expected {pending_index}, got {requested_index}."
        )
    return pending_index


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
    *,
    pending_episode: RecordedEpisodeSummary | None,
) -> tuple[Any, RecordedEpisodeSummary | None]:
    command = request.command
    payload = dict(request.payload)

    if command == "snapshot":
        return runtime.snapshot(), pending_episode
    if command == "start_episode":
        return runtime.start_episode(), None
    if command == "stop_episode":
        episode = runtime.stop_episode()
        summary = summarize_recorded_episode(episode)
        return summary, summary
    if command == "keep_episode":
        _validate_pending_episode(pending_episode, payload.get("episode_index"))
        record = runtime.keep_episode()
        return EpisodeExportHandle(episode_index=record.episode_index), None
    if command == "discard_episode":
        _validate_pending_episode(pending_episode, payload.get("episode_index"))
        runtime.discard_episode()
        return None, None
    if command == "return_robots_to_zero":
        timeout = float(payload.get("timeout", 10.0))
        return runtime.return_robots_to_zero(timeout=timeout), pending_episode
    if command == "wait_for_exports":
        runtime.wait_for_exports()
        return None, pending_episode
    if command == "wait_for_episode_export":
        return (
            runtime.wait_for_episode_export(
                int(payload["episode_index"]),
                payload.get("timeout"),
            ),
            pending_episode,
        )
    if command == "close":
        return None, pending_episode
    raise ValueError(f"Unsupported worker command: {command}")


def _runtime_worker_main(conn: Connection, launch_config: _WorkerLaunchConfig) -> None:
    runtime: AsyncCollectionRuntime | None = None
    pending_episode: RecordedEpisodeSummary | None = None
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
                result, pending_episode = _handle_worker_request(
                    runtime,
                    request,
                    pending_episode=pending_episode,
                )
                conn.send(_WorkerResponse(ok=True, result=result))
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


class InProcessCollectionRuntimeService:
    """Adapter that exposes one common service surface in-process."""

    def __init__(self, runtime: AsyncCollectionRuntime) -> None:
        self._runtime = runtime
        self._pending_episode: RecordedEpisodeSummary | None = None

    @property
    def video_codec(self) -> str:
        return self._runtime.video_codec

    @property
    def depth_codec(self) -> str:
        return self._runtime.depth_codec

    @property
    def scheduler_driver(self) -> str:
        return self._runtime.scheduler_driver

    @property
    def telemetry_hz(self) -> int:
        return self._runtime.telemetry_hz

    @property
    def control_hz(self) -> int:
        return self._runtime.control_hz

    @property
    def recording(self) -> bool:
        return self._runtime.recording

    @property
    def elapsed(self) -> float:
        return self._runtime.elapsed

    def open(self) -> None:
        self._runtime.open()

    def close(self) -> None:
        self._runtime.close()

    def snapshot(self) -> RuntimeSnapshot:
        return self._runtime.snapshot()

    def start_episode(self) -> int:
        self._pending_episode = None
        return self._runtime.start_episode()

    def stop_episode(self) -> RecordedEpisodeSummary:
        summary = summarize_recorded_episode(self._runtime.stop_episode())
        self._pending_episode = summary
        return summary

    def keep_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> EpisodeExportHandle:
        _validate_pending_episode(self._pending_episode, episode)
        record = self._runtime.keep_episode()
        self._pending_episode = None
        return EpisodeExportHandle(episode_index=record.episode_index)

    def discard_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> None:
        _validate_pending_episode(self._pending_episode, episode)
        self._runtime.discard_episode()
        self._pending_episode = None

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        return self._runtime.return_robots_to_zero(timeout=timeout)

    def export_status(self) -> tuple[int, int]:
        return self._runtime.export_status()

    def wait_for_exports(self) -> None:
        self._runtime.wait_for_exports()

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool:
        return self._runtime.wait_for_episode_export(episode_index, timeout)

    def latest_frames(self) -> dict[str, object]:
        return self._runtime.latest_frames()

    def latest_robot_states(self) -> dict[str, dict[str, object]]:
        return self._runtime.latest_robot_states()

    def latest_pair_modes(self) -> dict[str, str]:
        return self._runtime.latest_pair_modes()

    def action_layout(self) -> list[dict[str, int | str]]:
        return self._runtime.action_layout()

    def scheduler_metrics(self) -> dict[str, object]:
        return self._runtime.scheduler_metrics()

    def timing_diagnostics(self) -> object:
        return self._runtime.timing_diagnostics()


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
        self._video_codec = cfg.encoder.video_codec
        self._depth_codec = cfg.encoder.depth_codec
        self._scheduler_driver = str(scheduler_driver)
        self._telemetry_hz = int(cfg.async_pipeline.telemetry_hz)
        self._control_hz = int(cfg.async_pipeline.control_hz)
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._opened = False
        self._lock = threading.Lock()
        self._pending_episode: RecordedEpisodeSummary | None = None

    @property
    def video_codec(self) -> str:
        return self._video_codec

    @property
    def depth_codec(self) -> str:
        return self._depth_codec

    @property
    def scheduler_driver(self) -> str:
        return self._scheduler_driver

    @property
    def telemetry_hz(self) -> int:
        return self._telemetry_hz

    @property
    def control_hz(self) -> int:
        return self._control_hz

    @property
    def recording(self) -> bool:
        return self.snapshot().recording

    @property
    def elapsed(self) -> float:
        return self.snapshot().elapsed

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
        self._opened = False
        self._pending_episode = None

    def snapshot(self) -> RuntimeSnapshot:
        result = self._request("snapshot")
        if not isinstance(result, RuntimeSnapshot):
            raise TypeError("Runtime worker returned an invalid snapshot payload.")
        return result

    def start_episode(self) -> int:
        self._pending_episode = None
        return int(self._request("start_episode"))

    def stop_episode(self) -> RecordedEpisodeSummary:
        result = self._request("stop_episode")
        if not isinstance(result, RecordedEpisodeSummary):
            raise TypeError("Runtime worker returned an invalid episode summary.")
        self._pending_episode = result
        return result

    def keep_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> EpisodeExportHandle:
        episode_index = _validate_pending_episode(self._pending_episode, episode)
        result = self._request("keep_episode", episode_index=episode_index)
        self._pending_episode = None
        if not isinstance(result, EpisodeExportHandle):
            raise TypeError("Runtime worker returned an invalid export handle.")
        return result

    def discard_episode(
        self,
        episode: RecordedEpisodeSummary | RecordedEpisode | int | None = None,
    ) -> None:
        episode_index = _validate_pending_episode(self._pending_episode, episode)
        self._request("discard_episode", episode_index=episode_index)
        self._pending_episode = None

    def return_robots_to_zero(self, timeout: float = 10.0) -> dict[str, bool]:
        result = self._request("return_robots_to_zero", timeout=float(timeout))
        if not isinstance(result, dict):
            raise TypeError("Runtime worker returned an invalid zeroing result.")
        return result

    def export_status(self) -> tuple[int, int]:
        return self.snapshot().export_status

    def wait_for_exports(self) -> None:
        self._request("wait_for_exports", response_timeout=None)

    def wait_for_episode_export(
        self, episode_index: int, timeout: float | None = None
    ) -> bool:
        result = self._request(
            "wait_for_episode_export",
            response_timeout=None if timeout is None else max(float(timeout) + 5.0, 5.0),
            episode_index=int(episode_index),
            timeout=timeout,
        )
        return bool(result)

    def latest_frames(self) -> dict[str, object]:
        return self.snapshot().latest_frames

    def latest_robot_states(self) -> dict[str, dict[str, object]]:
        return self.snapshot().latest_robot_states

    def latest_pair_modes(self) -> dict[str, str]:
        return self.snapshot().latest_pair_modes

    def action_layout(self) -> list[dict[str, int | str]]:
        return self.snapshot().action_layout

    def scheduler_metrics(self) -> dict[str, object]:
        return self.snapshot().scheduler_metrics

    def timing_diagnostics(self) -> object:
        return self.snapshot().timing_diagnostics

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
                    raise TimeoutError(f"Timed out while waiting for runtime {context}.")
                poll_timeout = min(poll_timeout, remaining)
            if self._conn.poll(poll_timeout):
                response = self._conn.recv()
                if not isinstance(response, _WorkerResponse):
                    raise TypeError("Runtime worker returned an invalid response payload.")
                if response.ok:
                    return response
                error_suffix = f"\n{response.traceback_text}" if response.traceback_text else ""
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
    use_worker: bool = True,
    export_delay_sec: float = 0.0,
    scheduler_driver: str = "asyncio",
    preview_live_feedback: bool = False,
    bootstrap_entries: tuple[str, ...] = (),
) -> CollectionRuntimeService:
    """Build one runtime service for TUI, scripts, or tests."""

    resolved_bootstrap_entries = (
        tuple(str(entry) for entry in bootstrap_entries)
        if bootstrap_entries
        else tuple(str(entry) for entry in cfg.async_pipeline.worker_bootstrap)
    )
    if use_worker:
        return WorkerCollectionRuntimeService(
            cfg,
            export_delay_sec=export_delay_sec,
            scheduler_driver=scheduler_driver,
            preview_live_feedback=preview_live_feedback,
            bootstrap_entries=resolved_bootstrap_entries,
        )
    return InProcessCollectionRuntimeService(
        AsyncCollectionRuntime.from_config(
            cfg,
            export_delay_sec=export_delay_sec,
            scheduler_driver=scheduler_driver,
            preview_live_feedback=preview_live_feedback,
        )
    )
