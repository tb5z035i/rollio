"""Strict LeRobot v2.1 episode loader for replay."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from rollio.config.schema import RollioConfig

_FORMAT_FIELD_RE = re.compile(
    r"\{(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?::(?P<fmt>[^}]+))?\}"
)


@dataclass(frozen=True)
class ReplayCameraStream:
    """Metadata for one replayable camera stream."""

    name: str
    path: Path
    channel: str
    extension: str
    codec: str


@dataclass(frozen=True)
class ReplayEpisode:
    """Loaded episode data needed by the replay runtime."""

    dataset_root: Path
    episode_path: Path
    episode_relative_path: str
    episode_index: int
    episode_chunk: int
    fps: int
    timestamps: np.ndarray
    duration: float
    row_count: int
    action_layout: list[dict[str, int | str]]
    action: np.ndarray
    recorded_robot_states: dict[str, dict[str, np.ndarray]]
    camera_streams: dict[str, ReplayCameraStream]

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1]) if self.action.ndim == 2 else 0

    def state_at_index(self, robot_name: str, index: int) -> dict[str, np.ndarray]:
        """Return one robot-state snapshot for the requested replay row."""
        robot_state = self.recorded_robot_states.get(robot_name, {})
        if not robot_state:
            return {}
        safe_index = max(0, min(index, self.row_count - 1))
        return {
            key: np.asarray(values[safe_index], dtype=np.float32).reshape(-1)
            for key, values in robot_state.items()
        }

    def action_slice(self, entry: dict[str, int | str], index: int) -> np.ndarray:
        """Return the flattened action slice for one action-layout entry."""
        safe_index = max(0, min(index, self.row_count - 1))
        start = int(entry["start"])
        stop = int(entry["stop"])
        return np.asarray(
            self.action[safe_index, start:stop],
            dtype=np.float32,
        ).reshape(-1)


def load_replay_episode(
    cfg: RollioConfig,
    episode_path: str | Path,
) -> ReplayEpisode:
    """Load one replay episode strictly against the dataset metadata."""
    dataset_root = (Path(cfg.storage.root).expanduser() / cfg.project_name).resolve()
    selected_path = Path(episode_path).expanduser().resolve()
    if not selected_path.exists():
        raise FileNotFoundError(f"Replay episode not found: {selected_path}")

    meta_path = dataset_root / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {meta_path}"
        )

    info = json.loads(meta_path.read_text())
    codebase_version = str(info.get("codebase_version", "")).strip()
    if codebase_version != "v2.1":
        raise ValueError(
            f"Replay currently supports LeRobot v2.1 only, got {codebase_version!r}"
        )

    if cfg.storage.lerobot_version != "v2.1":
        raise ValueError(
            "Replay currently supports configurations targeting LeRobot v2.1 only"
        )

    try:
        episode_relative = selected_path.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Replay episode must live under dataset root {dataset_root}"
        ) from exc

    data_template = str(info.get("data_path", "")).strip()
    if not data_template:
        raise ValueError("Dataset metadata is missing data_path")

    parsed_values = _parse_template_match(
        template=data_template,
        candidate=episode_relative,
        label="episode parquet",
    )
    if "episode_index" not in parsed_values:
        raise ValueError("Dataset data_path template does not expose episode_index")

    episode_index = int(parsed_values["episode_index"])
    if "episode_chunk" in parsed_values:
        episode_chunk = int(parsed_values["episode_chunk"])
    else:
        episode_chunk = episode_index // 1000

    expected_relative = _format_template(
        data_template,
        {
            **parsed_values,
            "episode_index": episode_index,
            "episode_chunk": episode_chunk,
        },
    )
    if expected_relative != episode_relative:
        raise ValueError(
            "Replay episode path does not match dataset metadata: "
            f"expected {expected_relative}, got {episode_relative}"
        )

    table = pq.read_table(selected_path)
    column_names = list(table.column_names)

    if "timestamp" not in column_names:
        raise ValueError("Replay episode parquet is missing timestamp column")
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    row_count = int(len(timestamps))
    if row_count == 0:
        raise ValueError("Replay episode parquet is empty")
    duration = float(timestamps[-1] - timestamps[0]) if row_count > 1 else 0.0

    fps = int(info.get("fps", cfg.fps))
    if fps <= 0:
        raise ValueError(f"Invalid replay FPS in metadata: {fps}")

    raw_action_layout = info.get("action_layout", [])
    if not raw_action_layout:
        raise ValueError("Replay requires dataset action_layout metadata")
    action_layout = [_normalize_action_layout_entry(entry) for entry in raw_action_layout]

    if "action" not in column_names:
        raise ValueError("Replay requires an action column in the episode parquet")
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(-1, 1)
    if action.shape[0] != row_count:
        raise ValueError(
            f"Replay action rows mismatch timestamps: {action.shape[0]} vs {row_count}"
        )

    expected_action_dim = sum(int(entry["dim"]) for entry in action_layout)
    if action.shape[1] != expected_action_dim:
        raise ValueError(
            "Replay action dimension does not match metadata action_layout: "
            f"{action.shape[1]} vs {expected_action_dim}"
        )

    robots_by_name = {robot.name: robot for robot in cfg.robots}
    for entry in action_layout:
        follower = str(entry["follower"])
        if follower not in robots_by_name:
            raise ValueError(
                f"Replay action_layout follower is missing from config: {follower}"
            )

    recorded_robot_states = _extract_recorded_robot_states(table)

    video_template = str(info.get("video_path", "")).strip()
    if not video_template:
        raise ValueError("Dataset metadata is missing video_path")
    camera_streams = _resolve_camera_streams(
        cfg=cfg,
        info=info,
        dataset_root=dataset_root,
        video_template=video_template,
        template_values={
            **parsed_values,
            "episode_index": episode_index,
            "episode_chunk": episode_chunk,
        },
    )

    return ReplayEpisode(
        dataset_root=dataset_root,
        episode_path=selected_path,
        episode_relative_path=episode_relative,
        episode_index=episode_index,
        episode_chunk=episode_chunk,
        fps=fps,
        timestamps=timestamps,
        duration=duration,
        row_count=row_count,
        action_layout=action_layout,
        action=action,
        recorded_robot_states=recorded_robot_states,
        camera_streams=camera_streams,
    )


def _parse_template_match(
    *,
    template: str,
    candidate: str,
    label: str,
) -> dict[str, Any]:
    regex, field_specs = _compile_template_regex(template)
    match = regex.fullmatch(candidate)
    if match is None:
        raise ValueError(
            f"Selected {label} does not match dataset metadata template: {candidate}"
        )
    parsed: dict[str, Any] = {}
    for field_name, raw_value in match.groupdict().items():
        fmt = field_specs.get(field_name, "")
        if fmt.endswith("d"):
            parsed[field_name] = int(raw_value)
        else:
            parsed[field_name] = raw_value
    return parsed


def _compile_template_regex(template: str) -> tuple[re.Pattern[str], dict[str, str]]:
    parts: list[str] = []
    field_specs: dict[str, str] = {}
    cursor = 0
    for match in _FORMAT_FIELD_RE.finditer(template):
        parts.append(re.escape(template[cursor:match.start()]))
        field_name = match.group("name")
        field_fmt = match.group("fmt") or ""
        field_specs[field_name] = field_fmt
        if field_fmt.endswith("d"):
            parts.append(fr"(?P<{field_name}>\d+)")
        else:
            parts.append(fr"(?P<{field_name}>[^/]+)")
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("".join(parts)), field_specs


def _format_template(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise ValueError(
            f"Dataset metadata template requires unsupported field: {missing}"
        ) from exc


def _normalize_action_layout_entry(entry: dict[str, Any]) -> dict[str, int | str]:
    required = {"pair_name", "leader", "follower", "mode", "start", "stop", "dim"}
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(
            "Replay action_layout entry is missing keys: " + ", ".join(missing)
        )
    normalized = {
        "pair_name": str(entry["pair_name"]),
        "leader": str(entry["leader"]),
        "follower": str(entry["follower"]),
        "mode": str(entry["mode"]),
        "start": int(entry["start"]),
        "stop": int(entry["stop"]),
        "dim": int(entry["dim"]),
    }
    if normalized["stop"] - normalized["start"] != normalized["dim"]:
        raise ValueError(
            "Replay action_layout entry has inconsistent slice dimensions: "
            f"{normalized}"
        )
    return normalized


def _extract_recorded_robot_states(
    table: pq.Table,
) -> dict[str, dict[str, np.ndarray]]:
    recorded_states: dict[str, dict[str, np.ndarray]] = {}
    for column_name in table.column_names:
        prefix = "observation.state."
        if not column_name.startswith(prefix):
            continue
        suffix = column_name[len(prefix):]
        try:
            robot_name, property_name = suffix.split(".", 1)
        except ValueError:
            continue
        values = np.asarray(table.column(column_name).to_pylist(), dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        recorded_states.setdefault(robot_name, {})[property_name] = values
    return recorded_states


def _resolve_camera_streams(
    *,
    cfg: RollioConfig,
    info: dict[str, Any],
    dataset_root: Path,
    video_template: str,
    template_values: dict[str, Any],
) -> dict[str, ReplayCameraStream]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Dataset metadata features must be a mapping")

    cameras_by_name = {camera.name: camera for camera in cfg.cameras}
    metadata_video_keys = {
        str(feature_name).split(".", 2)[-1]
        for feature_name in features
        if str(feature_name).startswith("observation.images.")
    }
    missing_in_metadata = sorted(set(cameras_by_name) - metadata_video_keys)
    if missing_in_metadata:
        raise ValueError(
            "Replay config cameras are missing from dataset metadata: "
            + ", ".join(missing_in_metadata)
        )
    unexpected_in_metadata = sorted(metadata_video_keys - set(cameras_by_name))
    if unexpected_in_metadata:
        raise ValueError(
            "Dataset metadata contains camera streams not present in replay config: "
            + ", ".join(unexpected_in_metadata)
        )

    streams: dict[str, ReplayCameraStream] = {}
    for feature_name, feature_payload in features.items():
        if not str(feature_name).startswith("observation.images."):
            continue
        video_key = str(feature_name).split(".", 2)[-1]
        if not isinstance(feature_payload, dict):
            raise ValueError(
                f"Camera feature metadata must be an object for {feature_name}"
            )
        video_info = feature_payload.get("video_info")
        if not isinstance(video_info, dict):
            raise ValueError(
                f"Camera feature metadata is missing video_info for {feature_name}"
            )
        extension = str(video_info.get("extension", "")).strip()
        codec = str(video_info.get("codec", "")).strip()
        if not extension.startswith("."):
            raise ValueError(
                f"Camera feature metadata has invalid video extension for {feature_name}"
            )
        relative_path = _format_template(
            video_template,
            {
                **template_values,
                "video_key": video_key,
                "video_extension": extension,
            },
        )
        path = dataset_root / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Replay video file referenced by metadata is missing: {path}"
            )
        channel = cameras_by_name.get(video_key).channel if video_key in cameras_by_name else "color"
        streams[video_key] = ReplayCameraStream(
            name=video_key,
            path=path.resolve(),
            channel=str(channel or "color"),
            extension=extension,
            codec=codec,
        )
    return streams


__all__ = [
    "ReplayCameraStream",
    "ReplayEpisode",
    "load_replay_episode",
]
