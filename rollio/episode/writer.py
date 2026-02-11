"""Episode writer — saves episodes in LeRobot v2.1 format."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rollio.episode.recorder import EpisodeData


class LeRobotV21Writer:
    """Writes episodes to disk in LeRobot v2.1 (episode-based) layout.

    Layout::

        {root}/{project}/
        ├── meta/
        │   └── info.json
        ├── data/
        │   └── chunk-000/
        │       └── episode_000000.parquet
        └── videos/
            └── chunk-000/
                └── {camera_name}/
                    └── episode_000000.mp4
    """

    def __init__(self, root: str | Path, project_name: str,
                 fps: int = 30) -> None:
        self._root = Path(root).expanduser() / project_name
        self._fps = fps
        self._project = project_name
        self._total_episodes = 0
        self._total_frames = 0
        self._camera_names: list[str] = []
        self._robot_names: list[str] = []
        self._num_joints = 0

    # ── public ─────────────────────────────────────────────────────

    def write(self, ep: EpisodeData) -> Path:
        """Write one episode to disk.  Returns the episode directory."""
        idx = ep.episode_index
        chunk = f"chunk-{idx // 1000:03d}"

        # Discover structure from first episode
        if not self._camera_names:
            self._camera_names = list(ep.camera_frames.keys())
        if not self._robot_names and ep.robot_states:
            self._robot_names = list(ep.robot_states.keys())
            first_rob = next(iter(ep.robot_states.values()))
            if first_rob:
                self._num_joints = len(first_rob[0][1].get(
                    "position", np.zeros(0)))

        # ── Write video files ────────────────────────────────────
        for cam_name, frames_list in ep.camera_frames.items():
            if not frames_list:
                continue
            vid_dir = self._root / "videos" / chunk / cam_name
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"episode_{idx:06d}.mp4"
            self._write_video(vid_path, frames_list, ep.fps)

        # ── Write parquet ────────────────────────────────────────
        data_dir = self._root / "data" / chunk
        data_dir.mkdir(parents=True, exist_ok=True)
        pq_path = data_dir / f"episode_{idx:06d}.parquet"
        self._write_parquet(pq_path, ep)

        # ── Update meta ──────────────────────────────────────────
        self._total_episodes = max(self._total_episodes, idx + 1)
        self._total_frames += self._count_frames(ep)
        self._write_meta()

        return self._root

    # ── private ────────────────────────────────────────────────────

    def _write_video(self, path: Path,
                     frames: list[tuple[float, np.ndarray]],
                     fps: int) -> None:
        if not frames:
            return
        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        try:
            for _, frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _write_parquet(self, path: Path, ep: EpisodeData) -> None:
        # Build aligned rows at the recording FPS
        # Use robot timestamps as the canonical axis
        n_frames = self._count_frames(ep)
        if n_frames == 0:
            return

        rows: dict[str, list] = {
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
        }

        # Collect state arrays per robot
        for rob_name in self._robot_names:
            for key in ("position", "velocity", "effort"):
                col = f"observation.state.{rob_name}.{key}"
                rows[col] = []

        # Action = next position (shifted by 1 step) for the first robot
        if self._robot_names:
            rows["action"] = []

        states = ep.robot_states
        # Use the first robot's timestamps as canonical
        if self._robot_names:
            first_rob = self._robot_names[0]
            state_list = states.get(first_rob, [])
        else:
            state_list = []

        for i in range(n_frames):
            rows["frame_index"].append(i)
            rows["episode_index"].append(ep.episode_index)
            rows["index"].append(self._total_frames + i)

            if i < len(state_list):
                ts, st = state_list[i]
                rows["timestamp"].append(float(ts))
            else:
                rows["timestamp"].append(i / max(ep.fps, 1))

            for rob_name in self._robot_names:
                rob_states = states.get(rob_name, [])
                if i < len(rob_states):
                    _, st = rob_states[i]
                else:
                    st = {"position": np.zeros(self._num_joints, np.float32),
                          "velocity": np.zeros(self._num_joints, np.float32),
                          "effort":   np.zeros(self._num_joints, np.float32)}
                for key in ("position", "velocity", "effort"):
                    col = f"observation.state.{rob_name}.{key}"
                    rows[col].append(st.get(key, np.zeros(0)).tolist())

            # Action = next position of first robot (shifted by 1)
            if self._robot_names:
                next_i = min(i + 1, len(state_list) - 1)
                if next_i >= 0 and next_i < len(state_list):
                    _, ns = state_list[next_i]
                    rows["action"].append(
                        ns.get("position", np.zeros(0)).tolist())
                else:
                    rows["action"].append([0.0] * self._num_joints)

        table = pa.table(rows)
        pq.write_table(table, str(path))

    def _count_frames(self, ep: EpisodeData) -> int:
        counts = []
        for v in ep.camera_frames.values():
            counts.append(len(v))
        for v in ep.robot_states.values():
            counts.append(len(v))
        return max(counts) if counts else 0

    def _write_meta(self) -> None:
        meta_dir = self._root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        features: dict = {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        }

        for cam_name in self._camera_names:
            key = f"observation.images.{cam_name}"
            features[key] = {
                "dtype": "video",
                "shape": [],  # filled per-video
                "video_info": {"codec": "mp4v", "fps": self._fps},
            }

        for rob_name in self._robot_names:
            for prop in ("position", "velocity", "effort"):
                key = f"observation.state.{rob_name}.{prop}"
                features[key] = {
                    "dtype": "float32",
                    "shape": [self._num_joints],
                }

        if self._robot_names:
            features["action"] = {
                "dtype": "float32",
                "shape": [self._num_joints],
            }

        info = {
            "codebase_version": "v2.1",
            "fps": self._fps,
            "total_episodes": self._total_episodes,
            "total_frames": self._total_frames,
            "data_path": "data/chunk-{episode_chunk:03d}/"
                         "episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/"
                          "{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=2) + "\n")
