"""Episode writer — saves episodes in LeRobot v2.1 format."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rollio.config.schema import CameraConfig
from rollio.episode.recorder import EpisodeData
from rollio.episode.codecs import (
    CodecOption,
    get_depth_codec_option,
    get_rgb_codec_option,
)


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

    def __init__(
        self,
        root: str | Path,
        project_name: str,
        fps: int = 30,
        camera_configs: dict[str, CameraConfig] | None = None,
        video_codec: str = "libx264",
        depth_codec: str = "ffv1",
    ) -> None:
        self._root = Path(root).expanduser() / project_name
        self._fps = fps
        self._project = project_name
        self._total_episodes = 0
        self._total_frames = 0
        self._camera_names: list[str] = []
        self._robot_names: list[str] = []
        self._num_joints = 0
        self._camera_configs = camera_configs or {}
        self._video_codec = get_rgb_codec_option(video_codec)
        self._depth_codec = get_depth_codec_option(depth_codec)
        self._camera_video_info: dict[str, dict[str, str | int]] = {}

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
            codec_option = self._codec_for_camera(cam_name, frames_list)
            vid_path = vid_dir / f"episode_{idx:06d}{codec_option.file_extension}"
            self._write_video(vid_path, frames_list, ep.fps, codec_option)
            self._camera_video_info[cam_name] = {
                "codec": codec_option.name,
                "fps": self._fps,
                "extension": codec_option.file_extension,
            }

        # ── Write parquet ────────────────────────────────────────
        data_dir = self._root / "data" / chunk
        data_dir.mkdir(parents=True, exist_ok=True)
        pq_path = data_dir / f"episode_{idx:06d}.parquet"
        self._write_parquet(pq_path, ep)

        # ── Update meta ──────────────────────────────────────────
        self._total_episodes = max(self._total_episodes, idx + 1)
        self._total_frames += self._target_row_count(ep)
        self._write_meta()

        return self._root

    # ── private ────────────────────────────────────────────────────

    def _codec_for_camera(
        self,
        camera_name: str,
        frames: list[tuple[float, np.ndarray]],
    ) -> CodecOption:
        camera_cfg = self._camera_configs.get(camera_name)
        if camera_cfg is not None and camera_cfg.channel == "depth":
            return self._depth_codec
        if frames and frames[0][1].ndim == 2:
            return self._depth_codec
        return self._video_codec

    def _infer_input_pixel_format(self, frame: np.ndarray) -> str:
        if frame.ndim == 3 and frame.shape[2] == 3:
            return "bgr24"
        if frame.ndim == 2 and frame.dtype == np.uint16:
            return "gray16le"
        if frame.ndim == 2:
            return "gray"
        raise ValueError(f"Unsupported frame shape for export: {frame.shape}")

    def _write_video(
        self,
        path: Path,
        frames: list[tuple[float, np.ndarray]],
        fps: int,
        codec_option: CodecOption,
    ) -> None:
        if not frames:
            return
        first_frame = np.asarray(frames[0][1])
        h, w = first_frame.shape[:2]
        input_pix_fmt = self._infer_input_pixel_format(first_frame)
        output_pix_fmt = (
            codec_option.output_pixel_format
            if codec_option.kind == "rgb"
            else input_pix_fmt
        )
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            input_pix_fmt,
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec_option.ffmpeg_codec,
        ]
        if output_pix_fmt:
            command.extend(["-pix_fmt", output_pix_fmt])
        command.extend(codec_option.ffmpeg_args)
        command.append(str(path))
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            if codec_option.kind == "rgb":
                self._write_video_opencv(path, frames, fps)
                return
            raise RuntimeError("ffmpeg is required to export depth videos") from exc
        try:
            for _, frame in frames:
                arr = np.asarray(frame)
                if arr.shape[:2] != (h, w):
                    raise ValueError("All frames in one stream must share the same resolution")
                if proc.stdin is None:
                    raise RuntimeError("ffmpeg stdin was not created")
                proc.stdin.write(arr.tobytes())
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
        if return_code != 0:
            if codec_option.kind == "rgb":
                self._write_video_opencv(path, frames, fps)
                return
            raise RuntimeError(
                f"ffmpeg failed while encoding {path.name} with {codec_option.name}: {stderr.strip()}"
            )

    def _write_video_opencv(
        self,
        path: Path,
        frames: list[tuple[float, np.ndarray]],
        fps: int,
    ) -> None:
        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open fallback video writer for {path}")
        try:
            for _, frame in frames:
                arr = np.asarray(frame)
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                writer.write(arr)
        finally:
            writer.release()

    def _write_parquet(self, path: Path, ep: EpisodeData) -> None:
        timestamps = self._target_timestamps(ep)
        n_rows = len(timestamps)
        if n_rows == 0:
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

        sampled_states: dict[str, list[dict[str, np.ndarray]]] = {}
        for rob_name in self._robot_names:
            state_list = ep.robot_states.get(rob_name, [])
            sampled_states[rob_name] = [
                self._sample_state_at(state_list, ts)
                for ts in timestamps
            ]

        for i, ts in enumerate(timestamps):
            rows["frame_index"].append(i)
            rows["episode_index"].append(ep.episode_index)
            rows["index"].append(self._total_frames + i)
            rows["timestamp"].append(float(ts))

            for rob_name in self._robot_names:
                st = sampled_states[rob_name][i]
                for key in ("position", "velocity", "effort"):
                    col = f"observation.state.{rob_name}.{key}"
                    default = np.zeros(self._num_joints, np.float32)
                    rows[col].append(st.get(key, default).tolist())

            # Action = next sampled position of the first robot.
            if self._robot_names:
                first_rob = self._robot_names[0]
                next_i = min(i + 1, n_rows - 1)
                ns = sampled_states[first_rob][next_i]
                rows["action"].append(
                    ns.get("position", np.zeros(self._num_joints, np.float32)).tolist()
                )

        table = pa.table(rows)
        pq.write_table(table, str(path))

    def _sample_state_at(
        self,
        state_list: list[tuple[float, dict[str, np.ndarray]]],
        timestamp: float,
    ) -> dict[str, np.ndarray]:
        if not state_list:
            return {
                "position": np.zeros(self._num_joints, np.float32),
                "velocity": np.zeros(self._num_joints, np.float32),
                "effort": np.zeros(self._num_joints, np.float32),
            }

        state_timestamps = np.asarray([ts for ts, _ in state_list], dtype=np.float64)
        idx = int(np.searchsorted(state_timestamps, timestamp, side="left"))
        if idx <= 0:
            chosen = state_list[0][1]
        elif idx >= len(state_list):
            chosen = state_list[-1][1]
        else:
            prev_ts = state_timestamps[idx - 1]
            next_ts = state_timestamps[idx]
            chosen = state_list[idx - 1][1] if abs(timestamp - prev_ts) <= abs(next_ts - timestamp) else state_list[idx][1]

        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in chosen.items()
        }

    def _target_row_count(self, ep: EpisodeData) -> int:
        return len(self._target_timestamps(ep))

    def _target_timestamps(self, ep: EpisodeData) -> np.ndarray:
        if ep.duration <= 0:
            return np.zeros(1, dtype=np.float64)

        n_rows = max(1, int(round(ep.duration * max(ep.fps, 1))) + 1)
        if n_rows == 1:
            return np.zeros(1, dtype=np.float64)
        return np.linspace(0.0, float(ep.duration), n_rows, dtype=np.float64)

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
            video_info = self._camera_video_info.get(cam_name, {
                "codec": self._video_codec.name,
                "fps": self._fps,
                "extension": self._video_codec.file_extension,
            })
            features[key] = {
                "dtype": "video",
                "shape": [],  # filled per-video
                "video_info": video_info,
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
                          "{video_key}/episode_{episode_index:06d}{video_extension}",
            "features": features,
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=2) + "\n")
