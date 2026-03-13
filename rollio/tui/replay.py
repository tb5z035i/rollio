"""Replay TUI for previously recorded episodes."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from rollio.config.schema import RollioConfig
from rollio.replay.runtime import ReplayRuntime
from rollio.tui.app import (
    _SYNC_E,
    _SYNC_S,
    _Term,
    _format_control_interval_preview,
    _format_joint_preview,
    _key_label,
    _matches_key_binding,
    _pad_ansi,
    _visible_len,
    _write_lines,
)
from rollio.tui.renderer import (
    MODE_LABELS,
    RENDER_MODES,
    blit_frame,
    calc_render_size,
    render_depth,
    render_frame,
)


def _render_preview(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    render_mode: str,
    channel: str,
) -> bytes:
    if frame.ndim == 2:
        depth_mode = "turbo" if channel == "depth" else "gray"
        return render_depth(frame, width, height, depth_mode)
    if frame.ndim == 3 and frame.shape[2] == 1:
        depth_mode = "turbo" if channel == "depth" else "gray"
        return render_depth(frame[:, :, 0], width, height, depth_mode)
    if channel in {"depth", "infrared"} and frame.ndim == 3 and frame.shape[2] == 3:
        gray_frame = frame[:, :, 0]
        depth_mode = "turbo" if channel == "depth" else "gray"
        return render_depth(gray_frame, width, height, depth_mode)
    return render_frame(frame, width, height, render_mode)


def _estimate_dual_robot_panel_height(
    recorded_states: dict[str, dict[str, np.ndarray]],
    live_states: dict[str, dict[str, np.ndarray]],
) -> int:
    height = 1
    for name in sorted(set(recorded_states) | set(live_states)):
        recorded_pos = recorded_states.get(name, {}).get("position")
        live_pos = live_states.get(name, {}).get("position")
        joints = max(
            len(recorded_pos) if recorded_pos is not None else 0,
            len(live_pos) if live_pos is not None else 0,
            1,
        )
        height += 1 + (joints * 2) + 2
    return height


def _state_rows(
    source_label: str,
    state: dict[str, np.ndarray],
    *,
    robot_type: str,
    panel_w: int,
) -> list[str]:
    lines: list[str] = []
    prefix = f"  {source_label:<4}"
    position = state.get("position")
    if position is None or len(position) == 0:
        lines.append(f"{prefix} (no data)")
        return lines

    bar_width = max(panel_w - 22, 0)
    for joint_idx, raw_value in enumerate(position):
        value_text, frac = _format_joint_preview(robot_type, float(raw_value))
        bar_len = int(frac * bar_width)
        bar = "█" * max(bar_len, 0) + "░" * max(bar_width - bar_len, 0)
        lines.append(
            f"{prefix}j{joint_idx} \x1b[36m{value_text}\x1b[0m "
            f"\x1b[33m{bar}\x1b[0m"
        )
    control_interval = state.get("control_loop_interval_ms")
    control_target = state.get("control_loop_target_interval_ms")
    if control_interval is not None and len(control_interval) > 0:
        interval_text, frac = _format_control_interval_preview(
            float(control_interval[0]),
            float(control_target[0])
            if control_target is not None and len(control_target) > 0
            else float(control_interval[0]),
        )
        bar_len = int(frac * bar_width)
        bar = "█" * max(bar_len, 0) + "░" * max(bar_width - bar_len, 0)
        lines.append(
            f"{prefix}ctrl \x1b[36m{interval_text}\x1b[0m "
            f"\x1b[33m{bar}\x1b[0m"
        )
    return lines


def _dual_robot_panel_lines(
    recorded_states: dict[str, dict[str, np.ndarray]],
    live_states: dict[str, dict[str, np.ndarray]],
    robot_types: dict[str, str],
    *,
    panel_w: int,
    panel_h: int,
) -> list[str]:
    lines: list[str] = [
        f"\x1b[1;93m{' ROBOT STATE (REC/LIVE) ':─<{max(panel_w - 1, 1)}}\x1b[0m",
    ]
    names = [name for name in robot_types if name in recorded_states or name in live_states]
    for name in names:
        lines.append(f"\x1b[1;97m {name} \x1b[0m")
        robot_type = robot_types.get(name, "")
        lines.extend(
            _state_rows(
                "REC",
                recorded_states.get(name, {}),
                robot_type=robot_type,
                panel_w=panel_w,
            )
        )
        lines.extend(
            _state_rows(
                "LIVE",
                live_states.get(name, {}),
                robot_type=robot_type,
                panel_w=panel_w,
            )
        )
        lines.append("")
    return (lines + [""] * panel_h)[:panel_h]


def _help_panel_lines(
    cfg: RollioConfig,
    runtime: ReplayRuntime,
    *,
    panel_w: int,
    panel_h: int,
) -> list[str]:
    start_key = _key_label(cfg.controls.start_stop)
    path_label = runtime.episode.episode_relative_path
    if _visible_len(path_label) > panel_w - 4:
        path_label = "…" + path_label[-(panel_w - 5):]
    lines = [
        f"\x1b[1;96m{' REPLAY HELP ':─<{max(panel_w - 1, 1)}}\x1b[0m",
        f"\x1b[1mState:\x1b[0m {runtime.state}",
        f"\x1b[1mEpisode:\x1b[0m {path_label}",
        f"\x1b[1mFPS:\x1b[0m {runtime.fps}",
        "",
        "\x1b[1mKeys\x1b[0m",
        f"  {start_key:<8} start/pause/continue",
        "  m        render mode",
        "  q        quit",
        "\x1b[1mWorkflow\x1b[0m",
        "  1. start replay",
        "  2. pause / continue",
        "  3. wait for return",
        "",
        "\x1b[1mHardware\x1b[0m",
        "  all robots opened",
        "  followers commanded",
        "  rec + live state shown",
    ]
    return (lines + [""] * panel_h)[:panel_h]


def _state_line(runtime: ReplayRuntime) -> str:
    if runtime.returning:
        return "RETURNING"
    if runtime.playing and runtime.paused:
        return f"PAUSED {runtime.elapsed:.1f}s"
    if runtime.playing:
        return f"PLAY {runtime.elapsed:.1f}s / {runtime.episode.duration:.1f}s"
    if runtime.completed:
        return "DONE"
    return "IDLE"


def _status_lines(
    runtime: ReplayRuntime,
    *,
    actual_fps: float,
    render_mode: str,
) -> tuple[str, str]:
    line1 = (
        f" State: {_state_line(runtime)}"
        f" │ Episode: {runtime.episode.episode_index}"
        f" │ Dataset FPS: {runtime.fps}"
        f" │ TUI FPS: {actual_fps:.1f}"
    )
    line2 = (
        f" Render: {MODE_LABELS[render_mode]}"
        f" │ Video streams: {len(runtime.latest_frames())}"
        f" │ Controlled followers: {len(runtime.episode.action_layout)}"
    )
    return line1, line2


def run_replay(cfg: RollioConfig, episode_path: str | Path) -> None:
    """Run the replay TUI for a selected episode."""
    runtime = ReplayRuntime.from_config(cfg, episode_path)
    robot_types = {robot.name: robot.type for robot in cfg.robots}
    camera_channels = {camera.name: camera.channel for camera in cfg.cameras}
    runtime_opened = False
    mode_idx = 1
    actual_fps = 0.0
    previous_frame_t = time.monotonic()

    try:
        runtime.open()
        runtime_opened = True
        with _Term() as term:
            out = sys.stdout.buffer
            target_dt = 1.0 / max(runtime.fps, 1)
            while True:
                t0 = time.monotonic()
                dt_frame = t0 - previous_frame_t
                previous_frame_t = t0
                if dt_frame > 0:
                    actual_fps = 0.9 * actual_fps + 0.1 / dt_frame
                render_mode = RENDER_MODES[mode_idx]

                key = term.key()
                if key == "q":
                    break
                if key == "m":
                    mode_idx = (mode_idx + 1) % len(RENDER_MODES)
                elif _matches_key_binding(key, cfg.controls.start_stop):
                    if runtime.returning:
                        pass
                    elif runtime.playing and not runtime.paused:
                        runtime.pause_playback()
                    elif runtime.paused:
                        runtime.resume_playback()
                    else:
                        runtime.start_playback()

                runtime.update()

                latest_frames = runtime.latest_frames()
                recorded_states = runtime.latest_recorded_robot_states()
                live_states = runtime.latest_live_robot_states()

                W, H = term.cols, term.rows
                status_h = 2
                help_w = max(28, min(38, W // 4))
                left_w = max(20, W - help_w)
                body_h = max(2, H - status_h)
                robot_h = (
                    min(
                        max(10, _estimate_dual_robot_panel_height(recorded_states, live_states)),
                        max(10, body_h // 2),
                    )
                    if recorded_states or live_states
                    else 0
                )
                cam_h = max(2, body_h - robot_h)

                help_lines = _help_panel_lines(
                    cfg,
                    runtime,
                    panel_w=help_w,
                    panel_h=body_h,
                )
                robot_lines = (
                    _dual_robot_panel_lines(
                        recorded_states,
                        live_states,
                        robot_types,
                        panel_w=left_w,
                        panel_h=robot_h,
                    )
                    if robot_h
                    else []
                )
                status_line_1, status_line_2 = _status_lines(
                    runtime,
                    actual_fps=actual_fps,
                    render_mode=render_mode,
                )

                frame_out = bytearray()
                frame_out.extend(b"\x1b[H")
                _write_lines(
                    frame_out,
                    row=1,
                    col=1,
                    width=left_w,
                    height=body_h,
                    lines=[],
                )

                camera_names = list(latest_frames.keys())
                if camera_names:
                    cam_count = len(camera_names)
                    cam_row = 1
                    for idx, camera_name in enumerate(camera_names):
                        remaining = max(cam_h - (cam_row - 1), 1)
                        remaining_cams = cam_count - idx
                        cam_h_each = max(1, remaining // max(remaining_cams, 1))
                        dedicated_label_row = cam_h_each > 1
                        preview_h = max(1, cam_h_each - 1) if dedicated_label_row else 1
                        frame = latest_frames.get(camera_name)
                        if frame is not None:
                            fh, fw = frame.shape[:2]
                            rw, rh = calc_render_size(fw, fh, left_w, preview_h)
                            rendered = _render_preview(
                                frame,
                                width=rw,
                                height=rh,
                                render_mode=render_mode,
                                channel=camera_channels.get(camera_name, "color"),
                            )
                            frame_out.extend(blit_frame(rendered, cam_row, 1))
                        else:
                            _write_lines(
                                frame_out,
                                row=cam_row,
                                col=1,
                                width=left_w,
                                height=preview_h,
                                lines=["\x1b[90m(no preview)\x1b[0m"],
                            )
                        label_row = cam_row + preview_h - 1
                        if dedicated_label_row:
                            label_row += 1
                        _write_lines(
                            frame_out,
                            row=label_row,
                            col=1,
                            width=left_w,
                            height=1,
                            lines=[f"\x1b[1;96m[{idx + 1}] {camera_name}\x1b[0m"],
                        )
                        cam_row += cam_h_each
                else:
                    _write_lines(
                        frame_out,
                        row=1,
                        col=1,
                        width=left_w,
                        height=cam_h,
                        lines=["\x1b[90m(no replay video streams)\x1b[0m"],
                    )

                if robot_h:
                    _write_lines(
                        frame_out,
                        row=cam_h + 1,
                        col=1,
                        width=left_w,
                        height=robot_h,
                        lines=robot_lines,
                    )

                _write_lines(
                    frame_out,
                    row=1,
                    col=left_w + 1,
                    width=help_w,
                    height=body_h,
                    lines=help_lines,
                )

                status_bytes = (
                    f"\x1b[{H-1};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{_pad_ansi(status_line_1[:W], W)}"
                    f"\x1b[{H};1H"
                    f"\x1b[48;5;236m\x1b[38;5;250m"
                    f"{_pad_ansi(status_line_2[:W], W)}"
                    f"\x1b[0m"
                ).encode()

                out.write(_SYNC_S + bytes(frame_out) + status_bytes + _SYNC_E)
                out.flush()

                elapsed = time.monotonic() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
    finally:
        if runtime_opened:
            try:
                print("Returning robots to zero...", flush=True)
                runtime.return_robots_to_zero(timeout=5.0)
            finally:
                runtime.close()
