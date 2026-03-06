"""Tests for non-interactive collection TUI helpers."""
from __future__ import annotations

from types import SimpleNamespace

from rollio.tui.app import _help_panel_lines, _status_lines


def test_status_lines_include_queue_and_codecs() -> None:
    runtime = SimpleNamespace(
        recording=False,
        elapsed=0.0,
        video_codec="libx264",
        depth_codec="ffv1",
    )

    line1, line2 = _status_lines(
        runtime,
        pending_episode=None,
        episodes_kept=4,
        pending_exports=2,
        completed_exports=7,
        actual_fps=29.7,
        render_mode="16",
    )

    assert "State: IDLE" in line1
    assert "Export queue: 2 pending / 7 done" in line1
    assert "FPS: 29.7" in line1
    assert "Render: 16-clr" in line2
    assert "RGB codec: libx264" in line2
    assert "Depth codec: ffv1" in line2


def test_help_panel_shows_key_help_and_codecs() -> None:
    cfg = SimpleNamespace(
        mode="teleop",
        controls=SimpleNamespace(start_stop=" ", keep="k", discard="d"),
    )
    runtime = SimpleNamespace(video_codec="mpeg4", depth_codec="rawvideo")

    lines = _help_panel_lines(cfg, runtime, panel_w=32, panel_h=16)
    joined = "\n".join(lines)

    assert "RGB codec:" in joined
    assert "mpeg4" in joined
    assert "Depth codec:" in joined
    assert "rawvideo" in joined
    assert "SPACE" in joined
    assert "start / stop" in joined
