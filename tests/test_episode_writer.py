"""Tests for LeRobot episode export writing."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import rollio.episode.writer as writer_module
from rollio.episode.codecs import get_depth_codec_option
from rollio.episode.writer import LeRobotV21Writer


def test_writer_does_not_deadlock_on_noisy_encoder_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_path = tmp_path / "noisy_encoder.py"
    helper_path.write_text(
        "import sys\n"
        "\n"
        "while True:\n"
        "    chunk = sys.stdin.buffer.read(4096)\n"
        "    if not chunk:\n"
        "        break\n"
        "    sys.stderr.write('encoder failure\\n' * 512)\n"
        "    sys.stderr.flush()\n"
        "\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    real_popen = subprocess.Popen
    child_processes: list[subprocess.Popen[bytes]] = []

    def fake_popen(command, **kwargs):
        del command
        process = real_popen(
            [sys.executable, str(helper_path)],
            **kwargs,
        )
        child_processes.append(process)
        return process

    monkeypatch.setattr(writer_module.subprocess, "Popen", fake_popen)

    writer = LeRobotV21Writer(
        root=tmp_path,
        project_name="stderr_deadlock_guard",
        fps=10,
        camera_configs={},
        video_codec="libx264",
        depth_codec="ffv1",
    )
    frames = [(0.0, np.zeros((512, 512), dtype=np.uint16))]
    codec = get_depth_codec_option("ffv1")
    output_path = tmp_path / "episode_000000.mkv"
    result: dict[str, Exception] = {}

    def run_write() -> None:
        try:
            writer._write_video(  # pylint: disable=protected-access
                output_path,
                frames,
                10,
                codec,
            )
        except Exception as exc:  # pragma: no cover - assertion inspects captured value
            result["error"] = exc

    thread = threading.Thread(target=run_write, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    if thread.is_alive():
        for process in child_processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
        pytest.fail("Episode writer hung while the encoder filled stderr output.")

    error = result.get("error")
    assert isinstance(error, RuntimeError)
    assert "ffmpeg failed while encoding" in str(error)
