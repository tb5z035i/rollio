"""Tests for terminal frame rendering helpers."""
from __future__ import annotations

import cv2
import numpy as np

from rollio.tui import renderer


def test_render_frame_matches_previous_resize_order() -> None:
    rng = np.random.default_rng(1234)
    frame = rng.integers(0, 256, size=(47, 63, 3), dtype=np.uint8)
    width = 19
    height = 7

    previous_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    previous_rgb = cv2.resize(
        previous_rgb,
        (width, height * 2),
        interpolation=cv2.INTER_AREA,
    )

    for mode in renderer.RENDER_MODES:
        expected = renderer._BUILDERS[mode](previous_rgb, width, height)
        actual = renderer.render_frame(frame, width, height, mode)
        assert actual == expected
