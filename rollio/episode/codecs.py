"""Codec discovery and encoding presets for Rollio exports."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class CodecOption:
    """A user-selectable export codec."""

    name: str
    label: str
    kind: str
    ffmpeg_codec: str
    file_extension: str
    input_pixel_format: str
    output_pixel_format: str | None = None
    ffmpeg_args: tuple[str, ...] = field(default_factory=tuple)


RGB_CODEC_OPTIONS: tuple[CodecOption, ...] = (
    CodecOption(
        name="h264_nvenc",
        label="H.264 (NVIDIA NVENC)",
        kind="rgb",
        ffmpeg_codec="h264_nvenc",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        output_pixel_format="yuv420p",
        ffmpeg_args=("-preset", "p4", "-cq", "19"),
    ),
    CodecOption(
        name="libx264",
        label="H.264 (libx264)",
        kind="rgb",
        ffmpeg_codec="libx264",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        output_pixel_format="yuv420p",
        ffmpeg_args=("-preset", "veryfast", "-crf", "18"),
    ),
    CodecOption(
        name="mpeg4",
        label="MPEG-4",
        kind="rgb",
        ffmpeg_codec="mpeg4",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        output_pixel_format="yuv420p",
        ffmpeg_args=("-q:v", "2"),
    ),
)

DEPTH_CODEC_OPTIONS: tuple[CodecOption, ...] = (
    CodecOption(
        name="ffv1",
        label="FFV1 (lossless)",
        kind="depth",
        ffmpeg_codec="ffv1",
        file_extension=".mkv",
        input_pixel_format="gray16le",
        output_pixel_format="gray16le",
    ),
    CodecOption(
        name="rawvideo",
        label="Raw video (lossless, large)",
        kind="depth",
        ffmpeg_codec="rawvideo",
        file_extension=".mkv",
        input_pixel_format="gray16le",
        output_pixel_format="gray16le",
    ),
)

RGB_CODEC_ALIASES = {"mp4v": "mpeg4"}
DEPTH_CODEC_ALIASES = {"raw": "rawvideo"}
RGB_PROBE_SOURCE = "color=size=320x240:rate=1:color=black"
DEPTH_PROBE_SOURCE = "nullsrc=size=16x16:rate=1"


def parse_ffmpeg_encoder_names(output: str) -> set[str]:
    """Extract encoder names from `ffmpeg -encoders` output."""
    encoders: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("Encoders:")
            or stripped.startswith("--")
        ):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _probe_ffmpeg_encoder(codec: CodecOption) -> bool:
    """Quickly check whether FFmpeg can actually use this encoder here."""
    if codec.kind == "rgb":
        # Hardware encoders such as NVENC can reject tiny synthetic frames.
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            RGB_PROBE_SOURCE,
            "-frames:v",
            "1",
            "-c:v",
            codec.ffmpeg_codec,
        ]
        if codec.output_pixel_format:
            command.extend(["-pix_fmt", codec.output_pixel_format])
        command.extend(codec.ffmpeg_args)
        command.extend(["-f", "null", "-"])
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            DEPTH_PROBE_SOURCE,
            "-vf",
            "format=gray16le",
            "-frames:v",
            "1",
            "-c:v",
            codec.ffmpeg_codec,
            "-pix_fmt",
            codec.output_pixel_format or codec.input_pixel_format,
            "-f",
            "null",
            "-",
        ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@lru_cache(maxsize=1)
def discover_ffmpeg_encoders() -> set[str]:
    """Return the set of FFmpeg encoder names available on this machine."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    if result.returncode != 0:
        return set()
    return parse_ffmpeg_encoder_names(result.stdout)


@lru_cache(maxsize=1)
def available_rgb_codec_options() -> tuple[CodecOption, ...]:
    """Return usable RGB codec options ordered by preference."""
    encoders = discover_ffmpeg_encoders()
    available: list[CodecOption] = []
    for option in RGB_CODEC_OPTIONS:
        if option.ffmpeg_codec not in encoders:
            continue
        if _probe_ffmpeg_encoder(option):
            available.append(option)
    return tuple(available or (RGB_CODEC_OPTIONS[-1],))


@lru_cache(maxsize=1)
def available_depth_codec_options() -> tuple[CodecOption, ...]:
    """Return usable depth codec options ordered by preference."""
    encoders = discover_ffmpeg_encoders()
    available: list[CodecOption] = []
    for option in DEPTH_CODEC_OPTIONS:
        if option.ffmpeg_codec != "rawvideo" and option.ffmpeg_codec not in encoders:
            continue
        if _probe_ffmpeg_encoder(option):
            available.append(option)
    return tuple(available or (DEPTH_CODEC_OPTIONS[0],))


def default_rgb_codec_name() -> str:
    """Return the preferred RGB codec name for this machine."""
    return available_rgb_codec_options()[0].name


def default_depth_codec_name() -> str:
    """Return the preferred depth codec name for this machine."""
    return available_depth_codec_options()[0].name


def _normalize_codec_name(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def get_rgb_codec_option(name: str) -> CodecOption:
    """Resolve one RGB codec option by configured name."""
    normalized = _normalize_codec_name(name, RGB_CODEC_ALIASES)
    for option in RGB_CODEC_OPTIONS:
        if option.name == normalized:
            return option
    raise KeyError(f"Unknown RGB codec: {name}")


def get_depth_codec_option(name: str) -> CodecOption:
    """Resolve one depth codec option by configured name."""
    normalized = _normalize_codec_name(name, DEPTH_CODEC_ALIASES)
    for option in DEPTH_CODEC_OPTIONS:
        if option.name == normalized:
            return option
    raise KeyError(f"Unknown depth codec: {name}")
