---
name: PyAV Codec Refactor
overview: Replace the current ffmpeg-subprocess codec probing and encoding path with a PyAV-based implementation. Keep the config and TUI flow stable where possible, trim the exposed codec set to the PyAV-proven subset, and validate availability automatically at runtime.
todos:
  - id: redesign-codec-option
    content: Refactor codec metadata and probing in `rollio/episode/codecs.py` to be PyAV-native.
    status: pending
  - id: rewrite-writer
    content: Replace subprocess-based `_write_video()` in `rollio/episode/writer.py` with an in-process PyAV encoder path.
    status: pending
  - id: stabilize-surface
    content: Keep config validation, wizard codec menus, and metadata output stable across the backend swap.
    status: pending
  - id: refresh-tests
    content: Rewrite codec/helper and writer tests to validate PyAV probing, encode success, and fast failure behavior.
    status: pending
  - id: doc-and-smoke-check
    content: Update docs and run runtime/service smoke tests to verify reviewed episodes leave the export queue.
    status: pending
isProject: false
---

# PyAV Codec Refactor Plan

## Assumptions

- `av` becomes a required dependency in [pyproject.toml](pyproject.toml).
- The new implementation is pure PyAV for encoding and probing: no CLI `ffmpeg` subprocess backend and no secondary OpenCV encoder path in the initial refactor.
- The codec menu can be trimmed to the subset that can be probed and encoded reliably through PyAV.

## Current Touchpoints

- [rollio/episode/writer.py](rollio/episode/writer.py): `LeRobotV21Writer.write()` currently flows through `write() -> _codec_for_camera() -> _write_video()`, and `_write_video()` constructs an `ffmpeg` command then streams `arr.tobytes()` into `stdin`.
- [rollio/episode/codecs.py](rollio/episode/codecs.py): `available_rgb_codec_options()` and `available_depth_codec_options()` currently depend on `discover_ffmpeg_encoders()` and `_probe_ffmpeg_encoder()`.
- [rollio/config/schema.py](rollio/config/schema.py): `EncoderConfig` validates `video_codec` and `depth_codec` via `get_rgb_codec_option()` / `get_depth_codec_option()`.
- [rollio/tui/wizard.py](rollio/tui/wizard.py): the setup wizard populates its codec pickers from `available_rgb_codec_options()` and `available_depth_codec_options()`.
- [tests/test_config_helpers.py](tests/test_config_helpers.py): codec discovery tests are built around mocking `subprocess.run`.

## Implementation Steps

### 1. Redesign codec metadata around PyAV

- Refactor `CodecOption` in [rollio/episode/codecs.py](rollio/episode/codecs.py) so it models PyAV/container concerns instead of CLI `ffmpeg` arguments.
- Replace `ffmpeg_codec` / `ffmpeg_args` with fields such as container extension/format, PyAV codec name, target stream pixel format, and codec option dict.
- Keep public codec names and alias normalization helpers where still useful, even if the available set is trimmed.

### 2. Replace subprocess probing with PyAV capability probing

- Remove the `discover_ffmpeg_encoders()` and `_probe_ffmpeg_encoder()` subprocess design in [rollio/episode/codecs.py](rollio/episode/codecs.py).
- Implement cached PyAV-based probing that attempts a tiny in-process encode for each candidate codec/container combination.
- Make `available_rgb_codec_options()` and `available_depth_codec_options()` continue to return only machine-usable codecs, but now based on PyAV probe results.
- Decide the trimmed default codec set up front in code, likely one safe RGB codec and one or two depth/grayscale codecs that PyAV can consistently create on your target machines.

### 3. Rewrite video export in the writer using PyAV

- Replace `_write_video()` in [rollio/episode/writer.py](rollio/episode/writer.py) with a PyAV encode loop using `av.open(..., mode='w')`, `add_stream(...)`, frame conversion, packet muxing, and final stream flush.
- Preserve the current camera routing logic from `_codec_for_camera()` so RGB and depth streams still select different codecs.
- Preserve the current input frame assumptions from `_infer_input_pixel_format()` but map them to PyAV frame construction and stream pixel formats explicitly.
- Remove subprocess-specific error handling and replace it with clear Python exceptions that include codec name, container path, and frame characteristics.
- Remove `_write_video_opencv()` if the pure PyAV assumption holds.

### 4. Keep metadata, config, and UI behavior stable

- Keep `info.json` generation in [rollio/episode/writer.py](rollio/episode/writer.py) stable so downstream LeRobot metadata does not regress.
- Keep `EncoderConfig` in [rollio/config/schema.py](rollio/config/schema.py) validating codec names through the same `get_*_codec_option()` entry points, but backed by the PyAV-aware codec table.
- Keep the TUI settings flow in [rollio/tui/wizard.py](rollio/tui/wizard.py) unchanged at the call-site level so the codec menu still comes from `available_*_codec_options()` and automatically reflects the machine’s capabilities.
- Update docs such as [doc/configuration.md](doc/configuration.md) and [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) to say codec availability is probed through PyAV rather than the system `ffmpeg` binary.

### 5. Rebuild the tests around PyAV behavior

- Replace subprocess-mocking tests in [tests/test_config_helpers.py](tests/test_config_helpers.py) with PyAV-oriented probes or targeted monkeypatches around the new probing helpers.
- Replace the stderr/deadlock regression in [tests/test_episode_writer.py](tests/test_episode_writer.py) with PyAV-native failure tests: unsupported codec, invalid pixel format, or failed mux/flush should raise quickly without hanging.
- Re-run and adjust end-to-end export tests in [tests/test_collect_runtime.py](tests/test_collect_runtime.py) and [tests/test_collect_service.py](tests/test_collect_service.py) only where codec metadata or default names change.
- Add at least one writer-level regression for 16-bit depth/grayscale frames, since depth handling is the highest-risk part of the PyAV migration.

## Validation Strategy

- Run codec-helper tests to verify alias normalization and automatic codec discovery still behave predictably.
- Run writer-focused tests for RGB and depth encodes, including at least one intentional failure case.
- Run the async export/runtime tests to confirm the worker/export queue still transitions episodes from pending to done.
- Manually smoke-test `rollio setup` codec selection and `rollio collect` with one reviewed episode to confirm the TUI still shows an accurate codec menu and completed export.

## Main Risks To Watch

- PyAV codec names, container support, or option names may not map 1:1 to the current ffmpeg CLI presets, especially for `h264_nvenc` and some lossless depth choices.
- 16-bit grayscale/depth frame handling needs explicit validation because PyAV pixel-format expectations can be stricter than the current raw-byte pipe.
- If `av` is made optional instead of required, the plan expands materially: it would need a backend abstraction and a second supported encoder path rather than a straight replacement.

