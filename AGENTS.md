# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Rollio is a Python package for timestamp-aligned robot episode data collection. It uses pseudo cameras and pseudo robots for development/testing without hardware. See `DESIGN.md` for full architecture.

### Running tests

```bash
pytest -m "not hardware" -v
```

All non-hardware tests run in ~1 second. The `hardware` marker gates tests requiring physical AIRBOT Play arms or CAN bus.

### Running the CLI

The `rollio` CLI entry point is installed at `~/.local/bin/rollio`. Subcommands: `setup`, `collect`, `test`, `completion`. Use `rollio --help` for details.

### Key caveats

- **TUI requires a real TTY**: `rollio setup` and `rollio collect` use `termios`/`tty` for raw terminal input and ANSI rendering. They cannot run in non-interactive shells. To test the core data pipeline programmatically, instantiate `PseudoCamera`, `PseudoRobot`, `EpisodeRecorder`, and `LeRobotV21Writer` directly in Python (see `rollio/sensors/pseudo_camera.py`, `rollio/sensors/pseudo_robot.py`, `rollio/episode/recorder.py`, `rollio/episode/writer.py`).
- **No linter configured**: The project has no ruff/flake8/mypy/pyright config. Use `py_compile` or `python -m compileall` for syntax checks.
- **Optional hardware deps**: `pyrealsense2`, `airbot_hardware_py`, `python-can`, `pin` (Pinocchio) are optional. Install only if testing with real hardware.
- **PATH**: `pip install -e .` installs the `rollio` script to `~/.local/bin/`. This directory is added to `PATH` in `~/.bashrc`.
