# AGENTS.md

## Cursor Cloud specific instructions

**Rollio** is a single Python package for collecting timestamp-aligned robot episode data. No external services, Docker, or databases are required.

### Running tests

```bash
pytest -m "not hardware"
```

- `tests/test_robot.py` — 58 tests covering pseudo robot, kinematics, control modes, and integration workflows. All pass without hardware.
- `tests/test_airbot.py` — AIRBOT-specific tests (mocked). Currently the entire module is skipped at collection time because `pytest.importorskip("pinocchio")` in a `@pytest.mark.skipif` decorator triggers a module-level skip. This is expected; the mocked AIRBOT tests will only collect if `pinocchio` is installed.
- Hardware tests (`-m hardware`) require physical AIRBOT arms on CAN bus and are not runnable in cloud environments.

### Linting

Black and pylint are configured in `pyproject.toml`. Install with `pip install -e .[lint]`. Run `black .` to format and `pylint rollio tests` to lint.

### CLI entry point

The `rollio` CLI is installed to `~/.local/bin/`. Ensure `PATH` includes `/home/ubuntu/.local/bin`. Subcommands: `setup`, `collect`, `test`, `completion`.

### Demonstrating core functionality without hardware

Use pseudo sensors (PseudoCamera, PseudoRobot) to exercise the full data-collection pipeline: config → sensors → EpisodeRecorder → LeRobotV21Writer → LeRobot v2.1 format (Parquet + MP4). See `DESIGN.md` for architecture details.

### PATH note

`pip install --user` places scripts in `/home/ubuntu/.local/bin`, which may not be on PATH by default. Add it with `export PATH="/home/ubuntu/.local/bin:$PATH"`.
