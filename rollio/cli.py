"""CLI entry points for Rollio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_setup(args: argparse.Namespace) -> None:
    """Run the interactive TUI setup wizard."""
    from rollio.tui.wizard import run_wizard

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"Config file already exists: {out_path}")
        print("Use --force to overwrite, or choose a different -o path.")
        sys.exit(1)

    cfg = run_wizard(str(out_path))

    if cfg is None:
        print("Setup cancelled.")
        sys.exit(0)

    cfg.save(out_path)
    print(f"\n✓ Config saved to {out_path}")
    print(f"  Project:  {cfg.project_name}")
    print(f"  Cameras:  {[c.name for c in cfg.cameras]}")
    print(f"  Robots:   {[r.name for r in cfg.robots]}")
    print(f"  Storage:  {cfg.storage.root}")
    print(f"\nNext step:")
    print(f"  rollio collect --config {out_path}")


def _cmd_collect(args: argparse.Namespace) -> None:
    """Run the data collection TUI."""
    from rollio.config.schema import RollioConfig
    from rollio.tui.app import run_collection

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = RollioConfig.load(cfg_path)
        print(f"Loaded config from {cfg_path}")
    else:
        print(f"Config not found at {cfg_path}.")
        print(f"Run 'rollio setup' first, or specify --config path.")
        sys.exit(1)

    print(f"Project: {cfg.project_name}")
    print(f"Cameras: {[c.name for c in cfg.cameras]} "
          f"({[c.type for c in cfg.cameras]})")
    print(f"Robots:  {[r.name for r in cfg.robots]} "
          f"({[r.type for r in cfg.robots]})")
    print(f"Storage: {cfg.storage.root}/{cfg.project_name}")
    print(f"FPS:     {cfg.fps}")
    print("Starting TUI…")

    run_collection(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rollio",
        description="Rollio — robot episode data collection")
    sub = parser.add_subparsers(dest="command")

    # ── setup ─────────────────────────────────────────────────────
    p_setup = sub.add_parser(
        "setup",
        help="Interactive TUI wizard: scan hardware, preview, name channels")
    p_setup.add_argument(
        "-o", "--output", default="rollio_config.yaml",
        help="Output config path (default: rollio_config.yaml)")
    p_setup.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite existing config")

    # ── collect ───────────────────────────────────────────────────
    p_collect = sub.add_parser("collect", help="Run data collection TUI")
    p_collect.add_argument(
        "-c", "--config", default="rollio_config.yaml",
        help="Config file path")

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup(args)
    elif args.command == "collect":
        _cmd_collect(args)
    else:
        parser.print_help()
        sys.exit(0)
