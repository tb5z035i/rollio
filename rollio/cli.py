"""CLI entry points for Rollio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Try to import argcomplete for shell completion
try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


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

    try:
        run_collection(cfg)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"Collection startup failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_test(args: argparse.Namespace) -> None:
    """Run hardware tests."""
    from rollio.tests import get_available_tests, get_test_description, run_test

    # List tests if requested
    if args.list:
        print("Available tests:")
        for test_name in get_available_tests():
            desc = get_test_description(test_name) or ""
            print(f"  {test_name:<40} {desc}")
        return

    # Run specified test
    if not args.test_name:
        print("Error: No test specified.")
        print("Use --list to see available tests.")
        print("Usage: rollio test <test_name> [options]")
        sys.exit(1)

    test_name = args.test_name

    # Build kwargs from args
    kwargs = {}
    if hasattr(args, 'device') and args.device:
        kwargs['can_interface'] = args.device
    if hasattr(args, 'duration') and args.duration is not None:
        kwargs['duration'] = args.duration
    if hasattr(args, 'verbose'):
        kwargs['verbose'] = args.verbose
    if hasattr(args, 'no_return_zero'):
        kwargs['return_to_zero'] = not args.no_return_zero

    print(f"Running test: {test_name}")
    print("-" * 40)

    success = run_test(test_name, **kwargs)

    if success:
        print("\n✓ Test passed")
        sys.exit(0)
    else:
        print("\n✗ Test failed")
        sys.exit(1)


class TestNameCompleter:
    """Completer for test names (for argcomplete)."""

    def __call__(self, prefix: str, parsed_args=None, **kwargs) -> list[str]:
        from rollio.tests import get_available_tests
        tests = get_available_tests()
        return [t for t in tests if t.startswith(prefix)]


def _cmd_completion(args: argparse.Namespace) -> None:
    """Install or print shell completion."""
    import os
    import subprocess
    
    shell = args.shell
    
    # Auto-detect shell if not specified
    if shell is None:
        shell_path = os.environ.get("SHELL", "")
        if "zsh" in shell_path:
            shell = "zsh"
        elif "bash" in shell_path:
            shell = "bash"
        elif "fish" in shell_path:
            shell = "fish"
        else:
            shell = "bash"  # Default to bash
    
    # Check if argcomplete is installed
    if not ARGCOMPLETE_AVAILABLE:
        print("Shell completion requires argcomplete.")
        print("\nInstall with:")
        print("  pip install argcomplete")
        print("\nOr install rollio with completion support:")
        print("  pip install rollio[completion]")
        sys.exit(1)
    
    # Generate completion script
    completion_line = 'eval "$(register-python-argcomplete rollio)"'
    
    if shell == "bash":
        rc_file = Path.home() / ".bashrc"
        script = completion_line
    elif shell == "zsh":
        rc_file = Path.home() / ".zshrc"
        script = f"""autoload -U bashcompinit
bashcompinit
{completion_line}"""
    elif shell == "fish":
        rc_file = Path.home() / ".config/fish/completions/rollio.fish"
        # Fish uses a different format
        try:
            result = subprocess.run(
                ["register-python-argcomplete", "--shell", "fish", "rollio"],
                capture_output=True, text=True
            )
            script = result.stdout
        except Exception:
            print("Error: Could not generate fish completion script.")
            sys.exit(1)
    else:
        print(f"Unsupported shell: {shell}")
        print("Supported shells: bash, zsh, fish")
        sys.exit(1)
    
    # Print script if --print is specified
    if args.print_script:
        print(script)
        return
    
    # Install completion
    if args.install:
        # Check if already installed
        if rc_file.exists():
            content = rc_file.read_text()
            if "register-python-argcomplete rollio" in content:
                print(f"Completion already installed in {rc_file}")
                return
        
        # For fish, create the completions directory if needed
        if shell == "fish":
            rc_file.parent.mkdir(parents=True, exist_ok=True)
            rc_file.write_text(script)
            print(f"Fish completion installed to {rc_file}")
        else:
            # Append to rc file
            with open(rc_file, "a") as f:
                f.write(f"\n# Rollio shell completion\n{script}\n")
            print(f"Completion installed to {rc_file}")
        
        print(f"\nRestart your shell or run:")
        print(f"  source {rc_file}")
        return
    
    # Default: print instructions
    print(f"Shell completion for {shell}")
    print("=" * 40)
    print()
    print("Option 1: Install automatically")
    print(f"  rollio completion --install")
    print()
    print("Option 2: Add manually to your shell config")
    print(f"  Add the following to {rc_file}:")
    print()
    for line in script.split("\n"):
        print(f"    {line}")
    print()
    print("Option 3: Print the completion script")
    print("  rollio completion --print")


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

    # ── test ──────────────────────────────────────────────────────
    p_test = sub.add_parser(
        "test",
        help="Run hardware tests")
    p_test.add_argument(
        "test_name",
        nargs="?",
        help="Name of the test to run"
    ).completer = TestNameCompleter() if ARGCOMPLETE_AVAILABLE else None
    p_test.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available tests")
    p_test.add_argument(
        "-d", "--device",
        default="can0",
        help="Device/interface name (e.g., can0 for AIRBOT)")
    p_test.add_argument(
        "-t", "--duration",
        type=float,
        default=None,
        help="Test duration in seconds (default: indefinite)")
    p_test.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)")
    p_test.add_argument(
        "-q", "--quiet",
        action="store_false",
        dest="verbose",
        help="Quiet output")
    p_test.add_argument(
        "--no-return-zero",
        action="store_true",
        default=False,
        help="Do not return arm to zero position after test (AIRBOT tests)")

    # ── completion ────────────────────────────────────────────────────
    p_completion = sub.add_parser(
        "completion",
        help="Install shell completion for bash/zsh/fish")
    p_completion.add_argument(
        "shell",
        nargs="?",
        choices=["bash", "zsh", "fish"],
        help="Shell type (auto-detected if not specified)")
    p_completion.add_argument(
        "--install", "-i",
        action="store_true",
        help="Install completion to shell config file")
    p_completion.add_argument(
        "--print", "-p",
        action="store_true",
        dest="print_script",
        help="Print completion script to stdout")

    # Enable argcomplete if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup(args)
    elif args.command == "collect":
        _cmd_collect(args)
    elif args.command == "test":
        _cmd_test(args)
    elif args.command == "completion":
        _cmd_completion(args)
    else:
        parser.print_help()
        sys.exit(0)
