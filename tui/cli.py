"""Command-line interface: list, status, stop, launch. Invoked by llm-serve."""

from __future__ import annotations

import argparse
import sys

from tui.launch import (
    LaunchError,
    launch_background,
    launch_foreground,
    list_text,
    prepare_launch,
    status_text,
    stop_server,
)
from tui.paths import default_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-serve",
        add_help=False,
        description="Launch llama-server from models.json / presets.json",
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("args", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = default_paths()

    try:
        if args.help:
            print(list_text(paths=paths))
            return 0

        command = args.args[0] if args.args else ""
        rest = args.args[1:]

        if command == "status":
            text, code = status_text(paths=paths)
            print(text)
            return code

        if command == "stop":
            target = " ".join(rest).strip() or None
            print(stop_server(target, paths=paths))
            return 0

        if command == "list" or not command:
            print(list_text(paths=paths))
            return 0

        requested = " ".join(args.args)
        plan = prepare_launch(
            requested,
            remote=args.remote,
            paths=paths,
        )
        if requested != plan.model_key and requested != plan.display:
            print(f"Resolved: {requested} -> {plan.model_key}")
        print(f"Preset: [{plan.preset_slot}] {plan.preset_name} ({plan.quant})")
        print("\n".join(plan.banner_lines), end="" if plan.banner_lines[-1:] == [""] else "\n")

        if args.dry_run:
            print("Dry-run command:")
            print(f"  {plan.llama_server} {' '.join(plan.args)}")
            return 0

        if args.live:
            launch_foreground(plan)

        pid = launch_background(plan, paths=paths)
        print(f"Started {plan.model_key} in background (PID: {pid})")
        print(f"Logs:   {paths.log_file}")
        print()
        print("Use 'llm-serve status' to check")
        print("Use 'llm-serve stop' to stop")
        print(f"Use 'llm-serve stop {plan.model_key}' to stop this model")
        return 0
    except LaunchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
