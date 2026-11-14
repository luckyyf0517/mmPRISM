"""Six release commands for verification, reconstruction, adaptation and translation."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from mmprism.release.paths import PACKAGE_ROOT, configure
from mmprism.release.verify import verify_data as check_data
from mmprism.release.verify import verify_weights as check_weights

COMMANDS = ("verify-data", "verify-weights", "reconstruct", "evaluate", "adapt", "translate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmprism", add_help=False)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--data-root")
    parser.add_argument("--config-dir", type=Path)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    args, remainder = parser.parse_known_args(argv)
    try:
        paths = configure(data_root=args.data_root, config_dir=args.config_dir)
        if args.command in ("verify-data", "verify-weights"):
            from mmprism.release.data import SIGNERS

            verify_parser = argparse.ArgumentParser(prog=f"mmprism-{args.command}")
            verify_parser.add_argument("--signer", action="append", choices=SIGNERS)
            options = verify_parser.parse_args(remainder)
            results = (
                check_data(paths, signers=options.signer)
                if args.command == "verify-data"
                else check_weights(paths)
            )
            print(json.dumps(results, indent=2))
            return 0 if all(row.get("matches", row["exists"]) for row in results) else 1
        if args.command == "reconstruct":
            from mmprism.release.data import SIGNERS
            from mmprism.release.inference import reconstruct_signer
            from mmprism.release.settings import release_device_memory

            command_parser = argparse.ArgumentParser(prog="mmprism-reconstruct")
            command_parser.add_argument("--signer", action="append", choices=SIGNERS)
            command_parser.add_argument(
                "--mode", choices=("zero-shot", "adapted"), default="zero-shot"
            )
            command_parser.add_argument("--device", default="cpu")
            options = command_parser.parse_args(remainder)
            for signer in dict.fromkeys(options.signer or SIGNERS):
                reconstruct_signer(signer, mode=options.mode, device=options.device)
                release_device_memory(options.device)
            return 0
        script = {
            "evaluate": "eval_cell_visibility.py",
            "adapt": "harness.py",
            "translate": "decode_new_test.py",
        }[args.command]
        previous = sys.argv
        try:
            sys.argv = [f"mmprism-{args.command}", *remainder]
            runpy.run_path(str(PACKAGE_ROOT / "scripts" / script), run_name="__main__")
        finally:
            sys.argv = previous
        return 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"mmprism: {error}\n")


def verify_data() -> int:
    return main(["verify-data", *sys.argv[1:]])


def verify_weights() -> int:
    return main(["verify-weights", *sys.argv[1:]])


def reconstruct() -> int:
    return main(["reconstruct", *sys.argv[1:]])


def evaluate() -> int:
    return main(["evaluate", *sys.argv[1:]])


def adapt() -> int:
    return main(["adapt", *sys.argv[1:]])


def translate() -> int:
    return main(["translate", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
