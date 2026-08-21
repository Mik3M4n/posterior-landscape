"""Console entry point: one command runs or resumes the whole pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import load_settings
from .pipeline import run
from .validation import validate_configuration


def main(argv: list[str] | None = None) -> int:
    arguments_text = list(sys.argv[1:] if argv is None else argv)
    if arguments_text and arguments_text[0] not in {"run", "validate"} and not arguments_text[0].startswith("-"):
        arguments_text.insert(0, "run")
    parser = argparse.ArgumentParser(
        prog="posterior-landscape",
        description="Analyze posterior two-dimensional density landscapes.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run or resume an analysis")
    run_parser.add_argument("settings", help="Path to settings.ini")
    validate_parser = subparsers.add_parser(
        "validate", help="Validate input and resolved settings without analysis"
    )
    validate_parser.add_argument("settings", help="Path to settings.ini")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args(arguments_text)

    try:
        settings = load_settings(arguments.settings)
        if arguments.command == "validate":
            report = validate_configuration(settings)
            if arguments.as_json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(
                    f"Valid: {report['draws']} draws on {report['shape'][0]} x "
                    f"{report['shape'][1]}; domain={report['domain']}; "
                    f"input measure={report['input_density_measure']}; "
                    f"external={report['external_parameter']['available']}."
                )
                for warning in report["warnings"]:
                    print(f"WARNING: {warning}")
        else:
            run(settings)
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - the CLI boundary reports cleanly.
        print(f"posterior-landscape: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
