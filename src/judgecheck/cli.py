"""Command line entry point.

    judgecheck report <panel-dir> [--json] [--labels ...] [--positive-label ...]
                                  [--fail-under K] [--min-effective N] [--groups FILE]
                                  [--intervals]

A panel directory holds one `<rater>.jsonl` per rater and an optional
`truth.json`. Without truth, agreement, consensus, and triage still work;
only validity needs it.

Exit codes:

    0  the report ran, and any requested gate passed
    1  --fail-under was given and the panel fell short
    2  usage or input error

The two failure codes are kept apart so CI can tell "this panel does not agree
enough" from "you invoked the tool wrong". There is no default threshold: how
much agreement is enough depends on what the panel decides and what a wrong
call costs, so `--fail-under` has to be supplied deliberately.

`--min-effective` gates on a different question: not whether the judges agree,
but how many of them are worth counting. A panel can clear an agreement floor
*because* its judges are redundant, so the two gates can move in opposite
directions and both are worth setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import load_panel
from .report import build_report, check_gate, render_text, to_json
from .types import LABELS
from .validity import POSITIVE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgecheck",
        description="Validate LLM-as-judge reliability with agreement, not accuracy.",
    )
    parser.add_argument("--version", action="version", version=f"judgecheck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="agreement, validity, consensus, and triage for a panel")
    rep.add_argument("panel", type=Path, help="directory of <rater>.jsonl files")
    rep.add_argument("--json", action="store_true", help="machine-readable output, full precision")
    rep.add_argument(
        "--labels",
        default=",".join(LABELS),
        help=f"comma-separated label set (default: {','.join(LABELS)})",
    )
    rep.add_argument(
        "--fail-under",
        type=float,
        metavar="KAPPA",
        default=None,
        help=(
            "exit 1 unless Fleiss' kappa AND Krippendorff's alpha are both at "
            "or above KAPPA (no default; omit to only report)"
        ),
    )
    rep.add_argument(
        "--min-effective",
        type=float,
        metavar="N",
        default=None,
        help=(
            "exit 1 unless the panel is worth at least N independent judges "
            "(Kish effective sample size; no default)"
        ),
    )
    rep.add_argument(
        "--groups",
        type=Path,
        metavar="FILE",
        default=None,
        help=(
            "JSON object mapping rater name to group name, for the group-agreement "
            "section (default: read the vendor field from the panel files)"
        ),
    )
    rep.add_argument(
        "--intervals",
        action="store_true",
        help=(
            "add bootstrap confidence intervals to the coefficients "
            "(implied whenever a gate is set; costs about a second)"
        ),
    )
    rep.add_argument(
        "--positive-label",
        default=POSITIVE,
        help=f"label treated as positive for recall/precision (default: {POSITIVE})",
    )
    return parser


def _load_groups(path: Path) -> dict[str, str]:
    """Read a rater -> group JSON object.

    Validated rather than trusted, for the same reason the panel loader is:
    `json.loads` returns `Any`, so a nested object here would travel silently
    into a grouping and surface as an unrelated failure inside a statistic.
    A bad grouping file is a usage error, so this raises instead of skipping.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"--groups {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"--groups {path} must be a JSON object of rater -> group")
    out: dict[str, str] = {}
    for rater, group in data.items():
        if not isinstance(rater, str) or not isinstance(group, str):
            raise ValueError(f"--groups {path}: every key and value must be a string")
        if rater.strip() and group.strip():
            out[rater.strip()] = group.strip()
    if not out:
        raise ValueError(f"--groups {path} has no usable rater -> group entries")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    labels = tuple(lbl.strip() for lbl in args.labels.split(",") if lbl.strip())
    if len(labels) < 2:
        print("error: --labels needs at least two labels", file=sys.stderr)
        return 2
    if args.positive_label not in labels:
        print(
            f"error: --positive-label {args.positive_label!r} is not in the label set "
            f"({', '.join(labels)})",
            file=sys.stderr,
        )
        return 2

    if args.fail_under is not None and not -1.0 <= args.fail_under <= 1.0:
        print(
            f"error: --fail-under {args.fail_under} is outside the range of a kappa "
            "coefficient (-1.0 to 1.0)",
            file=sys.stderr,
        )
        return 2

    if args.min_effective is not None and args.min_effective < 1.0:
        # n_eff is bounded below by 1: even perfectly redundant judges are worth
        # one judge. A floor under 1 can never fail, which is not what anyone
        # who typed it meant.
        print(
            f"error: --min-effective {args.min_effective} is below 1.0, which no panel "
            "can fail (a panel is always worth at least one judge)",
            file=sys.stderr,
        )
        return 2

    try:
        panel = load_panel(args.panel)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        # OSError covers FileNotFoundError, IsADirectoryError and PermissionError;
        # the other two cover a corrupt or non-UTF-8 truth.json. All of them are
        # "you pointed me at something I cannot read", which the exit-code table
        # promises is a 2. Previously only FileNotFoundError was caught and the
        # rest reached the user as tracebacks.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    groups = None
    if args.groups is not None:
        try:
            groups = _load_groups(args.groups)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # A gate implies intervals. Deciding a build from a point estimate without
    # showing the range that estimate spans is the failure this flag exists to
    # prevent, so it is not left to the user to remember.
    gated = args.fail_under is not None or args.min_effective is not None
    report = build_report(
        panel,
        labels=labels,
        positive_label=args.positive_label,
        groups=groups,
        intervals=args.intervals or gated,
    )
    gate = check_gate(report, args.fail_under, args.min_effective) if gated else None

    print(to_json(report, gate) if args.json else render_text(report, gate))
    return 0 if gate is None or gate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
