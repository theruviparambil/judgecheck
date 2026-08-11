"""Command line entry point.

    judgecheck report <panel-dir> [--json] [--labels ...] [--positive-label ...]

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
"""

from __future__ import annotations

import argparse
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
        "--positive-label",
        default=POSITIVE,
        help=f"label treated as positive for recall/precision (default: {POSITIVE})",
    )
    return parser


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

    try:
        panel = load_panel(args.panel)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = build_report(panel, labels=labels, positive_label=args.positive_label)
    gate = check_gate(report, args.fail_under) if args.fail_under is not None else None

    print(to_json(report, gate) if args.json else render_text(report, gate))
    return 0 if gate is None or gate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
