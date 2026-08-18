#!/usr/bin/env python3
"""Break the code on purpose and confirm the test suite notices.

    python scripts/mutation_sweep.py

A passing test suite proves the tests agree with the code. It does not prove
the tests would *disagree* with wrong code. This applies each mutation below to
a copy of the source, runs the suite, and reports any mutant that survived. A
survivor means no test distinguishes correct behaviour from broken behaviour at
that line.

This is not decoration. On this package it found four real defects:

  * Fleiss' kappa was nondeterministic. It iterated a `set[str]`, string hashing
    is randomized per process, and float addition is not associative, so the
    published value moved in its last two digits between runs.
  * `fleiss_kappa` and `krippendorff_alpha` accepted only a sequence of raters
    while every other function took a name -> labels mapping, so the obvious
    call died with an unrelated AttributeError.
  * Every malformed-JSONL branch in the loader was unprotected, because the
    real panel is complete and well-formed.
  * Every threshold comparison was unprotected, because no rater in the real
    panel sits exactly on one.

Two traps worth knowing if you write one of these yourself:

  1. Detect failure with pytest's *exit code*. Grepping stdout for "failed"
     misses the uppercase "FAILED" summary line and reports every mutant as a
     survivor.
  2. Restore with `shutil.copy`, not `copy2`. `copy2` preserves mtime, CPython
     validates `__pycache__` on `(mtime, size)`, and consecutive `>=` -> `>`
     mutants are identical in size within the same second -- so Python reuses
     stale bytecode and the mutant looks like it survived. This script also
     purges caches and disables bytecode writing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "judgecheck"

#: (label, filename, find, replace). Every occurrence of `find` is replaced,
#: because some invariants (the sorted item traversal) appear in more than one
#: statistic and must hold in all of them.
MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "cohen: p_e uses the row marginal twice",
        "agreement.py",
        "p_e += (row / n) * (col / n)",
        "p_e += (row / n) * (row / n)",
    ),
    (
        "cohen: miscounts scored items",
        "agreement.py",
        "matrix[ia][ib] += 1\n        n += 1",
        "matrix[ia][ib] += 1\n        n += 2",
    ),
    ("fleiss: includes items with a single rating", "agreement.py", "if n_i < 2:", "if n_i < 1:"),
    (
        "fleiss: drops the -n_i correction",
        "agreement.py",
        "(sum_sq - n_i) / (n_i * (n_i - 1))",
        "sum_sq / (n_i * (n_i - 1))",
    ),
    ("kripp: coincidence weight 1/(m-1) -> 1/m", "agreement.py", "1 / (m_u - 1)", "1 / m_u"),
    (
        "kripp: drops the (n-1) small-sample factor",
        "agreement.py",
        "1 - (n - 1) * (do_sum / de_sum)",
        "1 - (do_sum / de_sum)",
    ),
    (
        "panel input: mapping shape not normalized",
        "agreement.py",
        "if isinstance(raters, Mapping):",
        "if False:",
    ),
    (
        "panel input: item traversal left unsorted",
        "agreement.py",
        "items = sorted(item_set)",
        "items = list(item_set)",
    ),
    (
        "consensus: MAJORITY becomes >= instead of >",
        "consensus.py",
        "elif top_n > voters / 2:",
        "elif top_n >= voters / 2:",
    ),
    (
        "consensus: UNANIMOUS tolerates one dissent",
        "consensus.py",
        "if voters > 0 and top_n == voters:",
        "if voters > 0 and top_n >= voters - 1:",
    ),
    (
        "validity: abstention counted as a catch",
        "validity.py",
        "if labels.get(item) == positive_label)",
        'if labels.get(item) in (positive_label, "NEEDS_INVESTIGATION"))',
    ),
    (
        "validity: precision counts only scored items",
        "validity.py",
        "        called += 1\n        if truth.get(item) == positive_label:",
        "        if truth.get(item) == positive_label:\n            called += 1\n"
        "        if truth.get(item) == positive_label:",
    ),
    (
        "validity: recall denominator counts all truth items",
        "validity.py",
        "truth_positive_ids = [item for item, lbl in truth.items() if lbl == positive_label]",
        "truth_positive_ids = [item for item, lbl in truth.items()"
        " if lbl != positive_label or True]",
    ),
    (
        "accuracy: unlabeled items count against the rater",
        "validity.py",
        "        if rater_label is None:\n            continue\n",
        "",
    ),
    (
        "triage: SKEWED becomes >= instead of >",
        "triage.py",
        "if max_share > SKEW_THRESHOLD:",
        "if max_share >= SKEW_THRESHOLD:",
    ),
    (
        "triage: ABSTAINS becomes >= instead of >",
        "triage.py",
        "if abstention > ABSTENTION_THRESHOLD:",
        "if abstention >= ABSTENTION_THRESHOLD:",
    ),
    (
        "triage: DROP requires three flags",
        "triage.py",
        "return DROP if len(self.flags) >= 2 else REVIEW",
        "return DROP if len(self.flags) >= 3 else REVIEW",
    ),
    (
        "triage: accuracy threshold 0.6 -> 0.45",
        "triage.py",
        "ACCURACY_THRESHOLD = 0.6",
        "ACCURACY_THRESHOLD = 0.45",
    ),
    (
        "triage: redundancy becomes > instead of >=",
        "triage.py",
        "if res.kappa >= redundant_kappa:",
        "if res.kappa > redundant_kappa:",
    ),
    (
        "leaning: LENIENT becomes > instead of >=",
        "triage.py",
        "if tp_share >= LEAN_THRESHOLD:",
        "if tp_share > LEAN_THRESHOLD:",
    ),
    (
        "leaning: STRICT becomes > instead of >=",
        "triage.py",
        "elif strict_share >= LEAN_THRESHOLD:",
        "elif strict_share > LEAN_THRESHOLD:",
    ),
    (
        "io: malformed rows kept instead of skipped",
        "io.py",
        "if finding_id is None or label is None:",
        "if False:",
    ),
    (
        "io: numeric ids not normalized to str at the boundary",
        "io.py",
        "    if isinstance(value, (int, float)):\n        return str(value)",
        "    if isinstance(value, (int, float)):\n        return value  # type: ignore\n",
    ),
    (
        "io: bool ids coerced instead of rejected",
        "io.py",
        "    if isinstance(value, bool):\n        return None",
        "    if isinstance(value, bool):\n        pass",
    ),
    (
        "report: rounds the published floats",
        "report.py",
        '"value": report.fleiss.value,',
        '"value": round(report.fleiss.value, 3),',
    ),
    (
        "report: consensus counts double-count",
        "report.py",
        "counts[entry.consensus] += 1",
        "counts[entry.consensus] += 2",
    ),
    (
        "report: pairwise emitted unsorted",
        "report.py",
        "for (a, b), r in sorted(report.pairwise.items())",
        "for (a, b), r in report.pairwise.items()",
    ),
    (
        "report: text and json disagree on precision",
        "report.py",
        "{f.value:+.3f}",
        "{f.value:+.2f}",
    ),
    (
        "report: validity computed without truth",
        "report.py",
        "validity(panel.raters, panel.truth, positive_label) if panel.truth else None",
        "validity(panel.raters, panel.truth or {}, positive_label)",
    ),
    (
        "gate: ignores Fleiss",
        "report.py",
        "if self.fleiss < self.threshold:",
        "if False:",
    ),
    (
        "gate: ignores Krippendorff",
        "report.py",
        "if self.krippendorff < self.threshold:",
        "if False:",
    ),
    (
        "gate: fails a panel sitting exactly on the threshold",
        "report.py",
        "if self.fleiss < self.threshold:",
        "if self.fleiss <= self.threshold:",
    ),
    (
        "cli: failing gate does not set exit 1",
        "cli.py",
        "return 0 if gate is None or gate.passed else 1",
        "return 0",
    ),
    (
        "cli: fail-under range not validated",
        "cli.py",
        "if args.fail_under is not None and not -1.0 <= args.fail_under <= 1.0:",
        "if False:",
    ),
    (
        "cli: gate applied even when not requested",
        "cli.py",
        "gate = check_gate(report, args.fail_under) if args.fail_under is not None else None",
        "gate = check_gate(report, args.fail_under or 0.9)",
    ),
    (
        "cli: positive label not validated",
        "cli.py",
        "if args.positive_label not in labels:",
        "if False:",
    ),
    ("cli: label count not validated", "cli.py", "if len(labels) < 2:", "if False:"),
    (
        "cli: missing panel raises instead of exiting 2",
        "cli.py",
        '    except FileNotFoundError as exc:\n        print(f"error: {exc}", file=sys.stderr)\n'
        "        return 2",
        '    except ZeroDivisionError as exc:\n        print(f"error: {exc}", file=sys.stderr)\n'
        "        return 2",
    ),
]


def _purge_pycache() -> None:
    for cache in list(ROOT.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def _restore(backup: Path) -> None:
    for path in backup.glob("*.py"):
        shutil.copy(path, SRC / path.name)  # copy, not copy2 -- see module docstring


def _suite_passes() -> bool:
    _purge_pycache()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode == 0


def main() -> int:
    survivors: list[str] = []
    restored_green = False

    with tempfile.TemporaryDirectory(prefix="judgecheck-mutation-") as tmp:
        backup = Path(tmp)
        for path in SRC.glob("*.py"):
            shutil.copy(path, backup / path.name)

        # This script edits your working tree. The finally block restores it even
        # on Ctrl-C or an unhandled error, because the alternative is deleting the
        # backup with the temp directory and leaving the source mutated.
        try:
            if not _suite_passes():
                print("baseline is RED -- fix the suite before mutating", file=sys.stderr)
                return 2
            print(f"baseline green, {len(MUTANTS)} mutants\n")

            for label, filename, find, replace in MUTANTS:
                _restore(backup)
                path = SRC / filename
                text = path.read_text(encoding="utf-8")
                if find not in text:
                    print(f"  SKIP      {label} (pattern not found -- has the code moved?)")
                    survivors.append(f"{label} [pattern not found]")
                    continue
                path.write_text(text.replace(find, replace), encoding="utf-8")
                if _suite_passes():
                    survivors.append(label)
                    print(f"  SURVIVED  {label}")
                else:
                    print(f"  caught    {label}")
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        finally:
            _restore(backup)
            _purge_pycache()
            restored_green = _suite_passes()
            if not restored_green:
                print(
                    f"WARNING: the suite is red after restoring. Your working tree may still "
                    f"be mutated. Compare against git, or recover from {backup}.",
                    file=sys.stderr,
                )

    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    if not restored_green:
        return 2
    if survivors:
        print("\nsurvivors:")
        for name in survivors:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
