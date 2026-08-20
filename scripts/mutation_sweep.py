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

This script never touches your working tree. It copies the repository into a
scratch directory and mutates the copy. That is not fastidiousness: the earlier
version edited `src/` in place and restored it in a `finally`, which loses the
race against a SIGKILL and, worse, against a second concurrent run. Two runs
overlapping means one restores a file while the other is measuring it, and
mutants report SURVIVED without ever having been tested. Both happened here.

Three traps worth knowing if you write one of these yourself:

  1. Detect failure with pytest's *exit code*. Grepping stdout for "failed"
     misses the uppercase "FAILED" summary line and reports every mutant as a
     survivor.
  2. Bytecode caching will lie to you. CPython validates `__pycache__` on
     `(mtime, size)`, and consecutive `>=` -> `>` mutants are identical in size
     within the same second, so a stale `.pyc` gets reused and the mutant looks
     like it survived. This script purges caches and sets
     `PYTHONDONTWRITEBYTECODE`.
  3. Do not mutate the tree you are working in. See above.
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

#: One mutant was retired as *equivalent* rather than left as a false survivor:
#: removing the `math.isnan` guard in `effective_raters` changes nothing,
#: because `max(0.0, nan)` returns `0.0` (nan > 0.0 is False) and the clamp
#: neutralizes NaN on its own. The guard stays because that behaviour depends
#: entirely on argument order -- `max(0.0, min(1.0, nan))` gives 1.0, i.e.
#: "maximally correlated" -- and relying on it implicitly is a trap. An
#: unkillable mutant is a fact about the code, not a gap in the tests, and
#: listing it would inflate the denominator with something no test can win.
#:
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
        "        elif top_n == voters:",
        "        elif top_n >= voters - 1:",
    ),
    (
        "validity: abstention counted as a catch",
        "validity.py",
        "if labels.get(item) == positive_label)",
        'if labels.get(item) in (positive_label, "NEEDS_INVESTIGATION"))',
    ),
    (
        "validity: precision denominator miscounts the calls",
        "validity.py",
        "        called += 1\n        if truth[item] == positive_label:",
        "        if truth[item] == positive_label:\n            called += 1\n"
        "        if truth[item] == positive_label:",
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
        "if abstention is not None and abstention > ABSTENTION_THRESHOLD:",
        "if abstention is not None and abstention >= ABSTENTION_THRESHOLD:",
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
        "cli: gate applied even when neither threshold was requested",
        "cli.py",
        "    gate = check_gate(report, args.fail_under, args.min_effective) if gated else None",
        "    gate = check_gate(report, args.fail_under, args.min_effective)",
    ),
    (
        "cli: positive label not validated",
        "cli.py",
        "if args.positive_label not in labels:",
        "if False:",
    ),
    ("cli: label count not validated", "cli.py", "if len(labels) < 2:", "if False:"),
    (
        "independence: negative mean phi not clamped, invents extra judges",
        "independence.py",
        "rho = min(1.0, max(0.0, mean_phi))",
        "rho = min(1.0, mean_phi)",
    ),
    (
        "independence: phi above one not clamped",
        "independence.py",
        "rho = min(1.0, max(0.0, mean_phi))",
        "rho = max(0.0, mean_phi)",
    ),
    (
        "independence: single rater not special-cased",
        "independence.py",
        "    if k < 2:\n        return float(k)",
        "    if False:\n        return float(k)",
    ),
    (
        "independence: design effect uses k instead of k-1",
        "independence.py",
        "return k / (1 + (k - 1) * rho)",
        "return k / (1 + k * rho)",
    ),
    (
        "independence: caution band boundary flipped",
        "independence.py",
        "if efficiency >= CAUTION_EFFICIENCY:",
        "if efficiency > CAUTION_EFFICIENCY:",
    ),
    (
        "independence: unmeasurable panel reports full independence",
        "independence.py",
        "    if k < 2 or not phi:",
        "    if False:",
    ),
    (
        "independence: phi computed on too few items anyway",
        "independence.py",
        "    if n < MIN_PAIR_ITEMS:\n        return None",
        "    if n < 0:\n        return None",
    ),
    (
        "independence: constant error vector correlated as zero not undefined",
        "independence.py",
        "    if da == 0.0 or db == 0.0:\n        return None",
        "    if da == 0.0 or db == 0.0:\n        return 0.0",
    ),
    (
        "independence: expected uses the joint rate instead of the product",
        "independence.py",
        "expected = (a_wrong / n) * (b_wrong / n) if n else 0.0",
        "expected = both / n if n else 0.0",
    ),
    (
        "independence: coincident error counts either-wrong not both-wrong",
        "independence.py",
        "both = sum(1 for x, y in zip(ea, eb, strict=True) if x and y)",
        "both = sum(1 for x, y in zip(ea, eb, strict=True) if x or y)",
    ),
    (
        "independence: undefined lift reported as zero",
        "independence.py",
        "lift=(observed / expected if expected > 0 else None),",
        "lift=(observed / expected if expected > 0 else 0.0),",
    ),
    (
        "independence: excess drops the independence baseline",
        "independence.py",
        "return both_wrong - (a_wrong * b_wrong / n if n else 0.0)",
        "return float(both_wrong)",
    ),
    (
        "independence: worst pair ranked by lift instead of excess",
        "independence.py",
        "worst = max(comparable, key=lambda p: p.excess) if comparable else None",
        "worst = max(comparable, key=lambda p: p.lift or 0.0) if comparable else None",
    ),
    (
        "independence: permutation p loses the add-one correction",
        "independence.py",
        "p_value = (hits + 1) / (permutations + 1)",
        "p_value = hits / permutations",
    ),
    (
        "independence: permutation null drawn unseeded",
        "independence.py",
        "rng = random.Random(PERMUTATION_SEED)",
        "rng = random.Random()",
    ),
    (
        "independence: empty within-group side reported as 0.0",
        "independence.py",
        "within_mean = sum(within) / len(within) if within else None",
        "within_mean = sum(within) / len(within) if within else 0.0",
    ),
    (
        "independence: ungrouped raters folded into the between side",
        "independence.py",
        "grouped = [n for n in names if n in groups]",
        "grouped = list(names)",
    ),
    (
        "independence: pairs with no shared items counted as agreement",
        "independence.py",
        "        if result.n == 0:\n            continue",
        "        if False:\n            continue",
    ),
    (
        "agreement: fleiss no-variance reported as perfect",
        "agreement.py",
        "            n=used_items, raters=len(rater_list), value=0.0, "
        "interpretation=UNDEFINED_NO_VARIANCE",
        "            n=used_items, raters=len(rater_list), value=1.0, "
        "interpretation=UNDEFINED_NO_VARIANCE",
    ),
    (
        "agreement: cohen no-variance reported as perfect",
        "agreement.py",
        "        return KappaResult(n=n, agreement=p_o, kappa=0.0, "
        "interpretation=UNDEFINED_NO_VARIANCE)",
        "        return KappaResult(n=n, agreement=p_o, kappa=1.0, "
        "interpretation=UNDEFINED_NO_VARIANCE)",
    ),
    (
        "agreement: alpha no-variance reported as perfect",
        "agreement.py",
        "    if de_sum == 0:",
        "    if False:",
    ),
    (
        "agreement: zero overlap banded as poor rather than undefined",
        "agreement.py",
        "        return KappaResult(n=0, agreement=0.0, kappa=0.0, interpretation=UNDEFINED)",
        "        return KappaResult(n=0, agreement=0.0, kappa=0.0, "
        "interpretation=interpret_kappa(0.0))",
    ),
    (
        "independence: k counts raters that contributed no comparable pair",
        "independence.py",
        "    contributing = sorted({name for pair in phi for name in pair})",
        "    contributing = sorted(named)",
    ),
    (
        "independence: reports the optimistic estimator instead of the conservative",
        "independence.py",
        "    lo, hi = min(kish, eigen), max(kish, eigen)",
        "    lo, hi = max(kish, eigen), min(kish, eigen)",
    ),
    (
        "independence: exchangeability never questioned",
        "independence.py",
        "    exchangeable = hi <= lo * EXCHANGEABILITY_TOLERANCE",
        "    exchangeable = True",
    ),
    (
        "independence: saturation flag never set",
        "independence.py",
        "        saturated=mean_phi <= 0.0,",
        "        saturated=False,",
    ),
    (
        "independence: eigenvalue shift dropped, power iteration finds the wrong end",
        "independence.py",
        "    shift = max(sum(abs(value) for value in row) for row in matrix)",
        "    shift = 0.0",
    ),
    (
        "independence: power iteration started from an eigenvector",
        "independence.py",
        "    vector = [1.0 + i / n for i in range(n)]",
        "    vector = [1.0 for _ in range(n)]",
    ),
    (
        "independence: panel shape guard the docstring promises is absent",
        "independence.py",
        "        if any(isinstance(v, str) for v in values.values()):",
        "        if False:",
    ),
    (
        "independence: NaN mean routed to full independence instead of not-measurable",
        "independence.py",
        "    if math.isnan(kish):",
        "    if False:",
    ),
    (
        "independence: permutation drawn on the wrong item basis",
        "independence.py",
        "        shared = coverage[a] & coverage[b]",
        "        shared = set.intersection(*coverage.values()) if coverage else set()",
    ),
    (
        "independence: permutation ignores per-judge error counts",
        "independence.py",
        "        out[name] = set(rng.sample(scored, count)) if count else set()",
        "        out[name] = set(rng.sample(scored, len(scored) // 2)) if count else set()",
    ),
    (
        "uncertainty: interval tracks the optimistic estimator not the reported one",
        "uncertainty.py",
        "            if drawn.effective is not None:\n                "
        "effective_draws.append(drawn.effective)",
        "            if drawn.effective_raters is not None:\n                "
        "effective_draws.append(drawn.effective_raters)",
    ),
    (
        "triage: no labels at all still reports an abstention rate",
        "triage.py",
        "        if ABSTENTION_LABEL in allowed and labeled:",
        "        if ABSTENTION_LABEL in allowed:",
    ),
    (
        "io: vendor not validated, structured value reaches the grouping",
        "io.py",
        'vendor=_as_text(obj.get("vendor")),',
        'vendor=obj.get("vendor"),',
    ),
    (
        "io: boolean confidence coerced to 1",
        "io.py",
        "    if isinstance(value, bool):\n        return None\n    if isinstance(value, int):",
        "    if isinstance(value, int):",
    ),
    (
        "gate: independence floor ignored",
        "report.py",
        "        if self.min_effective is not None:",
        "        if False:",
    ),
    (
        "gate: unmeasurable independence silently passes the floor",
        "report.py",
        "            if self.effective is None:",
        "            if False:",
    ),
    (
        "gate: independence floor fails a panel sitting exactly on it",
        "report.py",
        "            elif self.effective < self.min_effective:",
        "            elif self.effective <= self.min_effective:",
    ),
    (
        "cli: min-effective range not validated",
        "cli.py",
        "if args.min_effective is not None and args.min_effective < 1.0:",
        "if False:",
    ),
    (
        "uncertainty: bootstrap draws unseeded, report stops being reproducible",
        "uncertainty.py",
        "rng = random.Random(BOOTSTRAP_SEED)",
        "rng = random.Random()",
    ),
    (
        "uncertainty: resampled duplicates collapse instead of staying distinct",
        "uncertainty.py",
        'key = f"{item}\\x00{position}"',
        "key = item",
    ),
    (
        "uncertainty: interval endpoints taken from the wrong tail",
        "uncertainty.py",
        "        low=_percentile(ordered, tail),\n        high=_percentile(ordered, 1.0 - tail),",
        "        low=_percentile(ordered, 1.0 - tail),\n        high=_percentile(ordered, tail),",
    ),
    (
        "uncertainty: draws not sorted before taking percentiles",
        "uncertainty.py",
        "    ordered = sorted(draws)",
        "    ordered = list(draws)",
    ),
    (
        "gate: threshold inside its own interval is not called out",
        "report.py",
        "        if report.intervals is not None:\n            straddles: list[str] = []",
        "        if False:\n            straddles: list[str] = []",
    ),
    (
        "cli: a gate no longer implies intervals",
        "cli.py",
        "        intervals=args.intervals or gated,",
        "        intervals=args.intervals,",
    ),
    (
        "report: a p below the print resolution renders as zero",
        "report.py",
        'shown = f"{coin.p_value:.3f}" if coin.p_value >= 0.0005 else "<0.001"',
        'shown = f"{coin.p_value:.3f}"',
    ),
    (
        "gate: an undefined coefficient still meets a zero threshold",
        "report.py",
        "        if self.threshold is not None and not self.coefficients_defined:",
        "        if False:",
    ),
    (
        "gate: coefficients_defined never computed from the interpretation",
        "report.py",
        "        coefficients_defined=not (",
        "        coefficients_defined=bool(",
    ),
    (
        "agreement: mean pairwise averages pairs that were never compared",
        "agreement.py",
        "        if res.n == 0:\n            continue",
        "        if False:\n            continue",
    ),
    (
        "agreement: a rater with no comparable pair reported as 0.0",
        "agreement.py",
        "return {name: (sum(v) / len(v) if v else None) for name, v in sums.items()}",
        "return {name: (sum(v) / len(v) if v else 0.0) for name, v in sums.items()}",
    ),
    (
        "validity: precision counts calls on unadjudicated items",
        "validity.py",
        "        if lbl != positive_label or item not in truth:",
        "        if lbl != positive_label:",
    ),
    (
        "validity: accuracy ignores the label set",
        "validity.py",
        "        if label_set is not None and truth_label not in label_set:\n            continue",
        "        if False:\n            continue",
    ),
    (
        "io: valid JSON of the wrong shape reaches obj.get",
        "io.py",
        "            if not isinstance(obj, dict):",
        "            if False:",
    ),
    (
        "io: a scalar verdicts key is iterated",
        "io.py",
        "    verdicts = raw if isinstance(raw, list) else []",
        "    verdicts = raw",
    ),
    (
        "independence: permutations count not validated",
        "independence.py",
        "    if permutations < 1:",
        "    if False:",
    ),
    (
        "validity: no truth positives reported as zero recall",
        "types.py",
        "return self.caught / self.truth_positives if self.truth_positives else None",
        "return self.caught / self.truth_positives if self.truth_positives else 0.0",
    ),
    (
        "validity: no calls made reported as zero precision",
        "types.py",
        "return self.correct_calls / self.called if self.called else None",
        "return self.correct_calls / self.called if self.called else 0.0",
    ),
    (
        "validity: rater name dropped on the way out",
        "validity.py",
        "        rater=rater,",
        '        rater="",',
    ),
    (
        "report: positive label outside the set not rejected",
        "report.py",
        "    if positive_label not in label_tuple:",
        "    if False:",
    ),
    (
        "consensus: unvoted item reported as contested",
        "consensus.py",
        "        if voters == 0:",
        "        if False:",
    ),
    (
        "agreement: alpha reports near perfect on no data",
        "agreement.py",
        "            n=used_items, raters=len(rater_list), value=0.0, interpretation=UNDEFINED",
        "            n=used_items, raters=len(rater_list), value=1.0, interpretation=UNDEFINED",
    ),
    (
        "agreement: degenerate result banded as a measurement",
        "agreement.py",
        "n=0, raters=len(rater_list), value=0.0, interpretation=UNDEFINED",
        "n=0, raters=len(rater_list), value=0.0, interpretation=interpret_kappa(0.0)",
    ),
    (
        "triage: abstention invented for a label set without one",
        "triage.py",
        "        if ABSTENTION_LABEL in allowed and labeled:",
        "        if True:",
    ),
    (
        "triage: leaning invented for a label set without TP/FP",
        "triage.py",
        "interpretable = POSITIVE_LABEL in allowed and NEGATIVE_LABEL in allowed",
        "interpretable = True",
    ),
    (
        "io: truth file that yielded nothing does not warn",
        "io.py",
        "        if not truth:",
        "        if False:",
    ),
    (
        "cli: input errors traceback instead of exiting 2",
        "cli.py",
        "    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:",
        "    except FileNotFoundError as exc:",
    ),
    (
        "cli: missing panel raises instead of exiting 2",
        "cli.py",
        "    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:",
        "    except ZeroDivisionError as exc:",
    ),
]


def _purge_pycache(root: Path) -> None:
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def _suite_passes(root: Path) -> bool:
    _purge_pycache(root)
    # PYTHONPATH, not just cwd. judgecheck is installed editable, so a bare
    # `import judgecheck` inside the sandbox resolves to the *real* src tree and
    # every mutant would survive while looking like it had been tested.
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode == 0


#: Copied into the sandbox. Everything the suite imports or reads, and nothing
#: else -- no .venv, no .git, no caches.
_COPY = ("src", "tests", "scripts", "pyproject.toml", "README.md")


def _make_sandbox(dest: Path) -> Path:
    """Copy the parts of the repo the suite needs into a scratch directory.

    The sweep used to mutate `src/` in the real working tree and restore it in a
    `finally`. That is one SIGKILL away from leaving a stranger's checkout
    silently wrong, and two concurrent runs interleave: one restores a file
    while the other is measuring it, and mutants report SURVIVED without ever
    having been tested. Both of those happened. Working on a copy makes the
    failure mode impossible rather than unlikely.
    """
    for name in _COPY:
        source = ROOT / name
        if not source.exists():
            continue
        target = dest / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, target)
    return dest


def main() -> int:
    survivors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="judgecheck-mutation-") as tmp:
        sandbox = _make_sandbox(Path(tmp))
        sandbox_src = sandbox / "src" / "judgecheck"
        pristine = {p.name: p.read_text(encoding="utf-8") for p in sandbox_src.glob("*.py")}

        if not _suite_passes(sandbox):
            print("baseline is RED -- fix the suite before mutating", file=sys.stderr)
            return 2
        print(f"baseline green in sandbox, {len(MUTANTS)} mutants\n")

        try:
            for label, filename, find, replace in MUTANTS:
                for name, text in pristine.items():
                    (sandbox_src / name).write_text(text, encoding="utf-8")
                original = pristine.get(filename)
                if original is None or find not in original:
                    print(f"  SKIP      {label} (pattern not found -- has the code moved?)")
                    survivors.append(f"{label} [pattern not found]")
                    continue
                (sandbox_src / filename).write_text(
                    original.replace(find, replace), encoding="utf-8"
                )
                if _suite_passes(sandbox):
                    survivors.append(label)
                    print(f"  SURVIVED  {label}")
                else:
                    print(f"  caught    {label}")
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130

    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    if survivors:
        print("\nsurvivors:")
        for name in survivors:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
