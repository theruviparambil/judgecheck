"""The same panel must produce bit-identical numbers on every run.

Not a style preference. The panel statistics accumulate floats while iterating
the set of item ids, Python randomizes string hashing per process, and float
addition is not associative -- so before `agreement.py` sorted its item
traversal, Fleiss' kappa moved in its last two digits depending on
`PYTHONHASHSEED`. A package whose claim is exact reproduction of a reference
implementation cannot publish a number that changes between runs.

The seed is fixed for the life of a process, so this can only be tested by
spawning subprocesses. That makes these tests slower than the rest of the
suite; they earn it by covering a failure no in-process test can see.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "panel-real"

SEEDS = ("0", "1", "7", "42", "12345")

PROBE = """
import hashlib, json, sys
from judgecheck.io import load_panel
from judgecheck.report import build_report, check_gate, to_json

panel = load_panel(sys.argv[1])

# intervals=True on purpose. The report now contains two Monte Carlo
# procedures, a 1000-draw bootstrap and a 2000-draw permutation test, and both
# are the obvious way to break a bit-identical guarantee. `uncertainty.py`
# cites this file as the reason its seed is fixed, and until this line the
# probe ran with intervals off, so the citation described a test that did not
# exist.
report = build_report(panel, intervals=True)

# The entire report, not a hand-picked subset: triage carries floats too, and a
# probe that only checked the headline numbers would have missed them.
payload = to_json(report, check_gate(report, 0.2, 2.0))
print(json.dumps({
    "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    "fleiss": repr(report.fleiss.value),
    "krippendorff": repr(report.krippendorff.value),
    "permutation_p": repr(report.coincidence.p_value if report.coincidence else None),
    "interval_low": repr(
        report.intervals.effective_raters.low
        if report.intervals and report.intervals.effective_raters
        else None
    ),
    "bytes": len(payload),
}, sort_keys=True))
"""


def _run(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(DATA)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.slow
def test_the_entire_report_is_byte_identical_across_hash_seeds() -> None:
    baseline = _run(SEEDS[0])
    assert baseline, "probe produced no output"
    for seed in SEEDS[1:]:
        assert _run(seed) == baseline, f"report changed under PYTHONHASHSEED={seed}"


@pytest.mark.slow
def test_the_published_fleiss_value_is_exact_not_approximate() -> None:
    """The full float repr, so a last-digit drift fails rather than rounds away."""
    import json

    for seed in SEEDS:
        got = json.loads(_run(seed))
        assert got["fleiss"] == "0.13541263908579276"
        assert got["krippendorff"] == "0.14078274691755766"
