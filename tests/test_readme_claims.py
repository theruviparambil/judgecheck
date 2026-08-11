"""Every number in the README, checked against the code that produces it.

This file exists because a README is the one artifact nobody runs. Its numbers
were verified once by hand, and hand-verification is exactly what let a wrong
claim ship: an assertion about the reference implementation that sounded right,
was never executed, and was false.

So the claims are parsed out of README.md and asserted here. Each test also
asserts that its pattern was *found*, so rewording the README into a state
where a claim silently disappears fails loudly instead of passing vacuously.

Not everything is checkable this way. Prose about what a statistic means, or
about who wrote what, still needs a human. What is mechanical is pinned here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from judgecheck import build_report, load_panel
from judgecheck.agreement import cohens_kappa
from judgecheck.report import check_gate
from judgecheck.triage import (
    ABSTENTION_THRESHOLD,
    ACCURACY_THRESHOLD,
    REDUNDANT_KAPPA,
    SKEW_THRESHOLD,
)

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PANEL_DIR = Path(__file__).parent / "data" / "panel-real"


@pytest.fixture(scope="module")
def report():  # type: ignore[no-untyped-def]
    return build_report(load_panel(PANEL_DIR))


def _find(pattern: str) -> re.Match[str]:
    """Match a claim in the README, failing loudly if the claim vanished."""
    m = re.search(pattern, README)
    assert m is not None, f"README no longer contains a claim matching: {pattern}"
    return m


class TestHeadlineNumbers:
    def test_full_precision_values_are_exact(self, report) -> None:  # type: ignore[no-untyped-def]
        _find(r"`0\.13541263908579276` and `0\.14078274691755766`")
        assert repr(report.fleiss.value) == "0.13541263908579276"
        assert repr(report.krippendorff.value) == "0.14078274691755766"

    def test_the_displayed_rounding_matches(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"Fleiss' kappa\s+([+-]\d\.\d{3})\s+\((\w+)\)\s+n=(\d+), raters=(\d+)")
        assert m.group(1) == f"{report.fleiss.value:+.3f}"
        assert m.group(2) == report.fleiss.interpretation
        assert int(m.group(3)) == report.fleiss.n
        assert int(m.group(4)) == report.fleiss.raters

        k = _find(r"Krippendorff's alpha\s+([+-]\d\.\d{3})\s+\((\w+)\)")
        assert k.group(1) == f"{report.krippendorff.value:+.3f}"
        assert k.group(2) == report.krippendorff.interpretation


class TestReproducedCountsTable:
    def test_pairwise_count(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"\| Pairwise Cohen's κ \| (\d+) \(every model pair\) \|")
        assert int(m.group(1)) == len(report.pairwise)

    def test_recall_and_precision_count(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"\| Per-rater recall and precision \| (\d+) \|")
        assert report.validity is not None
        assert int(m.group(1)) == 2 * len(report.validity)

    def test_mean_pairwise_count(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"\| Mean pairwise κ \| (\d+) \|")
        assert int(m.group(1)) == len(report.mean_pairwise)

    def test_consensus_count(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"\| Consensus classification per finding \| (\d+) \|")
        assert int(m.group(1)) == len(report.consensus)

    def test_panel_shape_in_prose(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"The (\w+)-model panel in `tests/data/panel-real/`")
        assert m.group(1) == "seven"
        assert len(report.raters) == 7


class TestWhyKappaSection:
    def test_truth_composition(self) -> None:
        m = _find(r"truth is (\d+) TP, (\d+) NEEDS_INVESTIGATION, (\d+) FP")
        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        counts = {
            lbl: sum(1 for v in panel.truth.values() if v == lbl)
            for lbl in set(panel.truth.values())
        }
        assert int(m.group(1)) == counts["TP"]
        assert int(m.group(2)) == counts["NEEDS_INVESTIGATION"]
        assert int(m.group(3)) == counts["FP"]

    def test_always_tp_baseline_accuracy(self) -> None:
        m = _find(r"to everything scores (\d+)% accuracy")
        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        tp = sum(1 for v in panel.truth.values() if v == "TP")
        assert int(m.group(1)) == round(100 * tp / len(panel.truth))

    @pytest.mark.parametrize("rater", ["gemini", "glm", "grok"])
    def test_rater_table_row(self, report, rater: str) -> None:  # type: ignore[no-untyped-def]
        row = _find(rf"\| {rater} \| (\d+)% \| (\d+)/(\d+) \| ([\d.]+) \| ([^|]+)\|")
        t = report.triage[rater]
        assert report.validity is not None
        v = report.validity[rater]

        assert t.agreement_with_truth is not None
        assert int(row.group(1)) == round(100 * t.agreement_with_truth), "accuracy column"
        assert int(row.group(2)) == v.caught, "recall numerator"
        assert int(row.group(3)) == v.truth_positives, "recall denominator"
        assert float(row.group(4)) == round(report.mean_pairwise[rater], 3), "mean pairwise"

        # label mix, e.g. "18 TP, 1 FP, 4 NI"
        short = {"TP": "TP", "FP": "FP", "NEEDS_INVESTIGATION": "NI"}
        expected = ", ".join(
            f"{t.distribution[lbl]} {short[lbl]}" for lbl in short if t.distribution[lbl]
        )
        assert row.group(5).strip() == expected, "label mix"

    def test_glm_skew_percentage_and_recommendation(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"It also says TP to (\d+)% of everything it sees")
        t = report.triage["glm"]
        assert int(m.group(1)) == round(100 * t.distribution["TP"] / t.labeled)
        assert any(f.startswith("SKEWED") for f in t.flags)
        assert t.recommendation == "REVIEW"

    def test_pairwise_kappa_range(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"κ ranges from (-?\d+\.\d+) to (-?\d+\.\d+)")
        kappas = [r.kappa for r in report.pairwise.values()]
        assert float(m.group(1)) == round(min(kappas), 3)
        assert float(m.group(2)) == round(max(kappas), 3)

    def test_interpretation_bands_are_described_accurately(self) -> None:
        """The README claims a specific deviation from Landis and Koch."""
        _find(r"their `slight` band \(0\.00 to 0\.20\) is folded into `poor`")
        from judgecheck.agreement import interpret_kappa

        assert interpret_kappa(0.0) == "poor"
        assert interpret_kappa(0.19) == "poor", "0.00-0.20 is folded into poor, not 'slight'"
        assert interpret_kappa(0.2) == "fair"
        assert interpret_kappa(0.95) == "near perfect", "renamed from 'almost perfect'"


class TestReportSectionClaims:
    def test_split_count(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"This panel has (\d+)\.")
        assert int(m.group(1)) == len(report.splits)

    def test_triage_thresholds_match_the_constants(self) -> None:
        m = _find(
            r"being skewed \(>(\d+)% one label\), abstaining \(>(\d+)% NI\), "
            r"diverging from truth \(<(\d+)%\s*\n?\s*exact match\), or being redundant with "
            r"another rater \(pairwise κ >= ([\d.]+)\)"
        )
        assert int(m.group(1)) == round(100 * SKEW_THRESHOLD)
        assert int(m.group(2)) == round(100 * ABSTENTION_THRESHOLD)
        assert int(m.group(3)) == round(100 * ACCURACY_THRESHOLD)
        assert float(m.group(4)) == REDUNDANT_KAPPA


class TestGateSectionClaims:
    def test_the_split_threshold_example_is_real(self, report) -> None:  # type: ignore[no-untyped-def]
        """README claims 0.14 fails Fleiss at 0.1354 while Krippendorff 0.1408 clears."""
        m = _find(
            r"`--fail-under (\d\.\d+)` fails on Fleiss\s*\n?\s*\(([\d.]+)\) while "
            r"Krippendorff \(([\d.]+)\) clears it"
        )
        threshold = float(m.group(1))
        assert round(report.fleiss.value, 4) == float(m.group(2))
        assert round(report.krippendorff.value, 4) == float(m.group(3))

        gate = check_gate(report, threshold)
        assert not gate.passed
        assert len(gate.failures) == 1
        assert "Fleiss" in gate.failures[0]

    def test_the_shown_failure_output_is_what_the_gate_prints(self, report) -> None:  # type: ignore[no-untyped-def]
        gate = check_gate(report, 0.6)
        for line in gate.failures:
            assert f"FAIL  {line}" in README, f"README gate block is stale: {line}"


class TestLibrarySnippet:
    def test_fleiss_comment_value(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"fleiss_kappa\(panel\.raters\)\.value  # ([\d.]+)")
        assert m.group(1) == repr(report.fleiss.value)

    def test_consensus_counts_comment(self, report) -> None:  # type: ignore[no-untyped-def]
        m = _find(r"report\.consensus_counts  # (\{[^}]+\})")
        assert m.group(1) == str(report.consensus_counts)

    def test_cohens_kappa_call_is_valid(self, report) -> None:  # type: ignore[no-untyped-def]
        panel = load_panel(PANEL_DIR)
        assert cohens_kappa(panel.raters["claude"], panel.raters["gpt"]).n > 0


class TestVerificationBlockClaims:
    def test_mutant_count_matches_the_sweep(self) -> None:
        m = _find(r"mutation sweep  (\d+)/(\d+) mutants killed")
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import mutation_sweep
        finally:
            sys.path.pop(0)
        assert int(m.group(2)) == len(mutation_sweep.MUTANTS)
        assert m.group(1) == m.group(2), "README claims a clean sweep"

    def test_python_versions_match_pyproject(self) -> None:
        m = _find(r"pytest\s+\d+ passed on ([\d., ]+)\n")
        stated = {v.strip() for v in m.group(1).split(",")}
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        classified = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject))
        assert stated == classified, "README version list and pyproject classifiers disagree"

    def test_requirements_floor_matches_pyproject(self) -> None:
        m = _find(r"Python (\d+\.\d+)\+, tested on")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        floor = re.search(r'requires-python = ">=(\d+\.\d+)"', pyproject)
        assert floor is not None
        assert m.group(1) == floor.group(1)

    @pytest.mark.slow
    def test_stated_test_count_is_current(self) -> None:
        """Runs collection in a subprocess; the count includes this file."""
        m = _find(r"pytest\s+(\d+) passed on")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        total = sum(int(n) for n in re.findall(r"^\S+: (\d+)$", proc.stdout, re.M))
        assert total > 0, proc.stdout
        assert int(m.group(1)) == total, f"README says {m.group(1)} tests, collection found {total}"
