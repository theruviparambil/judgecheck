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
from judgecheck.report import PanelReport, check_gate
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
def report() -> PanelReport:
    return build_report(load_panel(PANEL_DIR))


def _find(pattern: str) -> re.Match[str]:
    """Match a claim in the README, failing loudly if the claim vanished."""
    m = re.search(pattern, README)
    assert m is not None, f"README no longer contains a claim matching: {pattern}"
    return m


class TestHeadlineNumbers:
    def test_full_precision_values_are_exact(self, report: PanelReport) -> None:
        _find(r"`0\.13541263908579276` and `0\.14078274691755766`")
        assert repr(report.fleiss.value) == "0.13541263908579276"
        assert repr(report.krippendorff.value) == "0.14078274691755766"

    def test_the_displayed_rounding_matches(self, report: PanelReport) -> None:
        m = _find(r"Fleiss' kappa\s+([+-]\d\.\d{3})\s+\((\w+)\)\s+n=(\d+), raters=(\d+)")
        assert m.group(1) == f"{report.fleiss.value:+.3f}"
        assert m.group(2) == report.fleiss.interpretation
        assert int(m.group(3)) == report.fleiss.n
        assert int(m.group(4)) == report.fleiss.raters

        k = _find(r"Krippendorff's alpha\s+([+-]\d\.\d{3})\s+\((\w+)\)")
        assert k.group(1) == f"{report.krippendorff.value:+.3f}"
        assert k.group(2) == report.krippendorff.interpretation


class TestReproducedCountsTable:
    def test_pairwise_count(self, report: PanelReport) -> None:
        m = _find(r"\| Pairwise Cohen's κ \| (\d+) \(every model pair\) \|")
        assert int(m.group(1)) == len(report.pairwise)

    def test_recall_and_precision_count(self, report: PanelReport) -> None:
        m = _find(r"\| Per-rater recall and precision \| (\d+) \|")
        assert report.validity is not None
        assert int(m.group(1)) == 2 * len(report.validity)

    def test_mean_pairwise_count(self, report: PanelReport) -> None:
        m = _find(r"\| Mean pairwise κ \| (\d+) \|")
        assert int(m.group(1)) == len(report.mean_pairwise)

    def test_consensus_count(self, report: PanelReport) -> None:
        m = _find(r"\| Consensus classification per finding \| (\d+) \|")
        assert int(m.group(1)) == len(report.consensus)

    def test_panel_shape_in_prose(self, report: PanelReport) -> None:
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
    def test_rater_table_row(self, report: PanelReport, rater: str) -> None:
        row = _find(rf"\| {rater} \| (\d+)% \| (\d+)/(\d+) \| ([\d.]+) \| ([^|]+)\|")
        t = report.triage[rater]
        assert report.validity is not None
        v = report.validity[rater]

        assert t.agreement_with_truth is not None
        assert int(row.group(1)) == round(100 * t.agreement_with_truth), "accuracy column"
        assert int(row.group(2)) == v.caught, "recall numerator"
        assert int(row.group(3)) == v.truth_positives, "recall denominator"
        mean_k = report.mean_pairwise[rater]
        assert mean_k is not None, f"{rater}: no comparable pair"
        assert float(row.group(4)) == round(mean_k, 3), "mean pairwise"

        # label mix, e.g. "18 TP, 1 FP, 4 NI"
        short = {"TP": "TP", "FP": "FP", "NEEDS_INVESTIGATION": "NI"}
        expected = ", ".join(
            f"{t.distribution[lbl]} {short[lbl]}" for lbl in short if t.distribution[lbl]
        )
        assert row.group(5).strip() == expected, "label mix"

    def test_glm_skew_percentage_and_recommendation(self, report: PanelReport) -> None:
        m = _find(r"It also says TP to (\d+)% of everything it sees")
        t = report.triage["glm"]
        assert int(m.group(1)) == round(100 * t.distribution["TP"] / t.labeled)
        assert any(f.startswith("SKEWED") for f in t.flags)
        assert t.recommendation == "REVIEW"

    def test_pairwise_kappa_range(self, report: PanelReport) -> None:
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
    def test_split_count(self, report: PanelReport) -> None:
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
    def test_the_split_threshold_example_is_real(self, report: PanelReport) -> None:
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

    def test_the_shown_failure_output_is_what_the_gate_prints(self, report: PanelReport) -> None:
        gate = check_gate(report, 0.6)
        for line in gate.failures:
            assert f"FAIL  {line}" in README, f"README gate block is stale: {line}"


class TestLibrarySnippet:
    def test_fleiss_comment_value(self, report: PanelReport) -> None:
        m = _find(r"fleiss_kappa\(panel\.raters\)\.value  # ([\d.]+)")
        assert m.group(1) == repr(report.fleiss.value)

    def test_consensus_counts_comment(self, report: PanelReport) -> None:
        m = _find(r"report\.consensus_counts  # (\{[^}]+\})")
        assert m.group(1) == str(report.consensus_counts)

    def test_cohens_kappa_call_is_valid(self, report: PanelReport) -> None:
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
        m = _find(r"pytest\s+\d+ tests on ([\d., ]+)\n")
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
    def test_stated_crossvalidation_count_is_current(self) -> None:
        """This number drifted from 4 to 7 unnoticed, so it gets pinned too."""
        pytest.importorskip("statsmodels", reason="the stated count includes the crossval tests")
        m = _find(r"(\d+) of the \d+ are the cross-validation tests")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_crossvalidation.py",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        found = sum(int(n) for n in re.findall(r"^\S+: (\d+)$", proc.stdout, re.M))
        assert int(m.group(1)) == found, f"README says {m.group(1)}, collection found {found}"

    def test_stated_test_count_is_current(self) -> None:
        """Runs collection in a subprocess; the count includes this file.

        The README states the full-suite count, which includes the four
        cross-validation tests. Those are not collected at all when the
        `crossval` extra is absent, so without it this check would compare
        against a smaller suite and fail for the wrong reason.
        """
        pytest.importorskip("statsmodels", reason="the stated count includes the crossval tests")
        m = _find(r"pytest\s+(\d+) tests on")
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


class TestIndependenceClaims:
    """Numbers in the "How many judges do you actually have?" section.

    Several are quoted twice, once in a rendered sample block and once in the
    prose around it, so both copies are checked against the computed value.
    """

    def test_both_effective_judge_estimators(self, report: PanelReport) -> None:
        """Both are printed, so both are pinned."""
        ind = report.independence
        assert ind is not None
        assert ind.effective_raters is not None and ind.effective_raters_eigen is not None
        kish = _find(r"effective judges, Kish\s+(\d+\.\d+) of (\d+)")
        eigen = _find(r"effective judges, eigen\s+(\d+\.\d+) of (\d+)")
        assert float(kish.group(1)) == round(ind.effective_raters, 2)
        assert float(eigen.group(1)) == round(ind.effective_raters_eigen, 2)
        assert int(kish.group(2)) == int(eigen.group(2)) == ind.raters

    def test_the_reported_figure_is_the_conservative_one(self, report: PanelReport) -> None:
        ind = report.independence
        assert ind is not None and ind.effective is not None
        m = _find(r"reported\s+(\d+\.\d+) of (\d+)")
        assert float(m.group(1)) == round(ind.effective, 2)
        assert ind.effective == min(ind.effective_raters or 0, ind.effective_raters_eigen or 0)

    def test_the_claimed_disagreement_factor(self, report: PanelReport) -> None:
        """ "they disagree by a factor of three" has to stay true."""
        ind = report.independence
        assert ind is not None
        assert ind.effective_raters is not None and ind.effective_raters_eigen is not None
        _find(r"they disagree by a factor of three")
        ratio = ind.effective_raters / ind.effective_raters_eigen
        assert 2.5 <= ratio <= 3.5, ratio

    def test_the_claimed_phi_range(self, report: PanelReport) -> None:
        from judgecheck.independence import _error_vectors, _phi

        m = _find(r"run from \*\*(-\d+\.\d+) to\s*\n(\+\d+\.\d+)\*\*")
        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        names = sorted(panel.raters)
        values = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ea, eb = _error_vectors(
                    panel.raters[names[i]],
                    panel.raters[names[j]],
                    panel.truth,
                    frozenset(report.labels),
                )
                v = _phi(ea, eb)
                if v is not None:
                    values.append(v)
        assert round(min(values), 2) == float(m.group(1))
        assert round(max(values), 2) == float(m.group(2))

    def test_mean_error_correlation_sd_and_pair_count(self, report: PanelReport) -> None:
        m = _find(r"mean error correlation\s+([-+]\d+\.\d+)\s+\(sd (\d+\.\d+)\)\s+over (\d+) pairs")
        ind = report.independence
        assert ind is not None and ind.mean_phi is not None and ind.phi_sd is not None
        assert float(m.group(1)) == round(ind.mean_phi, 3)
        assert float(m.group(2)) == round(ind.phi_sd, 3)
        assert int(m.group(3)) == ind.pairs

    def test_stated_efficiency_percentage(self, report: PanelReport) -> None:
        m = _find(r"\((\d+)% of nominal, (not exchangeable \(estimators disagree\))\)")
        ind = report.independence
        assert ind is not None and ind.efficiency is not None
        assert int(m.group(1)) == round(ind.efficiency * 100)
        assert m.group(2) == ind.interpretation

    def test_the_claimed_agreement_disagrees_with_independence(self, report: PanelReport) -> None:
        """The gating section leans on these two pointing opposite ways."""
        m = _find(
            r"poor agreement \(κ (\d+\.\d+)\) and good independence\s*\n?\((\d+\.\d+) of (\d+)\)"
        )
        assert report.independence is not None
        assert report.independence.effective_raters is not None
        assert float(m.group(1)) == round(report.fleiss.value, 3)
        assert float(m.group(2)) == round(report.independence.effective_raters, 2)

    def test_no_within_family_pairs(self, report: PanelReport) -> None:
        assert report.groups is not None
        assert report.groups.within_pairs == 0
        _find(r"has \*\*zero\*\* within-family pairs")

    def test_worst_coincident_pair(self, report: PanelReport) -> None:
        m = _find(r"worst pair\s+(\w+) \+ (\w+)\s+both wrong on (\d+)/(\d+), ([-+]\d+\.\d+) above")
        assert report.coincidence is not None
        w = report.coincidence.worst
        assert w is not None
        assert (m.group(1), m.group(2)) == (w.a, w.b)
        assert int(m.group(3)) == w.both_wrong
        assert int(m.group(4)) == w.n
        assert float(m.group(5)) == round(w.excess, 1)

    def test_the_permutation_p(self, report: PanelReport) -> None:
        m = _find(r"permutation p (\d+\.\d+)")
        assert report.coincidence is not None
        assert report.coincidence.p_value is not None
        assert float(m.group(1)) == round(report.coincidence.p_value, 3)

    def test_the_worst_pair_error_correlation(self, report: PanelReport) -> None:
        m = _find(r"\n  error correlation\s+([-+]\d+\.\d+)")
        assert report.coincidence is not None
        w = report.coincidence.worst
        assert w is not None and w.phi is not None
        assert float(m.group(1)) == round(w.phi, 3)

    def test_the_prose_repeats_the_worst_pair_names(self, report: PanelReport) -> None:
        m = _find(r"`(\w+)` called seven of them FP outright\s*\nand `(\w+)` abstained")
        assert report.coincidence is not None
        w = report.coincidence.worst
        assert w is not None
        assert {m.group(1), m.group(2)} == {w.a, w.b}

    def test_all_eleven_joint_errors_really_are_truth_positives(self, report: PanelReport) -> None:
        """The README's strongest sentence about that pair, so it gets checked."""
        m = _find(r"All (\w+) of those items are findings adjudicated \*\*TP\*\*")
        words = {"eleven": 11, "ten": 10, "twelve": 12}
        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        assert report.coincidence is not None
        w = report.coincidence.worst
        assert w is not None
        a, b = panel.raters[w.a], panel.raters[w.b]
        joint = [
            i for i in panel.truth if a.get(i) != panel.truth[i] and b.get(i) != panel.truth[i]
        ]
        assert words[m.group(1)] == len(joint) == w.both_wrong
        assert all(panel.truth[i] == "TP" for i in joint)

    def test_the_ratio_ranking_claim_about_the_ceiling(self, report: PanelReport) -> None:
        """ "the four highest-ratio pairs all sit exactly at their structural ceiling"."""
        m = _find(r"the (\w+) highest-ratio pairs all sit exactly at\s*\ntheir structural ceiling")
        words = {"three": 3, "four": 4, "five": 5}
        assert report.coincidence is not None
        rated = [p for p in report.coincidence.pairs if p.lift is not None]
        top = sorted(rated, key=lambda p: p.lift or 0.0, reverse=True)[: words[m.group(1)]]
        for pair in top:
            ceiling = pair.n / max(pair.a_wrong, pair.b_wrong)
            assert pair.lift == pytest.approx(ceiling)

    def test_kohli_panel_agreement_claim(self) -> None:
        """ "the two agree within 1% (2.18 against 2.16)" is a claim about the paper."""
        m = _find(r"agree within 1% \((\d+\.\d+) against (\d+\.\d+)\)")
        assert (m.group(1), m.group(2)) == ("2.18", "2.16")
        assert abs(2.18 - 2.16) / 2.18 < 0.02

    def test_the_documented_reductio_still_holds(self, report: PanelReport) -> None:
        """The README claims the old label-based metric inverted. Recompute it."""
        from itertools import combinations

        from judgecheck.agreement import cohens_kappa
        from judgecheck.independence import effective_raters

        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        perfect = {f"j{i}": dict(panel.truth) for i in range(7)}
        kappas = [
            cohens_kappa(perfect[a], perfect[b]).kappa for a, b in combinations(sorted(perfect), 2)
        ]
        old_style = effective_raters(7, sum(kappas) / len(kappas)) / 7
        m = _find(r"seven judges that all match truth exactly\s+->\s+(\d+)% of nominal")
        assert int(m.group(1)) == round(old_style * 100)


class TestCitedLiteratureClaims:
    """Numbers attributed to outside papers, pinned so they cannot drift.

    These cannot be computed, so the test only checks the README still states
    what was verified against the sources on 2026-08-19. If someone edits a
    figure, this fails and sends them back to the paper rather than letting a
    quietly wrong citation ship.
    """

    def test_kohli_correlation_figures(self) -> None:
        m = _find(
            r"φ = (\d+\.\d+) and φ = (\d+\.\d+), against a cross-family\s*\nmean of φ = (\d+\.\d+)"
        )
        assert (m.group(1), m.group(2), m.group(3)) == ("0.437", "0.435", "0.389")

    def test_kohli_stated_gap_is_the_difference_of_the_stated_figures(self) -> None:
        """The gap is against the mean of both same-family pairs, not just one.

        Stating 0.437 alone with a gap of 0.047 does not reconcile (that
        subtraction gives 0.048), which is how this test caught a wrong
        paraphrase of the source before it shipped.
        """
        m = _find(r"a gap of \*\*(\d+\.\d+)\*\*")
        assert float(m.group(1)) == pytest.approx((0.437 + 0.435) / 2 - 0.389, abs=5e-4)

    def test_kohli_one_per_family_made_independence_worse(self) -> None:
        m = _find(r"n_eff fell from (\d+\.\d+) to\s*\n?(\d+\.\d+)")
        assert (m.group(1), m.group(2)) == ("2.18", "1.93")
        assert float(m.group(2)) < float(m.group(1)), "the direction is the whole point"

    def test_the_kohli_quote_names_the_statistic_it_rules_out(self) -> None:
        """The correction rests on this quote, so its key terms are pinned."""
        assert "phi coefficient reduces to the Pearson" in README
        assert "conflate prevalence with dependence" in README
        assert "Cohen's kappa" in README

    def test_both_papers_are_linked_by_arxiv_id(self) -> None:
        assert "arxiv.org/abs/2605.29800" in README
        assert "arxiv.org/abs/2502.04313" in README
