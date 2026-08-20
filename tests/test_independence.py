"""Panel independence, coincident error, and group agreement.

`TestTheMetricIsNotBackwards` is the most important class here. An earlier
version of this module computed the effective judge count from mean pairwise
Cohen's kappa on *labels*, which is maximized by a panel that shares nothing
and minimized by a panel that is uniformly correct. It passed 270 tests, a
54-mutant sweep, and a citation check before an outside reviewer pointed at it.
Those tests exist so that the inversion cannot come back quietly.

The rest are guard tests. Every `None` in this module's return types is load
bearing: it marks a quantity that is undefined rather than zero. The whole
reason the feature is shaped this way is that returning 0.0 for "undefined"
produces a confident and false statement, and in one case let a panel with no
overlapping items at all pass an independence gate.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from judgecheck.agreement import UNDEFINED, UNDEFINED_NO_VARIANCE, cohens_kappa
from judgecheck.cli import main
from judgecheck.independence import (
    CAUTION_EFFICIENCY,
    NULL_DRAWS,
    PERMUTATIONS,
    _largest_eigenvalue,
    _permuted,
    coincident_errors,
    effective_raters,
    group_agreement,
    interpret_efficiency,
    panel_independence,
    rater_groups_from_panel,
)
from judgecheck.io import load_panel
from judgecheck.report import build_report, check_gate, to_dict
from judgecheck.types import Panel

PANEL = Path(__file__).parent / "data" / "panel-real"

TRUTH = {"i1": "TP", "i2": "TP", "i3": "FP", "i4": "FP", "i5": "TP", "i6": "FP"}

THREE_RATERS = {
    "a1": {"i1": "TP", "i2": "TP", "i3": "FP"},
    "a2": {"i1": "TP", "i2": "TP", "i3": "FP"},
    "b1": {"i1": "FP", "i2": "TP", "i3": "TP"},
}


@pytest.fixture(scope="module")
def panel() -> Panel:
    return load_panel(PANEL)


class TestTheMetricIsNotBackwards:
    """Regression guards against the label-agreement version of `n_eff`.

    Under the old implementation every one of these produced the opposite of
    the right answer. They are written as behavioural claims about panels
    rather than about kappa or phi, so they stay meaningful if the estimator
    is ever revised again.
    """

    def _labels(self) -> list[str]:
        return ["TP", "FP", "NEEDS_INVESTIGATION", "OUT_OF_SCOPE"]

    def test_identical_judges_are_worth_one_judge(self, panel: Panel) -> None:
        """Seven copies of one judge carry one judge's evidence."""
        assert panel.truth is not None
        clones = {f"c{i}": dict(panel.raters["qwen"]) for i in range(7)}
        ind = panel_independence(clones, panel.truth)
        assert ind.mean_phi == pytest.approx(1.0)
        assert ind.effective_raters == pytest.approx(1.0)
        assert ind.interpretation == "highly correlated"

    def test_a_uniformly_correct_panel_is_not_called_highly_correlated(self, panel: Panel) -> None:
        """The old metric scored this 14% of nominal. It is not a correlated panel.

        The honest answer is that error correlation is undefined when there are
        no errors, so it reports not-measurable rather than a passing or a
        failing number.
        """
        assert panel.truth is not None
        perfect = {f"j{i}": dict(panel.truth) for i in range(7)}
        ind = panel_independence(perfect, panel.truth)
        assert ind.interpretation == "not measurable"
        assert ind.effective_raters is None
        assert ind.incomparable_pairs == 21

    def test_a_rubber_stamp_rater_does_not_raise_measured_independence_much(
        self, panel: Panel
    ) -> None:
        """Under the old metric a constant rater bought most of an extra judge.

        A constant rater has kappa 0 against everyone, which the label version
        read as maximal independence. On errors it can no longer manufacture
        agreement-based independence out of contributing nothing.
        """
        assert panel.truth is not None
        base = panel_independence(panel.raters, panel.truth)
        stamped = dict(panel.raters)
        stamped["stamp"] = dict.fromkeys(panel.truth, "TP")
        after = panel_independence(stamped, panel.truth)
        assert base.efficiency is not None and after.efficiency is not None
        # It may still shift; what must not happen is the old behaviour, where a
        # zero-information rater made a correlated panel look less correlated.
        assert after.mean_phi is not None and base.mean_phi is not None
        assert after.mean_phi <= base.mean_phi + 0.05

    def test_n_eff_is_documented_as_orthogonal_to_quality(self, panel: Panel) -> None:
        """Independently-wrong judges score high, and that is correct.

        Condorcet needs independence *and* better-than-chance accuracy.
        judgecheck reports accuracy separately, so this test pins that the two
        are not conflated rather than pretending n_eff measures quality.
        """
        assert panel.truth is not None
        rng = random.Random(11)
        labels = self._labels()
        noise = {f"r{i}": {item: rng.choice(labels) for item in panel.truth} for i in range(7)}
        ind = panel_independence(noise, panel.truth)
        assert ind.efficiency is not None
        assert ind.efficiency > 0.4  # substantially independent...
        report = build_report(Panel(name="n", raters=noise, truth=panel.truth))
        assert report.validity is not None
        # f1 is optional now; a judge that never used the positive label has
        # no score rather than a zero, and either way none of them is any good.
        scores = [v.f1 for v in report.validity.values() if v.f1 is not None]
        assert scores, "at least one judge should have a computable score"
        assert max(scores) < 0.5  # ...and useless


class TestOnTheRealPanel:
    """The published numbers. Seven judges, seven developers, 23 items."""

    def test_mean_error_correlation(self, panel: Panel) -> None:
        assert panel.truth is not None
        phi = panel_independence(panel.raters, panel.truth).mean_phi
        assert phi is not None
        assert round(phi, 4) == -0.0127

    def test_effective_judges(self, panel: Panel) -> None:
        """Two estimators, and on this panel they disagree by 3.2x.

        Kish averages the pairwise correlations and assumes they are roughly
        equal. This panel's correlations run from -0.68 to +0.74, so that
        assumption fails and the eigenvalue form is the one reported.
        """
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth)
        assert (ind.raters, ind.pairs, ind.incomparable_pairs) == (7, 21, 0)
        assert ind.effective_raters == pytest.approx(7.0)
        assert ind.effective_raters_eigen == pytest.approx(2.19, abs=0.01)
        assert ind.effective == pytest.approx(2.19, abs=0.01)
        assert ind.efficiency == pytest.approx(0.31, abs=0.01)
        assert not ind.exchangeable
        assert ind.saturated
        assert ind.interpretation == "not exchangeable (estimators disagree)"

    def test_the_pairwise_correlations_are_wildly_unequal(self, panel: Panel) -> None:
        """Which is why a single averaged number cannot describe this panel."""
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth)
        assert ind.phi_sd is not None and ind.mean_phi is not None
        assert ind.phi_sd > 0.35
        assert ind.phi_sd > 20 * abs(ind.mean_phi)

    def test_every_rater_is_a_different_developer(self, panel: Panel) -> None:
        """The premise of the group section. If this fails, the README is wrong."""
        groups = rater_groups_from_panel(panel)
        assert len(groups) == 7
        assert len(set(groups.values())) == 7

    def test_there_are_no_within_family_pairs_to_compare(self, panel: Panel) -> None:
        """Which is why judgecheck does not ship a same-family/cross-family split."""
        g = group_agreement(panel.raters, rater_groups_from_panel(panel))
        assert (g.within_pairs, g.between_pairs) == (0, 21)
        assert g.within is None and g.delta is None
        assert g.between is not None
        assert round(g.between, 3) == 0.199

    def test_the_worst_pair_is_significant_and_cross_family(self, panel: Panel) -> None:
        """deepseek (DeepSeek) and grok (xAI): different developers, joint failure.

        Ranked by excess joint errors rather than by ratio, and reported with a
        permutation p that already accounts for selecting the worst of 21 pairs.
        """
        assert panel.truth is not None
        c = coincident_errors(panel.raters, panel.truth)
        assert c.worst is not None and c.p_value is not None
        assert (c.worst.a, c.worst.b) == ("deepseek", "grok")
        assert c.worst.both_wrong == 11
        assert round(c.worst.excess, 2) == 4.22
        assert c.p_value < 0.05

    def test_the_panel_as_a_whole_fails_roughly_independently(self, panel: Panel) -> None:
        """The counterweight to the worst pair, and it points the other way."""
        assert panel.truth is not None
        phi = coincident_errors(panel.raters, panel.truth).mean_phi
        assert phi is not None
        assert abs(phi) < 0.05

    def test_the_worst_pair_jointly_misses_real_findings(self, panel: Panel) -> None:
        """What the correlation actually consists of, pinned so the prose stays true."""
        assert panel.truth is not None
        d, g = panel.raters["deepseek"], panel.raters["grok"]
        joint = [
            i for i in panel.truth if d.get(i) != panel.truth[i] and g.get(i) != panel.truth[i]
        ]
        assert len(joint) == 11
        assert all(panel.truth[i] == "TP" for i in joint)

    def test_ranking_by_lift_would_pick_a_different_and_worse_pair(self, panel: Panel) -> None:
        """Lift is ceiling-limited, which is why the report does not sort on it.

        gemini/glm tops the lift ranking at exactly n / max(a_wrong, b_wrong),
        its structural maximum, so that ranking reports a property of the metric.
        """
        assert panel.truth is not None
        pairs = coincident_errors(panel.raters, panel.truth).pairs
        by_lift = max((p for p in pairs if p.lift is not None), key=lambda p: p.lift or 0.0)
        assert (by_lift.a, by_lift.b) == ("gemini", "glm")
        assert by_lift.lift == pytest.approx(by_lift.n / max(by_lift.a_wrong, by_lift.b_wrong))

    def test_the_highest_agreement_pair_agrees_mostly_by_being_right(self, panel: Panel) -> None:
        """Retired claim, kept as a test so it cannot be re-asserted.

        The README once said this pair's agreement was "substantially shared
        error rather than shared competence". Removing every joint error moves
        their kappa by less than 0.01.
        """
        from judgecheck.agreement import cohens_kappa

        assert panel.truth is not None
        t = panel.truth
        gem, glm = panel.raters["gemini"], panel.raters["glm"]
        both_right = sum(1 for i in t if gem.get(i) == t[i] and glm.get(i) == t[i])
        both_wrong = sum(1 for i in t if gem.get(i) != t[i] and glm.get(i) != t[i])
        assert both_right == 17
        assert both_wrong == 3
        keep = [i for i in t if not (gem.get(i) != t[i] and glm.get(i) != t[i])]
        full = cohens_kappa(gem, glm).kappa
        trimmed = cohens_kappa(
            {i: gem[i] for i in keep if i in gem}, {i: glm[i] for i in keep if i in glm}
        ).kappa
        assert abs(full - trimmed) < 0.01


class TestEffectiveRaters:
    def test_uncorrelated_judges_are_all_worth_counting(self) -> None:
        assert effective_raters(7, 0.0) == 7.0

    def test_perfectly_correlated_judges_are_worth_one(self) -> None:
        assert effective_raters(7, 1.0) == pytest.approx(1.0)

    def test_a_negative_mean_does_not_manufacture_extra_judges(self) -> None:
        assert effective_raters(7, -0.5) == 7.0

    def test_a_correlation_above_one_is_clamped(self) -> None:
        assert effective_raters(5, 1.5) == pytest.approx(1.0)

    def test_nan_does_not_slip_the_clamp(self) -> None:
        """NaN compares false against everything, so min/max would pass it through.

        It propagates rather than becoming `k`. Returning full independence for
        a failed computation would send it through `--min-effective` as a pass,
        which is the failure this package fixes everywhere else.
        """
        assert math.isnan(effective_raters(5, float("nan")))

    def test_zero_raters_perfectly_correlated_does_not_divide_by_zero(self) -> None:
        assert effective_raters(0, 1.0) == 0.0

    @pytest.mark.parametrize("k", [2, 3, 5, 7, 20])
    @pytest.mark.parametrize("rho", [-1.0, -0.3, 0.0, 0.05, 0.5, 0.9, 1.0])
    def test_n_eff_is_always_between_one_and_k(self, k: int, rho: float) -> None:
        n_eff = effective_raters(k, rho)
        assert 1.0 <= n_eff <= k
        assert math.isfinite(n_eff)


class TestInterpretEfficiency:
    @pytest.mark.parametrize(
        ("efficiency", "expected"),
        [
            (1.0, "near independent"),
            (0.8, "near independent"),
            (0.79, "moderately correlated"),
            (0.5, "moderately correlated"),
            (0.49, "highly correlated"),
            (0.0, "highly correlated"),
        ],
    )
    def test_bands(self, efficiency: float, expected: str) -> None:
        assert interpret_efficiency(efficiency) == expected


class TestNotMeasurable:
    """The failure mode that let a panel with no overlap pass a gate."""

    def test_disjoint_raters_are_not_independent_they_are_uncomparable(self) -> None:
        disjoint = {"a": {"i1": "TP"}, "b": {"i2": "FP"}, "c": {"i3": "TP"}}
        truth = {"i1": "TP", "i2": "TP", "i3": "FP"}
        ind = panel_independence(disjoint, truth)
        assert ind.interpretation == "not measurable"
        assert ind.effective_raters is None
        assert ind.efficiency is None
        assert ind.incomparable_pairs == 3

    def test_a_gate_fails_rather_than_passes_when_nothing_is_measurable(self) -> None:
        """The old behaviour was `n_eff 3.00 of 3, 100%` and a PASS."""
        disjoint = {"a": {"i1": "TP"}, "b": {"i2": "FP"}, "c": {"i3": "TP"}}
        truth = {"i1": "TP", "i2": "TP", "i3": "FP"}
        report = build_report(Panel(name="d", raters=disjoint, truth=truth))
        gate = check_gate(report, min_effective=3.0)
        assert not gate.passed
        assert "not measurable" in gate.failures[0]

    def test_a_single_rater_is_not_measurable(self) -> None:
        ind = panel_independence({"a": {"i1": "TP"}}, {"i1": "FP"})
        assert ind.mean_phi is None
        assert ind.pairs == 0

    def test_an_empty_panel_does_not_divide_by_zero(self) -> None:
        assert panel_independence({}, {}).raters == 0

    def test_a_panel_without_truth_reports_no_independence_at_all(self) -> None:
        """No error vectors means no honest number, so the section is absent."""
        report = build_report(Panel(name="n", raters=THREE_RATERS))
        assert report.independence is None
        assert report.coincidence is None
        assert "PANEL INDEPENDENCE" not in json.dumps(to_dict(report))

    def test_gating_independence_without_truth_fails_loudly(self) -> None:
        report = build_report(Panel(name="n", raters=THREE_RATERS))
        gate = check_gate(report, min_effective=2.0)
        assert not gate.passed
        assert "adjudicated truth" in gate.failures[0]


#: A bare sequence of raters, the shape `agreement` has always accepted.
SEQ_RATERS = [
    {"i1": "TP", "i2": "FP", "i3": "TP", "i4": "FP"},
    {"i1": "FP", "i2": "FP", "i3": "TP", "i4": "TP"},
]
SEQ_TRUTH = {"i1": "TP", "i2": "TP", "i3": "TP", "i4": "TP"}


class TestPanelShapeIsAcceptedEverywhere:
    """`types.PanelLabels` calls the mapping/sequence split a trap. It was one here."""

    def test_panel_independence_accepts_a_sequence(self) -> None:
        assert panel_independence(SEQ_RATERS, SEQ_TRUTH).raters == 2

    def test_coincident_errors_accepts_a_sequence(self) -> None:
        """This raised `TypeError: '<' not supported between instances of 'dict'`."""
        assert len(coincident_errors(SEQ_RATERS, SEQ_TRUTH).pairs) == 1

    def test_group_agreement_accepts_a_sequence(self) -> None:
        got = group_agreement(SEQ_RATERS, {"rater0": "A", "rater1": "A"})
        assert got.within_pairs == 1


class TestCoincidentError:
    def test_a_pair_that_is_never_wrong_has_no_defined_lift_or_phi(self) -> None:
        perfect = dict(TRUTH)
        c = coincident_errors({"a": perfect, "b": dict(perfect)}, TRUTH)
        assert c.pairs[0].lift is None
        assert c.pairs[0].phi is None
        assert c.mean_phi is None

    def test_identical_wrong_raters_correlate_perfectly(self) -> None:
        wrong = {"i1": "FP", "i2": "FP", "i3": "TP", "i4": "FP", "i5": "TP", "i6": "FP"}
        pc = coincident_errors({"a": wrong, "b": dict(wrong)}, TRUTH).pairs[0]
        assert pc.phi == pytest.approx(1.0)
        assert pc.excess > 0

    def test_excess_is_zero_when_joint_errors_match_independence(self) -> None:
        a = {"i1": "FP", "i2": "FP", "i3": "FP", "i4": "FP", "i5": "TP", "i6": "FP"}
        b = {"i1": "TP", "i2": "TP", "i3": "TP", "i4": "FP", "i5": "TP", "i6": "FP"}
        pc = coincident_errors({"a": a, "b": b}, TRUTH).pairs[0]
        assert pc.excess == pytest.approx(pc.both_wrong - pc.a_wrong * pc.b_wrong / pc.n)

    def test_no_overlap_yields_a_zero_row_not_a_crash(self) -> None:
        c = coincident_errors({"a": {"i1": "TP"}, "b": {"zzz": "TP"}}, TRUTH)
        assert c.pairs[0].n == 0
        assert c.pairs[0].lift is None
        assert c.worst is None

    def test_items_missing_from_truth_are_not_scored(self) -> None:
        raters = {"a": {"i1": "TP", "i9": "FP"}, "b": {"i1": "TP", "i9": "TP"}}
        assert coincident_errors(raters, {"i1": "TP"}).pairs[0].n == 1

    def test_out_of_set_labels_are_excluded(self) -> None:
        raters = {"a": {"i1": "BOGUS", "i2": "TP"}, "b": {"i1": "TP", "i2": "TP"}}
        assert coincident_errors(raters, TRUTH).pairs[0].n == 1

    def test_pairs_come_back_in_a_stable_order(self) -> None:
        raters = {"z": dict(TRUTH), "a": dict(TRUTH), "m": dict(TRUTH)}
        got = [(p.a, p.b) for p in coincident_errors(raters, TRUTH).pairs]
        assert got == [("a", "m"), ("a", "z"), ("m", "z")]

    def test_the_permutation_p_is_deterministic(self, panel: Panel) -> None:
        """The report is asserted bit-identical across hash seeds; this is inside it."""
        assert panel.truth is not None
        first = coincident_errors(panel.raters, panel.truth).p_value
        second = coincident_errors(panel.raters, panel.truth).p_value
        assert first == second

    def test_the_p_value_is_never_exactly_zero(self) -> None:
        """Add-one estimator: 2000 draws cannot support a claim of p = 0.

        Built so that no reshuffle can match the observed excess: two judges
        wrong on exactly the same half of the items. `hits` is 0, so a plain
        `hits / draws` would report p = 0.0, which asserts more than 2000 draws
        can establish. `(hits + 1) / (draws + 1)` reports the smallest value
        the experiment can actually support.
        """
        n = 40
        truth = {f"i{i}": "TP" for i in range(n)}
        wrong_half = {f"i{i}": ("FP" if i < n // 2 else "TP") for i in range(n)}
        c = coincident_errors({"a": wrong_half, "b": dict(wrong_half)}, truth)
        assert c.p_value is not None
        assert c.p_value > 0.0
        assert c.p_value == pytest.approx(1 / (PERMUTATIONS + 1))


class TestGroupAgreement:
    def test_within_and_between_are_split_correctly(self) -> None:
        g = group_agreement(THREE_RATERS, {"a1": "A", "a2": "A", "b1": "B"})
        assert (g.within_pairs, g.between_pairs) == (1, 2)
        assert g.within == pytest.approx(1.0)
        assert g.delta is not None

    def test_an_empty_side_is_none_rather_than_zero(self) -> None:
        g = group_agreement(THREE_RATERS, {"a1": "A", "a2": "B", "b1": "C"})
        assert g.within_pairs == 0
        assert g.within is None and g.delta is None

    def test_ungrouped_raters_are_listed_not_silently_dropped(self) -> None:
        g = group_agreement(THREE_RATERS, {"a1": "A", "a2": "A"})
        assert g.ungrouped == ("b1",)
        assert (g.within_pairs, g.between_pairs) == (1, 0)

    def test_pairs_with_no_shared_items_are_not_counted_as_agreement(self) -> None:
        """cohens_kappa returns 0.0 for an empty overlap; that is not a measurement."""
        raters = {"a": {"i1": "TP"}, "b": {"i2": "TP"}}
        g = group_agreement(raters, {"a": "G", "b": "G"})
        assert g.within_pairs == 0
        assert g.within is None

    def test_groups_and_members_are_sorted(self) -> None:
        g = group_agreement(THREE_RATERS, {"b1": "Z", "a2": "A", "a1": "A"})
        assert list(g.groups) == ["A", "Z"]
        assert g.groups["A"] == ("a1", "a2")


class TestGroupsFromPanel:
    def test_reads_the_vendor_field(self, panel: Panel) -> None:
        assert rater_groups_from_panel(panel)["claude"] == "Anthropic"

    def test_a_panel_without_judgments_yields_nothing(self) -> None:
        assert rater_groups_from_panel(Panel(name="p", raters={"a": {}})) == {}


class TestReportIntegration:
    def test_the_report_carries_independence(self, panel: Panel) -> None:
        r = build_report(panel)
        assert r.independence is not None
        assert r.independence.effective == pytest.approx(2.19, abs=0.01)
        assert r.coincidence is not None and r.groups is not None

    def test_a_supplied_grouping_overrides_the_vendor_default(self, panel: Panel) -> None:
        r = build_report(panel, groups={"gemini": "X", "glm": "X", "gpt": "Y", "claude": "Y"})
        assert r.groups is not None
        assert r.groups.within_pairs == 2
        assert set(r.groups.ungrouped) == {"deepseek", "grok", "qwen"}

    def test_none_survives_json_as_null(self, panel: Panel) -> None:
        d = json.loads(json.dumps(to_dict(build_report(panel))))
        assert d["groups"]["withinKappa"] is None
        assert d["groups"]["delta"] is None
        assert d["independence"]["meanPhi"] == pytest.approx(-0.0127, abs=1e-4)
        assert d["independence"]["basis"] == "error correlation vs adjudicated truth"

    def test_adding_independence_did_not_move_any_existing_statistic(self, panel: Panel) -> None:
        r = build_report(panel)
        assert round(r.fleiss.value, 4) == 0.1354
        assert round(r.krippendorff.value, 4) == 0.1408
        assert len(r.pairwise) == 21


class TestIndependenceGate:
    def test_a_panel_above_the_floor_passes(self, panel: Panel) -> None:
        assert check_gate(build_report(panel), min_effective=2.0).passed

    def test_the_gate_reads_the_conservative_estimator(self, panel: Panel) -> None:
        """Kish says 7.00, eigen says 2.19. A gate at 5 must not pass on Kish."""
        report = build_report(panel)
        assert report.independence is not None
        assert report.independence.effective_raters == pytest.approx(7.0)
        assert not check_gate(report, min_effective=5.0).passed

    def test_a_panel_below_the_floor_fails(self, panel: Panel) -> None:
        clones = {f"c{i}": dict(panel.raters["qwen"]) for i in range(7)}
        report = build_report(Panel(name="c", raters=clones, truth=panel.truth))
        gate = check_gate(report, min_effective=4.0)
        assert not gate.passed
        assert "effective judges 1.00 of 7 < 4.00" in gate.failures[0]

    def test_a_panel_exactly_on_the_floor_passes(self, panel: Panel) -> None:
        """`<` not `<=`. A floor is a minimum, and meeting it is not falling short."""
        report = build_report(panel)
        assert report.independence is not None
        exact = report.independence.effective
        assert exact is not None
        assert check_gate(report, min_effective=exact).passed
        assert not check_gate(report, min_effective=exact + 1e-9).passed

    def test_the_two_gates_are_independent(self, panel: Panel) -> None:
        """This panel has poor agreement and good independence, so they disagree."""
        gate = check_gate(build_report(panel), threshold=0.6, min_effective=2.0)
        assert not gate.passed
        assert all("effective judges" not in f for f in gate.failures)

    def test_no_thresholds_means_nothing_to_fail(self, panel: Panel) -> None:
        assert check_gate(build_report(panel)).passed


class TestMinEffectiveCli:
    def test_a_sufficient_panel_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL), "--min-effective", "2"]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_a_floor_below_one_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL), "--min-effective", "0.5"]) == 2
        assert "below 1.0" in capsys.readouterr().err

    def test_a_bad_groups_file_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        bad = tmp_path / "g.json"
        bad.write_text('{"claude": {"nested": 1}}', encoding="utf-8")
        assert main(["report", str(PANEL), "--groups", str(bad)]) == 2
        assert "must be a string" in capsys.readouterr().err

    def test_a_groups_file_changes_the_reported_grouping(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        g = tmp_path / "g.json"
        g.write_text('{"gemini": "X", "glm": "X", "gpt": "Y", "claude": "Y"}', encoding="utf-8")
        assert main(["report", str(PANEL), "--groups", str(g)]) == 0
        assert "within-group pairs   2" in capsys.readouterr().out


class TestPermutationPDisplay:
    def test_a_p_below_the_resolution_is_not_printed_as_zero(self) -> None:
        """The add-one estimator keeps p positive; rounding must not undo that."""
        from judgecheck.report import render_text

        n = 40
        truth = {f"i{i}": "TP" for i in range(n)}
        half = {f"i{i}": ("FP" if i < n // 2 else "TP") for i in range(n)}
        panel = Panel(name="p", raters={"a": half, "b": dict(half)}, truth=truth)
        text = render_text(build_report(panel))
        assert "permutation p <0.001" in text
        assert "permutation p 0.000" not in text

    def test_an_ordinary_p_still_prints_three_places(self, panel: Panel) -> None:
        from judgecheck.report import render_text

        assert "permutation p 0.002" in render_text(build_report(panel))


class TestGuardsTheSecondSweepFound:
    """Eight guards added during the round-2 fixes that no test distinguished.

    Every one was correct code with nothing proving it had to be. Written after
    the mutation sweep flagged them as survivors, which is the entire argument
    for running the sweep after a change rather than before.
    """

    def test_the_eigenvalue_shift_finds_the_largest_not_the_most_negative(self) -> None:
        """Power iteration converges to the largest eigenvalue by *magnitude*.

        Without the shift, a matrix whose most negative eigenvalue dominates
        returns that one's magnitude, which is a different number entirely.
        Here the eigenvalues are +2 and -4: unshifted power iteration finds 4,
        the answer is 2.
        """
        assert _largest_eigenvalue([[-1.0, -3.0], [-3.0, -1.0]]) == pytest.approx(2.0)

    def test_the_eigenvalue_matches_a_known_correlation_matrix(self) -> None:
        """Equicorrelation rho gives eigenvalues 1+(k-1)rho and 1-rho."""
        rho, k = 0.5, 4
        matrix = [[1.0 if i == j else rho for j in range(k)] for i in range(k)]
        assert _largest_eigenvalue(matrix) == pytest.approx(1 + (k - 1) * rho)

    def test_a_rater_with_no_comparable_pair_does_not_inflate_k(self, panel: Panel) -> None:
        """The round-1 fix missed this and a perfect oracle bought a whole judge."""
        assert panel.truth is not None
        base = panel_independence(panel.raters, panel.truth)
        with_oracle = panel_independence({**panel.raters, "oracle": dict(panel.truth)}, panel.truth)
        assert with_oracle.raters == base.raters
        assert with_oracle.excluded_raters == ("oracle",)
        assert with_oracle.effective == base.effective

    def test_adding_a_zero_information_rater_never_raises_independence(self, panel: Panel) -> None:
        """The property the whole module exists to satisfy."""
        assert panel.truth is not None
        base = panel_independence(panel.raters, panel.truth)
        assert base.efficiency is not None
        for tag, extra in (
            ("stamp", dict.fromkeys(panel.truth, "TP")),
            ("clone-gemini", dict(panel.raters["gemini"])),
            ("clone-gpt", dict(panel.raters["gpt"])),
        ):
            after = panel_independence({**panel.raters, tag: extra}, panel.truth)
            assert after.efficiency is not None
            assert after.efficiency <= base.efficiency, tag

    def test_a_single_raters_labels_are_rejected_with_a_useful_error(self) -> None:
        """`_named`'s docstring has always described this; now it enforces it.

        The bad input goes through `Any` because mypy already rejects it, which
        is the point: the guard is for callers who do not type-check, and
        without it they got `AttributeError: 'str' object has no attribute
        'get'` from inside a statistic.
        """
        one_raters_labels: Any = {"i1": "TP"}
        for call in (
            lambda: panel_independence(one_raters_labels, {"i1": "TP"}),
            lambda: coincident_errors(one_raters_labels, {"i1": "TP"}),
            lambda: group_agreement(one_raters_labels, {"i1": "A"}),
        ):
            with pytest.raises(TypeError, match="expected a panel"):
                call()

    def test_a_nan_correlation_reports_not_measurable(
        self, panel: Panel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive, and the failure it prevents is the one this package fixes.

        `_phi` cannot currently emit NaN, so this guard is unreachable from
        real data and is pinned by forcing it. Returning `k` for a failed
        computation would send it through `--min-effective` as a pass.
        """
        monkeypatch.setattr("judgecheck.independence._phi", lambda ea, eb: float("nan"))
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth)
        assert ind.interpretation == "not measurable"
        assert ind.effective is None

    def test_the_permutation_uses_each_pair_s_own_overlap(self) -> None:
        """Drawn on the complete-case intersection, phi = 1.0 read as chance.

        Two judges wrong on the identical half of the items, plus a third that
        scored only a few, so the intersection every rater shares is tiny. The
        p value must describe the pair it is printed beside.
        """
        truth = {f"i{i}": "TP" for i in range(20)}
        both_wrong = {f"i{i}": ("FP" if i < 10 else "TP") for i in range(20)}
        sparse = {f"i{i}": "TP" for i in range(0, 20, 4)}
        got = coincident_errors({"a": both_wrong, "b": dict(both_wrong), "c": sparse}, truth)
        assert got.worst is not None and got.worst.phi == pytest.approx(1.0)
        assert got.p_value is not None
        assert got.p_value < 0.01, "identical failures are not consistent with chance"

    def test_the_permutation_preserves_each_judges_error_count(self) -> None:
        """The null is "equally accurate, failing independently".

        A draw that changes how often a judge is wrong is testing a different
        hypothesis, and would make any judge look correlated with any other.
        """
        rng = random.Random(5)
        flags = {
            "a": {f"i{i}": int(i < 7) for i in range(20)},
            "b": {f"i{i}": int(i % 3 == 0) for i in range(20)},
        }
        for _ in range(20):
            drawn = _permuted(flags, rng)
            for name, errs in flags.items():
                assert len(drawn[name]) == sum(errs.values()), name
                assert drawn[name] <= set(errs)


class TestDegenerateCohen:
    """`cohens_kappa` had two results that read as findings and were not."""

    def test_two_constant_raters_are_undefined_not_perfect(self) -> None:
        """Both say TP to everything: the textbook case kappa exists to catch.

        1.0 here also clears any `--fail-under`, so a panel of rubber stamps
        passed a gate while triage flagged every one of its raters DROP.
        """
        constant = {f"i{i}": "TP" for i in range(10)}
        got = cohens_kappa(constant, dict(constant))
        assert got.kappa == 0.0
        assert got.interpretation == UNDEFINED_NO_VARIANCE
        assert got.n == 10, "the items were compared; there was just no variance"

    def test_no_shared_items_is_undefined_not_poor(self) -> None:
        """ "poor" sounds like a finding. They were never compared."""
        got = cohens_kappa({"i1": "TP"}, {"i2": "FP"})
        assert got.n == 0
        assert got.interpretation == UNDEFINED

    def test_a_panel_of_rubber_stamps_fails_the_gate(self) -> None:
        items = [f"i{i}" for i in range(23)]
        stampers = {f"r{j}": dict.fromkeys(items, "TP") for j in range(7)}
        report = build_report(Panel(name="s", raters=stampers, truth=dict.fromkeys(items, "TP")))
        assert report.fleiss.interpretation == UNDEFINED_NO_VARIANCE
        assert report.krippendorff.interpretation == UNDEFINED_NO_VARIANCE
        assert not check_gate(report, 0.9).passed


class TestPermutationsParameter:
    """A public escape hatch for large panels, so it gets validated."""

    def test_zero_permutations_is_rejected(self, panel: Panel) -> None:
        """It returned p = 1.0, i.e. "consistent with chance", from no evidence."""
        assert panel.truth is not None
        with pytest.raises(ValueError, match="at least 1"):
            coincident_errors(panel.raters, panel.truth, permutations=0)

    def test_negative_permutations_is_rejected(self, panel: Panel) -> None:
        """It returned a negative p, which the renderer printed as "<0.001"."""
        assert panel.truth is not None
        with pytest.raises(ValueError, match="at least 1"):
            coincident_errors(panel.raters, panel.truth, permutations=-5)

    def test_a_small_count_still_works(self, panel: Panel) -> None:
        assert panel.truth is not None
        got = coincident_errors(panel.raters, panel.truth, permutations=50)
        assert got.p_value is not None
        assert 0.0 < got.p_value <= 1.0


class TestNullCalibration:
    """The percentage is unreadable without the null it should be read against.

    `k / lambda_max` is biased downward when items are few relative to judges:
    the top eigenvalue of a correlation matrix built from a handful of
    observations is inflated by sampling noise alone. So judges that are
    independent by construction do not score 100%, and reporting "31% of
    nominal" against an implied ceiling of 100% overstates the dependence by
    about a factor of two.
    """

    def test_the_null_is_far_below_one_on_this_panel(self, panel: Panel) -> None:
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        assert ind.null_efficiency is not None
        assert 0.45 < ind.null_efficiency.point < 0.65, ind.null_efficiency.point
        assert ind.null_efficiency.high < 0.8, "independent judges never reach 100% here"

    def test_the_observed_panel_is_below_the_null(self, panel: Panel) -> None:
        """This is the inferential claim, and it survives calibration."""
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        assert ind.efficiency is not None and ind.null_efficiency is not None
        assert ind.efficiency < ind.null_efficiency.low
        assert ind.p_value is not None and ind.p_value < 0.05

    def test_the_caution_threshold_would_fire_on_independent_panels(self, panel: Panel) -> None:
        """Which is why the raw percentage cannot carry the claim alone.

        Kohli's 0.5 line was proposed for a panel where the estimator is
        unbiased. Here the null 95% band straddles it, so a quarter of
        independent panels of this shape would trip it.
        """
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        assert ind.null_efficiency is not None
        assert ind.null_efficiency.low < CAUTION_EFFICIENCY < ind.null_efficiency.high

    def test_genuinely_independent_judges_are_not_flagged(self, panel: Panel) -> None:
        """The calibration's whole job: an independent panel must come out clean."""
        assert panel.truth is not None
        rng = random.Random(3)
        items = sorted(panel.truth)
        # Same error counts as the real panel, redistributed independently.
        counts = {
            name: sum(1 for i in items if panel.raters[name].get(i) != panel.truth[i])
            for name in panel.raters
        }
        raters = {}
        for name, count in counts.items():
            wrong = set(rng.sample(items, count))
            raters[name] = {
                i: ("FP" if panel.truth[i] == "TP" else "TP") if i in wrong else panel.truth[i]
                for i in items
            }
        ind = panel_independence(raters, panel.truth, null_draws=NULL_DRAWS)
        assert ind.p_value is not None
        assert ind.p_value > 0.05, "an independent panel must not read as correlated"

    def test_calibration_is_off_by_default(self, panel: Panel) -> None:
        """It costs about a third of a second; a plain report should stay fast."""
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth)
        assert ind.null_efficiency is None
        assert ind.p_value is None

    def test_it_is_deterministic(self, panel: Panel) -> None:
        assert panel.truth is not None
        a = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        b = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        assert a.null_efficiency == b.null_efficiency
        assert a.p_value == b.p_value

    def test_a_gate_turns_it_on(self, panel: Panel) -> None:
        """Same rule the bootstrap follows: calibrate where a decision is made."""
        plain = build_report(panel)
        assert plain.independence is not None
        assert plain.independence.null_efficiency is None
        gated = build_report(panel, intervals=True)
        assert gated.independence is not None
        assert gated.independence.null_efficiency is not None

    def test_the_p_value_is_never_exactly_zero(self, panel: Panel) -> None:
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        assert ind.p_value is not None and ind.p_value > 0.0

    def test_the_band_is_ordered(self, panel: Panel) -> None:
        """low <= median <= high, which requires the draws to be sorted first.

        Every other assertion here checks the band's values and would pass on
        an unordered one that happened to land in range. This is the structural
        invariant: a percentile taken from an unsorted list is not a percentile.
        """
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        band = ind.null_efficiency
        assert band is not None
        assert band.low <= band.point <= band.high
        assert band.width > 0.0, "500 draws of a noisy statistic are not all equal"

    def test_the_band_brackets_a_realistic_share_of_draws(self, panel: Panel) -> None:
        """A 95% band that is not built from sorted draws will not do this."""
        assert panel.truth is not None
        ind = panel_independence(panel.raters, panel.truth, null_draws=NULL_DRAWS)
        band = ind.null_efficiency
        assert band is not None
        # The observed panel sits below the band; an independent redraw sits inside it.
        assert ind.efficiency is not None
        assert ind.efficiency < band.low
        assert band.low < band.point < band.high
