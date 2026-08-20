"""Bootstrap intervals, and the gate warning they exist to power.

Two independent reviewers named the same gap: judgecheck turns point estimates
into build failures on a 23-item panel, and reported no uncertainty anywhere
near the thresholds. `--fail-under 0.6` and `--min-effective 4` decided a build
from numbers whose intervals are wide enough to contain very different answers.

These tests pin the intervals themselves, the determinism the rest of the
report depends on, and the case that makes the feature worth having: a
threshold sitting inside its own interval.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from judgecheck.cli import main
from judgecheck.io import load_panel
from judgecheck.report import build_report, check_gate, render_text, to_dict
from judgecheck.types import Interval, Panel
from judgecheck.uncertainty import (
    BOOTSTRAP_DRAWS,
    CONFIDENCE,
    _resample,
    bootstrap_intervals,
)

PANEL = Path(__file__).parent / "data" / "panel-real"


@pytest.fixture(scope="module")
def panel() -> Panel:
    return load_panel(PANEL)


class TestInterval:
    def test_contains_is_inclusive_at_both_ends(self) -> None:
        band = Interval(point=0.5, low=0.2, high=0.8)
        assert band.contains(0.2) and band.contains(0.8)
        assert band.contains(0.5)
        assert not band.contains(0.19) and not band.contains(0.81)

    def test_width(self) -> None:
        assert Interval(0.5, 0.2, 0.8).width == pytest.approx(0.6)


class TestOnTheRealPanel:
    def test_the_point_estimates_match_the_report(self, panel: Panel) -> None:
        """The interval must be around the number actually published."""
        iv = bootstrap_intervals(panel)
        report = build_report(panel)
        assert iv.fleiss.point == report.fleiss.value
        assert iv.krippendorff.point == report.krippendorff.value
        assert report.independence is not None
        assert iv.effective_raters is not None
        assert iv.effective_raters.point == report.independence.effective

    def test_the_intervals_bracket_their_point_estimates(self, panel: Panel) -> None:
        iv = bootstrap_intervals(panel)
        for band in (iv.fleiss, iv.krippendorff):
            assert band.low <= band.point <= band.high

    def test_agreement_is_poor_but_probably_above_zero(self, panel: Panel) -> None:
        """The one robust conclusion available at 23 items."""
        iv = bootstrap_intervals(panel)
        assert iv.fleiss.high < 0.4, "still poor at the top of the interval"
        assert iv.fleiss.low > -0.05, "and not plausibly negative"

    def test_the_effective_judge_interval_is_wide(self, panel: Panel) -> None:
        """2.19 is not a measurement to two decimals."""
        iv = bootstrap_intervals(panel)
        assert iv.effective_raters is not None
        assert iv.effective_raters.width > 0.5
        assert iv.effective_raters.low < 2.19 < iv.effective_raters.high

    def test_the_interval_is_not_pinned_at_the_rater_count(self, panel: Panel) -> None:
        """An earlier version reported the Kish figure, which is bounded at k.

        62% of its bootstrap draws landed exactly on 7.00, so the "upper limit"
        of the interval was also its median and its point estimate. Reporting
        the conservative estimator instead removes the boundary atom rather
        than papering over it.
        """
        iv = bootstrap_intervals(panel)
        assert iv.effective_raters is not None
        assert iv.effective_raters.high < len(panel.raters) - 1

    def test_n_eff_never_exceeds_the_rater_count_in_any_draw(self, panel: Panel) -> None:
        """Its upper limit piles up at k, so the interval is asymmetric by design."""
        iv = bootstrap_intervals(panel)
        assert iv.effective_raters is not None
        assert iv.effective_raters.high <= len(panel.raters)

    def test_every_draw_was_measurable(self, panel: Panel) -> None:
        iv = bootstrap_intervals(panel)
        assert iv.effective_defined_in == iv.draws == BOOTSTRAP_DRAWS


class TestDeterminism:
    """The full report is asserted bit-identical across hash seeds."""

    def test_two_runs_agree_exactly(self, panel: Panel) -> None:
        first, second = bootstrap_intervals(panel), bootstrap_intervals(panel)
        assert first == second

    def test_the_rendered_report_is_stable(self, panel: Panel) -> None:
        a = render_text(build_report(panel, intervals=True))
        b = render_text(build_report(panel, intervals=True))
        assert a == b


class TestResampling:
    def test_a_repeated_item_is_kept_as_a_distinct_observation(self) -> None:
        """Otherwise the second copy overwrites the first and the draw shrinks.

        Sampling 20 items with replacement draws about 12-13 distinct ones, so
        keying the resample by item id alone silently produces a 13-item panel
        while claiming to bootstrap a 20-item one. Every interval then widens
        toward the small-sample end, which makes the report look more uncertain
        than the data warrants. Asserted against `_resample` directly, because
        from outside the only symptom is intervals that are subtly too wide.
        """
        raters = {
            "a": {f"i{i}": ("TP" if i % 2 else "FP") for i in range(20)},
            "b": {f"i{i}": ("TP" if i % 3 else "FP") for i in range(20)},
        }
        panel = Panel(name="p", raters=raters, truth={f"i{i}": "TP" for i in range(20)})
        items = sorted(raters["a"])
        rng = random.Random(3)
        for _ in range(10):
            drawn, truth = _resample(panel, items, rng)
            assert len(drawn["a"]) == 20, "a draw must have as many observations as the panel"
            assert len(drawn["b"]) == 20
            assert len(truth) == 20
            assert len(set(drawn["a"])) == 20, "keys must be unique per position"

    def test_the_full_bootstrap_reports_the_original_item_count(self) -> None:
        raters = {
            "a": {f"i{i}": ("TP" if i % 2 else "FP") for i in range(20)},
            "b": {f"i{i}": ("TP" if i % 3 else "FP") for i in range(20)},
        }
        iv = bootstrap_intervals(Panel(name="p", raters=raters), draws=50)
        assert iv.items == 20
        assert iv.fleiss.low <= iv.fleiss.point <= iv.fleiss.high

    def test_zero_draws_collapses_to_the_point_estimate(self) -> None:
        raters = {"a": {"i1": "TP"}, "b": {"i1": "TP"}}
        iv = bootstrap_intervals(Panel(name="p", raters=raters), draws=0)
        assert iv.draws == 0
        assert iv.fleiss.low == iv.fleiss.point == iv.fleiss.high

    def test_an_empty_panel_does_not_crash(self) -> None:
        iv = bootstrap_intervals(Panel(name="p", raters={}))
        assert iv.items == 0
        assert iv.effective_raters is None

    def test_a_panel_without_truth_has_no_effective_interval(self) -> None:
        raters = {"a": {"i1": "TP", "i2": "FP"}, "b": {"i1": "TP", "i2": "TP"}}
        iv = bootstrap_intervals(Panel(name="p", raters=raters), draws=20)
        assert iv.effective_raters is None
        assert iv.fleiss.point is not None


class TestReportIntegration:
    def test_intervals_are_off_by_default(self, panel: Panel) -> None:
        """They cost about a second; a plain report should stay fast."""
        assert build_report(panel).intervals is None

    def test_intervals_appear_when_asked(self, panel: Panel) -> None:
        assert build_report(panel, intervals=True).intervals is not None

    def test_json_carries_the_intervals(self, panel: Panel) -> None:
        d = json.loads(json.dumps(to_dict(build_report(panel, intervals=True))))
        assert d["intervals"]["confidence"] == CONFIDENCE
        assert d["intervals"]["draws"] == BOOTSTRAP_DRAWS
        assert d["intervals"]["effectiveRaters"]["low"] < 2.19

    def test_json_omits_intervals_when_not_computed(self, panel: Panel) -> None:
        assert "intervals" not in to_dict(build_report(panel))

    def test_the_text_report_shows_brackets(self, panel: Panel) -> None:
        text = render_text(build_report(panel, intervals=True))
        assert "percentile bootstrap" in text
        assert text.count("[") > 2


class TestTheGateStraddleWarning:
    """The reason this module exists."""

    def test_a_threshold_inside_the_interval_is_called_out(self, panel: Panel) -> None:
        report = build_report(panel, intervals=True)
        text = render_text(report, check_gate(report, min_effective=2.2))
        assert "coin flip" in text
        assert "falls inside the effective-judges interval" in text

    def test_the_gate_still_passes_or_fails_as_asked(self, panel: Panel) -> None:
        """The note is information, not a veto."""
        report = build_report(panel, intervals=True)
        gate = check_gate(report, min_effective=2.0)
        assert gate.passed
        assert "coin flip" in render_text(report, gate)

    def test_a_threshold_outside_the_interval_is_not_flagged(self, panel: Panel) -> None:
        report = build_report(panel, intervals=True)
        assert "coin flip" not in render_text(report, check_gate(report, min_effective=1.0))

    def test_an_agreement_threshold_inside_its_interval_is_flagged(self, panel: Panel) -> None:
        report = build_report(panel, intervals=True)
        assert report.intervals is not None
        inside = report.intervals.fleiss.point
        text = render_text(report, check_gate(report, threshold=inside))
        assert "coin flip" in text

    def test_no_note_without_intervals(self, panel: Panel) -> None:
        report = build_report(panel)
        assert "coin flip" not in render_text(report, check_gate(report, min_effective=2.2))


class TestCli:
    def test_a_gate_implies_intervals(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A build decided from a point estimate must show the estimate's range."""
        assert main(["report", str(PANEL), "--min-effective", "2.2"]) == 1
        out = capsys.readouterr().out
        assert "percentile bootstrap" in out
        assert "coin flip" in out

    def test_the_flag_works_without_a_gate(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL), "--intervals"]) == 0
        assert "percentile bootstrap" in capsys.readouterr().out

    def test_a_plain_report_has_no_intervals(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL)]) == 0
        assert "percentile bootstrap" not in capsys.readouterr().out

    def test_json_gate_output_carries_intervals(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL), "--json", "--fail-under", "0.6"]) == 1
        assert "intervals" in json.loads(capsys.readouterr().out)
