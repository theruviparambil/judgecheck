"""The report layer and the CLI.

Two things worth testing here that the statistic tests cannot:

  * the text report and the JSON report must carry the same numbers. They are
    rendered by different code, so they can drift; `build_report` computing
    once is only a safeguard if something checks it held.
  * the CLI must fail usefully. A tool that prints a traceback on a missing
    directory is not finished.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from judgecheck import load_panel
from judgecheck.agreement import mean_pairwise_kappa
from judgecheck.cli import main
from judgecheck.consensus import UNSCORED, consensus, split_items
from judgecheck.io import load_judgments, load_truth
from judgecheck.report import build_report, check_gate, render_text, to_dict, to_json
from judgecheck.triage import BALANCED, LENIENT, UNKNOWN, split_leaning, triage
from judgecheck.types import LABELS, Panel
from judgecheck.validity import accuracy, validity

PANEL_DIR = Path(__file__).parent / "data" / "panel-real"


@pytest.fixture(scope="module")
def report_dict() -> dict[str, Any]:
    return to_dict(build_report(load_panel(PANEL_DIR)))


class TestReportContent:
    def test_carries_the_panel_shape(self, report_dict: dict[str, Any]) -> None:
        assert report_dict["items"] == 23
        assert report_dict["raters"] == [
            "claude",
            "deepseek",
            "gemini",
            "glm",
            "gpt",
            "grok",
            "qwen",
        ]

    def test_reports_the_reproduced_panel_statistics(self, report_dict: dict[str, Any]) -> None:
        agreement = report_dict["agreement"]
        assert agreement["fleissKappa"]["value"] == 0.13541263908579276
        assert agreement["krippendorffAlpha"]["value"] == 0.14078274691755766
        assert agreement["fleissKappa"]["interpretation"] == "poor"
        assert len(agreement["pairwiseKappa"]) == 21

    def test_pairwise_entries_are_ordered_and_unique(self, report_dict: dict[str, Any]) -> None:
        pairs = [(p["a"], p["b"]) for p in report_dict["agreement"]["pairwiseKappa"]]
        assert all(a < b for a, b in pairs), "each pair should be keyed a < b"
        assert len(set(pairs)) == 21
        assert pairs == sorted(pairs)

    def test_to_dict_sorts_pairs_it_did_not_build(self) -> None:
        """Ordering is `to_dict`'s job, not an accident of how the mapping arrived.

        `pairwise_kappa` happens to emit pairs already sorted, so the real panel
        cannot tell the two apart. Feed it a shuffled mapping instead.
        """
        report = build_report(load_panel(PANEL_DIR))
        shuffled = dict(reversed(list(report.pairwise.items())))
        assert list(shuffled) != sorted(shuffled), "fixture should be out of order"

        emitted = to_dict(replace(report, pairwise=shuffled))["agreement"]["pairwiseKappa"]
        pairs = [(p["a"], p["b"]) for p in emitted]
        assert pairs == sorted(pairs)
        assert len(pairs) == 21

    def test_consensus_counts_sum_to_the_item_count(self, report_dict: dict[str, Any]) -> None:
        counts = report_dict["consensus"]["counts"]
        assert sum(counts.values()) == 23
        assert counts == {"UNANIMOUS": 1, "MAJORITY": 19, "SPLIT": 3, "UNSCORED": 0}

    def test_validity_matches_the_reference_harness(self, report_dict: dict[str, Any]) -> None:
        expected = {
            "claude": (12, 15, 12, 13),
            "deepseek": (3, 15, 3, 3),
            "gemini": (15, 15, 15, 18),
            "glm": (15, 15, 15, 21),
            "gpt": (10, 15, 10, 10),
            "grok": (2, 15, 2, 2),
            "qwen": (9, 15, 9, 12),
        }
        for name, (caught, truth_pos, correct, called) in expected.items():
            v = report_dict["validity"][name]
            assert (v["caught"], v["truthPositives"], v["correctCalls"], v["called"]) == (
                caught,
                truth_pos,
                correct,
                called,
            ), name

    def test_every_rater_gets_a_recommendation(self, report_dict: dict[str, Any]) -> None:
        allowed = {"KEEP", "REVIEW", "DROP / DOWN-WEIGHT"}
        for name, t in report_dict["triage"].items():
            assert t["recommendation"] in allowed, name
            assert t["labeled"] == 23

    def test_floats_are_not_rounded(self, report_dict: dict[str, Any]) -> None:
        """Rounding would defeat the point: cross-checking another implementation."""
        value = report_dict["agreement"]["fleissKappa"]["value"]
        assert repr(value) == "0.13541263908579276"

    def test_json_survives_a_round_trip(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        assert json.loads(to_json(report)) == json.loads(json.dumps(to_dict(report)))


class TestTextAndJsonAgree:
    """Rendered by different code paths; they must not drift apart."""

    def test_headline_kappa_appears_in_both(self, report_dict: dict[str, Any]) -> None:
        text = render_text(build_report(load_panel(PANEL_DIR)))
        fleiss = report_dict["agreement"]["fleissKappa"]["value"]
        assert f"{fleiss:+.3f}" in text

    def test_every_recall_fraction_appears_in_both(self, report_dict: dict[str, Any]) -> None:
        text = render_text(build_report(load_panel(PANEL_DIR)))
        for name, v in report_dict["validity"].items():
            assert f"{v['caught']}/{v['truthPositives']}" in text, name

    def test_every_flagged_rater_is_named_in_the_text(self, report_dict: dict[str, Any]) -> None:
        text = render_text(build_report(load_panel(PANEL_DIR)))
        for name, t in report_dict["triage"].items():
            assert name in text
            for flag in t["flags"]:
                assert flag in text, f"{name}: {flag}"

    def test_contested_items_are_listed(self, report_dict: dict[str, Any]) -> None:
        text = render_text(build_report(load_panel(PANEL_DIR)))
        splits = [
            e["findingId"] for e in report_dict["consensus"]["items"] if e["consensus"] == "SPLIT"
        ]
        assert splits, "the real panel has contested items"
        for finding_id in splits:
            assert finding_id in text


class TestCli:
    def test_text_report_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL_DIR)]) == 0
        out = capsys.readouterr().out
        assert "PANEL AGREEMENT" in out
        assert "+0.135" in out

    def test_json_report_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL_DIR), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["agreement"]["fleissKappa"]["value"] == 0.13541263908579276

    def test_a_missing_panel_directory_is_an_error_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(tmp_path / "nope")]) == 2
        assert "error:" in capsys.readouterr().err

    def test_a_directory_with_no_raters_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(tmp_path)]) == 2
        assert "no rater" in capsys.readouterr().err

    def test_a_positive_label_outside_the_label_set_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["report", str(PANEL_DIR), "--labels", "TP,FP", "--positive-label", "NOPE"])
        assert code == 2
        assert "not in the label set" in capsys.readouterr().err

    def test_a_single_label_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL_DIR), "--labels", "TP"]) == 2
        assert "at least two labels" in capsys.readouterr().err

    def test_a_restricted_label_set_changes_the_statistics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Dropping NEEDS_INVESTIGATION from the label set must change the answer."""
        assert main(["report", str(PANEL_DIR), "--labels", "TP,FP", "--json"]) == 0
        restricted = json.loads(capsys.readouterr().out)
        assert restricted["agreement"]["fleissKappa"]["value"] != 0.13541263908579276
        assert restricted["labels"] == ["TP", "FP"]

    def test_a_panel_without_truth_reports_no_validity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for name in ("a", "b"):
            (tmp_path / f"{name}.jsonl").write_text(
                "\n".join(
                    json.dumps({"findingId": f"f{i}", "label": "TP" if i % 2 else "FP"})
                    for i in range(4)
                ),
                encoding="utf-8",
            )
        assert main(["report", str(tmp_path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "validity" not in payload, "no truth.json means no validity section"
        # everything that does not need truth is still reported
        assert payload["agreement"]["fleissKappa"]["n"] == 4
        assert sum(payload["consensus"]["counts"].values()) == 4
        assert set(payload["triage"]) == {"a", "b"}
        assert all(t["agreementWithTruth"] is None for t in payload["triage"].values())

    def test_no_subcommand_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "judgecheck" in capsys.readouterr().out


class TestGate:
    """`--fail-under` is the only part of the tool that renders a verdict.

    It checks both panel coefficients, because passing on the kinder one and
    failing the other is exactly the borderline case a gate exists to stop.
    """

    def test_a_threshold_below_both_coefficients_passes(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        gate = check_gate(report, 0.1)
        assert gate.passed
        assert gate.failures == ()

    def test_a_threshold_above_both_coefficients_fails_on_both(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        gate = check_gate(report, 0.6)
        assert not gate.passed
        assert len(gate.failures) == 2

    def test_a_threshold_between_the_two_still_fails(self) -> None:
        """Fleiss is 0.1354 and Krippendorff 0.1408, so 0.14 splits them."""
        report = build_report(load_panel(PANEL_DIR))
        gate = check_gate(report, 0.14)
        assert not gate.passed
        assert gate.failures == ("Fleiss' kappa 0.135 < 0.140",)

    def test_a_coefficient_exactly_on_the_threshold_passes(self) -> None:
        """At or above, not strictly above: a panel that exactly meets the bar met it."""
        report = build_report(load_panel(PANEL_DIR))
        gate = check_gate(report, report.fleiss.value)
        assert gate.fleiss == report.fleiss.value
        assert "Fleiss" not in " ".join(gate.failures)

    def test_gate_appears_in_json_only_when_requested(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        assert "gate" not in to_dict(report)
        assert "gate" in to_dict(report, check_gate(report, 0.5))

    def test_gate_appears_in_text_only_when_requested(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        assert "GATE" not in render_text(report)
        assert "FAIL" in render_text(report, check_gate(report, 0.9))
        assert "PASS" in render_text(report, check_gate(report, -0.5))


class TestGateCli:
    def test_a_failing_gate_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL_DIR), "--fail-under", "0.6"]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_a_passing_gate_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["report", str(PANEL_DIR), "--fail-under", "0.1"]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_no_gate_flag_never_fails_on_low_agreement(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reporting is not judging. Without --fail-under this panel still exits 0."""
        assert main(["report", str(PANEL_DIR)]) == 0
        assert "GATE" not in capsys.readouterr().out

    def test_a_failing_gate_still_prints_the_whole_report(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(PANEL_DIR), "--fail-under", "0.9", "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["gate"]["passed"] is False
        assert len(payload["agreement"]["pairwiseKappa"]) == 21
        assert "validity" in payload

    def test_a_threshold_outside_the_kappa_range_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(PANEL_DIR), "--fail-under", "5"]) == 2
        assert "outside the range" in capsys.readouterr().err
        assert main(["report", str(PANEL_DIR), "--fail-under", "-2"]) == 2

    def test_a_non_numeric_threshold_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["report", str(PANEL_DIR), "--fail-under", "high"])
        assert exc.value.code == 2

    def test_an_input_error_outranks_the_gate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing panel is exit 2, not exit 1: you cannot gate what you cannot read."""
        assert main(["report", str(tmp_path / "nope"), "--fail-under", "0.9"]) == 2
        assert "error:" in capsys.readouterr().err


class TestInputErrorsExitTwo:
    """The exit-code table promises 2 for "usage or input error".

    Only `FileNotFoundError` was caught, so a corrupt `truth.json`, a rater path
    that is a directory, and a non-UTF-8 file each reached the user as a
    traceback while a sibling test asserted that a *missing* directory did not.
    """

    def _raters(self, d: Path) -> None:
        (d / "a.jsonl").write_text('{"findingId": "f1", "label": "TP"}\n', encoding="utf-8")

    def test_a_corrupt_truth_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._raters(tmp_path)
        (tmp_path / "truth.json").write_text('{"verdicts": [', encoding="utf-8")
        assert main(["report", str(tmp_path)]) == 2
        assert "error:" in capsys.readouterr().err

    def test_a_rater_path_that_is_a_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.jsonl").mkdir()
        assert main(["report", str(tmp_path)]) == 2
        assert "error:" in capsys.readouterr().err

    def test_a_non_utf8_rater_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.jsonl").write_bytes(b"\xff\xfe\x00bad")
        assert main(["report", str(tmp_path)]) == 2
        assert "error:" in capsys.readouterr().err

    def test_a_missing_directory_still_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(tmp_path / "nope")]) == 2
        assert "error:" in capsys.readouterr().err


class TestDegenerateInputIsNotReportedAsAMeasurement:
    """Guards for output that used to be confidently wrong rather than absent.

    Each of these produced a plausible number from no evidence: a contested
    item nobody had voted on, an abstention rate for a label set with no
    abstention label, a lenient/strict reading with no TP/FP axis to read it on.
    """

    def test_an_item_nobody_voted_on_is_unscored_not_split(self) -> None:
        entries = consensus({"a": {"x": "BOGUS"}, "b": {"x": "ALSO_BOGUS"}})
        assert entries[0].consensus == UNSCORED
        assert entries[0].consensus_label is None
        assert split_items(entries) == ()

    def test_unscored_items_do_not_appear_as_needing_adjudication(self) -> None:
        raters = {"a": {"x": "BOGUS"}, "b": {"y": "TP"}}
        text = render_text(build_report(Panel(name="p", raters=raters)))
        assert "contested items" not in text

    def test_abstention_is_none_for_a_label_set_without_an_abstention_label(self) -> None:
        raters = {"x": {f"i{i}": ("UNSURE" if i else "GOOD") for i in range(10)}}
        t = triage(raters, None, ("GOOD", "BAD", "UNSURE"))["x"]
        assert t.abstention is None
        assert not any("ABSTAINS" in f for f in t.flags)

    def test_abstention_still_measured_for_the_standard_label_set(self) -> None:
        raters = {"x": {f"i{i}": ("NEEDS_INVESTIGATION" if i else "TP") for i in range(10)}}
        t = triage(raters, None, LABELS)["x"]
        assert t.abstention == pytest.approx(0.9)
        assert any("ABSTAINS" in f for f in t.flags)

    def test_leaning_is_not_reported_as_balanced_without_a_tp_fp_axis(self) -> None:
        raters = {"x": {"i1": "GOOD", "i2": "BAD"}}
        got = split_leaning(raters, ("i1", "i2"), ("GOOD", "BAD"))["x"][1]
        assert got == UNKNOWN
        assert got != BALANCED

    def test_leaning_still_measured_for_the_standard_label_set(self) -> None:
        raters = {"x": {"i1": "TP", "i2": "TP", "i3": "TP"}}
        assert split_leaning(raters, ("i1", "i2", "i3"), LABELS)["x"][1] == LENIENT

    def test_an_unmeasurable_abstention_renders_as_a_dash_not_a_zero(self) -> None:
        raters = {"x": {"i1": "GOOD", "i2": "BAD"}, "y": {"i1": "GOOD", "i2": "GOOD"}}
        panel = Panel(name="p", raters=raters)
        text = render_text(build_report(panel, labels=("GOOD", "BAD"), positive_label="GOOD"))
        triage_rows = [
            line
            for line in text.split("RATER TRIAGE")[1].split("\n")
            if line.startswith("  x ") or line.startswith("  y ")
        ]
        assert triage_rows, text
        assert all("0%" not in row for row in triage_rows), triage_rows


class TestUndefinedValidityIsNotZero:
    """`0/0 (0%)` read as total failure where the question was simply empty."""

    def test_no_truth_positives_gives_none_not_zero(self) -> None:
        raters = {"a": {"i1": "TP", "i2": "FP"}}
        got = validity(raters, {"i1": "FP", "i2": "FP"}, positive_label="TP")["a"]
        assert got.truth_positives == 0
        assert got.recall is None

    def test_no_calls_made_gives_none_precision(self) -> None:
        raters = {"a": {"i1": "FP", "i2": "FP"}}
        got = validity(raters, {"i1": "TP", "i2": "TP"}, positive_label="TP")["a"]
        assert got.called == 0
        assert got.precision is None
        assert got.f1 is None

    def test_a_real_score_is_still_a_number(self) -> None:
        raters = {"a": {"i1": "TP", "i2": "FP"}}
        got = validity(raters, {"i1": "TP", "i2": "TP"}, positive_label="TP")["a"]
        assert got.recall == pytest.approx(0.5)
        assert got.precision == pytest.approx(1.0)
        assert got.f1 == pytest.approx(2 / 3)

    def test_the_renderer_prints_a_dash(self) -> None:
        text = render_text(build_report(load_panel(PANEL_DIR), positive_label="OUT_OF_SCOPE"))
        block = text.split("VALIDITY")[1].split("CONSENSUS")[0]
        assert "(-)" in block
        assert "(0%)" not in block

    def test_rater_validity_carries_its_own_name(self) -> None:
        """It used to return `rater=""` and let `validity()` hand-copy six fields."""
        got = validity({"claude": {"i1": "TP"}}, {"i1": "TP"})["claude"]
        assert got.rater == "claude"

    def test_build_report_rejects_a_positive_label_outside_the_set(self) -> None:
        """The CLI checked this; a library caller got a table of zeros."""
        with pytest.raises(ValueError, match="not in the label set"):
            build_report(load_panel(PANEL_DIR), positive_label="NOPE")


class TestGuardsTheThirdSweepFound:
    """Defects an outside reviewer found that the suite did not.

    Each of these produced a confident wrong answer rather than a crash, and
    each sits on a path the complete 7x23 fixture cannot reach.
    """

    def test_a_zero_threshold_does_not_pass_an_undefined_panel(self) -> None:
        """`--fail-under 0.0` passed a panel the report calls undefined.

        Both degenerate branches return value 0.0, which is right, and the gate
        read only `.value`. So a report printing "undefined (all ratings in one
        category)" three sections above still exited 0 against a floor of zero.
        """
        items = [f"i{i}" for i in range(23)]
        stampers = {f"r{j}": dict.fromkeys(items, "TP") for j in range(7)}
        report = build_report(Panel(name="s", raters=stampers, truth=dict.fromkeys(items, "TP")))
        gate = check_gate(report, 0.0)
        assert not gate.passed
        assert "undefined" in gate.failures[0]
        # The value matters too, not just the interpretation. 1.0 here would
        # still fail this gate (which reads the interpretation) while sailing
        # through anyone reading `.value` directly, including the JSON consumer.
        assert report.fleiss.value == 0.0
        assert report.krippendorff.value == 0.0

    def test_a_real_panel_still_passes_a_zero_threshold(self) -> None:
        report = build_report(load_panel(PANEL_DIR))
        assert check_gate(report, 0.0).passed

    def test_mean_pairwise_skips_pairs_that_were_never_compared(self) -> None:
        """A disjoint third rater used to drag a perfect pair down to 0.5."""
        shared = {"i1": "TP", "i2": "FP", "i3": "TP"}
        raters = {
            "a": shared,
            "b": dict(shared),
            "c": {"z1": "TP", "z2": "FP", "z3": "TP"},
        }
        got = mean_pairwise_kappa(raters)
        assert got["a"] == pytest.approx(1.0)
        assert got["b"] == pytest.approx(1.0)
        assert got["c"] is None, "never compared is not the same as agrees with nobody"

    def test_accuracy_respects_the_label_set(self) -> None:
        """Truth verdicts outside the label set were scored as failures here.

        Every other statistic drops them, so triage flagged LOW ACCURACY off a
        denominator no other number in the report used.
        """
        labels = {"i1": "TP", "i2": "FP", "i3": "TP"}
        truth = {"i1": "TP", "i2": "FP", "i3": "NEEDS_INVESTIGATION"}
        assert accuracy(labels, truth) == (2, 3)
        assert accuracy(labels, truth, ("TP", "FP")) == (2, 2)

    def test_precision_is_scored_only_over_adjudicated_items(self) -> None:
        """Unadjudicated positives silently counted as false positives.

        Recall and precision were computed over different populations and F1
        combined them anyway. Partial adjudication is the normal case.
        """
        rater = {f"i{i}": "TP" for i in range(1, 6)}
        truth = {"i1": "TP", "i2": "FP"}
        got = validity({"a": rater}, truth)["a"]
        assert got.called == 2, "only the adjudicated calls count"
        assert got.correct_calls == 1
        assert got.precision == pytest.approx(0.5)
        assert got.recall == pytest.approx(1.0)

    def test_a_valid_json_non_object_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """`[1,2]` and `null` parse, then `.get` raised and escaped as exit 1.

        Exit 1 is reserved for "the gate failed", so CI read a loader crash as
        an agreement result.
        """
        p = tmp_path / "a.jsonl"
        p.write_text('[1,2]\nnull\n"str"\n{"findingId":"f1","label":"TP"}\n', encoding="utf-8")
        assert [j.finding_id for j in load_judgments(p)] == ["f1"]

    def test_a_scalar_verdicts_key_does_not_crash(self, tmp_path: Path) -> None:
        p = tmp_path / "truth.json"
        p.write_text('{"verdicts": 3}', encoding="utf-8")
        assert load_truth(p) == {}

    def test_the_cli_exits_two_not_one_on_a_broken_panel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.jsonl").write_bytes(b"\xff\xfe\x00bad")
        assert main(["report", str(tmp_path)]) == 2
        assert "error:" in capsys.readouterr().err


class TestPublicApiShape:
    """Things that get harder to change once anyone imports this."""

    def test_every_constant_a_caller_branches_on_is_exported(self) -> None:
        """Result strings and thresholds, because the submodules are shadowed.

        `from .triage import triage` makes `judgecheck.triage` the *function*,
        so `import judgecheck.triage as t; t.KEEP` raises AttributeError. Four
        names collide that way (triage, validity, consensus, accuracy). Renaming
        would break every caller, so instead nothing needs the module.
        """
        import judgecheck

        needed = [
            "KEEP",
            "REVIEW",
            "DROP",
            "BALANCED",
            "LENIENT",
            "STRICT",
            "UNKNOWN",
            "SKEW_THRESHOLD",
            "ABSTENTION_THRESHOLD",
            "ACCURACY_THRESHOLD",
            "REDUNDANT_KAPPA",
            "LEAN_THRESHOLD",
            "POSITIVE",
            "UNANIMOUS",
            "MAJORITY",
            "SPLIT",
            "UNSCORED",
            "UNDEFINED",
            "UNDEFINED_NO_VARIANCE",
            "CAUTION_EFFICIENCY",
            "MIN_PAIR_ITEMS",
            "PERMUTATIONS",
            "BOOTSTRAP_DRAWS",
        ]
        missing = [n for n in needed if n not in judgecheck.__all__]
        assert not missing, f"not exported: {missing}"
        assert all(hasattr(judgecheck, n) for n in needed)

    def test_the_module_shadowing_is_real_and_documented(self) -> None:
        """Pinned so the docstring's explanation cannot quietly become false."""
        import judgecheck

        assert callable(judgecheck.triage)
        assert callable(judgecheck.validity)
        assert callable(judgecheck.consensus)
        assert "judgecheck.triage` is the" in (judgecheck.__doc__ or "")

    def test_result_types_are_keyword_only(self) -> None:
        """Field order is not public API. `PanelReport` has 17 fields.

        Without this, inserting a statistic is a breaking change for anyone who
        constructed one positionally.

        Each call below supplies the *correct number* of positional arguments,
        which matters: a short call raises "missing N required positional
        arguments" whether or not the class is keyword-only, so an earlier
        version of this test passed either way and let the mutation sweep
        catch it.
        """
        from judgecheck import KappaResult

        # KappaResult has exactly four fields, so this is a complete
        # positional construction and can only fail on kw_only.
        with pytest.raises(TypeError, match="takes 1 positional argument"):
            KappaResult(23, 0.87, 0.52, "moderate")  # type: ignore[call-arg]

        # And it still builds by keyword.
        got = KappaResult(n=23, agreement=0.87, kappa=0.52, interpretation="moderate")
        assert got.kappa == pytest.approx(0.52)

    def test_the_report_type_is_keyword_only_too(self) -> None:
        """17 fields, so positional construction is where a silent break lives."""
        import dataclasses

        from judgecheck import PanelReport

        fields = dataclasses.fields(PanelReport)
        assert len(fields) > 10, "if this shrank, the argument below weakened"
        assert all(f.kw_only for f in fields), "every field must be keyword-only"

    def test_interval_stays_positional_on_purpose(self) -> None:
        """Three fields, one natural order, nowhere to insert a fourth."""
        from judgecheck import Interval

        assert Interval(0.5, 0.2, 0.8).width == pytest.approx(0.6)
