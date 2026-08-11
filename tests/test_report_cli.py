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
from judgecheck.cli import main
from judgecheck.report import build_report, check_gate, render_text, to_dict, to_json

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
        assert counts == {"UNANIMOUS": 1, "MAJORITY": 19, "SPLIT": 3}

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
