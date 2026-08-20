"""Loading panels off disk, including the malformed cases.

The real panel is complete and well-formed: seven raters, twenty-three
findings, every cell filled. That means the reproduction tests never exercise a
blank line, a truncated JSON object, a missing field, or an absent truth file.
Mutation testing confirmed those branches were unprotected.

Panels in practice are stitched together from separate model runs, so a
half-written line is the normal failure, not an exotic one. The loader skips
the bad line and keeps the rest of the rater; these tests pin that.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from judgecheck.agreement import fleiss_kappa, krippendorff_alpha
from judgecheck.consensus import consensus
from judgecheck.io import load_judgments, load_labels, load_panel, load_truth


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _row(finding_id: str, label: str, **extra: object) -> str:
    return json.dumps({"findingId": finding_id, "label": label, **extra})


class TestLoadJudgments:
    def test_reads_every_well_formed_row(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP"), _row("f2", "FP")])
        got = load_judgments(p)
        assert [(j.finding_id, j.label) for j in got] == [("f1", "TP"), ("f2", "FP")]

    def test_carries_optional_fields_through(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "r.jsonl",
            [_row("f1", "TP", confidence=4, reasoning="because", model="m", vendor="v")],
        )
        j = load_judgments(p)[0]
        assert (j.confidence, j.reasoning, j.model, j.vendor) == (4, "because", "m", "v")

    def test_skips_blank_and_whitespace_lines(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP"), "", "   ", _row("f2", "FP")])
        assert len(load_judgments(p)) == 2

    def test_skips_a_truncated_json_line_and_keeps_the_rest(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "r.jsonl",
            [_row("f1", "TP"), '{"findingId": "f2", "lab', _row("f3", "FP")],
        )
        assert [j.finding_id for j in load_judgments(p)] == ["f1", "f3"]

    def test_skips_a_row_with_no_finding_id(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [json.dumps({"label": "TP"}), _row("f2", "FP")])
        assert [j.finding_id for j in load_judgments(p)] == ["f2"]

    def test_skips_a_row_with_no_label(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [json.dumps({"findingId": "f1"}), _row("f2", "FP")])
        assert [j.finding_id for j in load_judgments(p)] == ["f2"]

    def test_skips_a_row_with_an_empty_label(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", ""), _row("f2", "FP")])
        assert [j.finding_id for j in load_judgments(p)] == ["f2"]

    def test_a_file_of_only_junk_yields_nothing_rather_than_raising(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", ["not json", "{", ""])
        assert load_judgments(p) == ()

    def test_load_labels_flattens_to_id_label(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP"), _row("f2", "FP")])
        assert load_labels(p) == {"f1": "TP", "f2": "FP"}

    def test_a_repeated_finding_keeps_the_last_label(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP"), _row("f1", "FP")])
        assert load_labels(p) == {"f1": "FP"}


class TestLoadTruth:
    def test_reads_verdicts(self, tmp_path: Path) -> None:
        p = tmp_path / "truth.json"
        p.write_text(
            json.dumps({"verdicts": [{"findingId": "f1", "label": "TP"}]}), encoding="utf-8"
        )
        assert load_truth(p) == {"f1": "TP"}

    def test_tolerates_missing_verdicts_key(self, tmp_path: Path) -> None:
        p = tmp_path / "truth.json"
        p.write_text(json.dumps({"notes": "none"}), encoding="utf-8")
        assert load_truth(p) == {}

    def test_tolerates_a_top_level_list(self, tmp_path: Path) -> None:
        p = tmp_path / "truth.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert load_truth(p) == {}

    def test_skips_incomplete_verdicts(self, tmp_path: Path) -> None:
        p = tmp_path / "truth.json"
        p.write_text(
            json.dumps(
                {
                    "verdicts": [
                        {"findingId": "f1", "label": "TP"},
                        {"findingId": "f2"},
                        {"label": "FP"},
                        "not-an-object",
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert load_truth(p) == {"f1": "TP"}


class TestLoadPanel:
    def test_loads_raters_and_truth(self, tmp_path: Path) -> None:
        _write(tmp_path / "alpha.jsonl", [_row("f1", "TP"), _row("f2", "FP")])
        _write(tmp_path / "beta.jsonl", [_row("f1", "FP"), _row("f2", "FP")])
        (tmp_path / "truth.json").write_text(
            json.dumps({"verdicts": [{"findingId": "f1", "label": "TP"}]}), encoding="utf-8"
        )

        panel = load_panel(tmp_path)
        assert panel.rater_names == ("alpha", "beta")
        assert panel.item_ids == ("f1", "f2")
        assert panel.truth == {"f1": "TP"}
        assert panel.judgments is not None
        assert len(panel.judgments["alpha"]) == 2

    def test_truth_is_optional(self, tmp_path: Path) -> None:
        _write(tmp_path / "alpha.jsonl", [_row("f1", "TP")])
        panel = load_panel(tmp_path)
        assert panel.truth is None

    def test_truth_json_is_not_mistaken_for_a_rater(self, tmp_path: Path) -> None:
        _write(tmp_path / "alpha.jsonl", [_row("f1", "TP")])
        (tmp_path / "truth.json").write_text(json.dumps({"verdicts": []}), encoding="utf-8")
        # An empty verdict list is a truth file that yielded nothing, which now
        # warns; this test is about rater discovery, not about that.
        with pytest.warns(UserWarning, match="no usable verdicts"):
            assert load_panel(tmp_path).rater_names == ("alpha",)

    def test_a_directory_with_no_raters_raises(self, tmp_path: Path) -> None:
        (tmp_path / "truth.json").write_text(json.dumps({"verdicts": []}), encoding="utf-8")
        # Raises before the truth file is read, so no warning here.
        with pytest.raises(FileNotFoundError, match="no rater"):
            load_panel(tmp_path)

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_panel(tmp_path / "nope")

    def test_raters_with_different_item_sets_union_cleanly(self, tmp_path: Path) -> None:
        _write(tmp_path / "alpha.jsonl", [_row("f1", "TP")])
        _write(tmp_path / "beta.jsonl", [_row("f2", "FP")])
        panel = load_panel(tmp_path)
        assert panel.item_ids == ("f1", "f2")
        assert panel.raters["alpha"] == {"f1": "TP"}


class TestBoundaryTypeNormalization:
    """`json.loads` returns Any, so annotations stop being enforced at this line.

    A row like `{"findingId": 1, "label": "TP"}` used to load without complaint
    and put an int into a mapping typed `str`. Nothing failed until a later
    `sorted()` over mixed types raised TypeError from four separate call sites:
    fleiss_kappa, krippendorff_alpha, consensus, and Panel.item_ids.

    Mutation testing cannot find this. It perturbs lines that exist, and the
    missing thing was validation nobody had written. That is the argument for
    testing the boundary directly.
    """

    def test_a_numeric_finding_id_is_coerced_to_str(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [json.dumps({"findingId": 1, "label": "TP"})])
        got = load_judgments(p)
        assert got[0].finding_id == "1"
        assert isinstance(got[0].finding_id, str)

    def test_mixed_id_types_no_longer_crash_the_statistics(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            _write(
                tmp_path / f"{name}.jsonl",
                [_row("f-01", "TP"), json.dumps({"findingId": 2, "label": "FP"})],
            )
        panel = load_panel(tmp_path)
        assert panel.item_ids == ("2", "f-01")
        assert all(isinstance(i, str) for i in panel.item_ids)
        fleiss_kappa(panel.raters)
        krippendorff_alpha(panel.raters)
        consensus(panel.raters)

    def test_a_float_id_is_coerced(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [json.dumps({"findingId": 1.5, "label": "TP"})])
        assert load_judgments(p)[0].finding_id == "1.5"

    def test_a_boolean_id_is_rejected_not_coerced(self, tmp_path: Path) -> None:
        """bool subclasses int; "True" is not an id anyone meant to write."""
        p = _write(
            tmp_path / "r.jsonl",
            [json.dumps({"findingId": True, "label": "TP"}), _row("f-2", "FP")],
        )
        assert [j.finding_id for j in load_judgments(p)] == ["f-2"]

    def test_structured_ids_are_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "r.jsonl",
            [
                json.dumps({"findingId": {"a": 1}, "label": "TP"}),
                json.dumps({"findingId": ["x"], "label": "TP"}),
                _row("f-3", "FP"),
            ],
        )
        assert [j.finding_id for j in load_judgments(p)] == ["f-3"]

    def test_a_non_string_label_is_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "r.jsonl",
            [json.dumps({"findingId": "f-1", "label": 1}), _row("f-2", "FP")],
        )
        assert [j.finding_id for j in load_judgments(p)] == ["f-2"]

    def test_whitespace_only_values_are_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "r.jsonl",
            [_row("   ", "TP"), _row("f-1", "  "), _row("f-2", "FP")],
        )
        assert [j.finding_id for j in load_judgments(p)] == ["f-2"]

    def test_ids_and_labels_are_stripped(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("  f-1  ", "  TP  ")])
        j = load_judgments(p)[0]
        assert (j.finding_id, j.label) == ("f-1", "TP")

    def test_truth_json_normalizes_the_same_way(self, tmp_path: Path) -> None:
        t = tmp_path / "truth.json"
        t.write_text(
            json.dumps(
                {
                    "verdicts": [
                        {"findingId": 1, "label": "TP"},
                        {"findingId": True, "label": "FP"},
                        {"findingId": "f-2", "label": 3},
                        {"findingId": "f-3", "label": "FP"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert load_truth(t) == {"1": "TP", "f-3": "FP"}


class TestOptionalFieldNormalization:
    """The annotation fields are validated too, not just id and label.

    `confidence`, `reasoning`, `model` and `vendor` came straight off
    `json.loads` with no check, so `Judgment` promised `int | None` and
    `str | None` and delivered whatever the file held. That was survivable
    while nothing read them. `vendor` is now the default grouping key for
    panel independence, so a dict in that field would have travelled from a
    malformed line into a grouping and failed somewhere unrelated.

    Nothing is coerced here, unlike ids. These fields are descriptive: a
    number in `vendor` means the row is not what we think it is, and turning
    it into `"1"` would hide that instead of surfacing it.
    """

    def test_a_structured_vendor_is_dropped_not_carried(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", vendor={"name": "Anthropic"})])
        assert load_judgments(p)[0].vendor is None

    def test_a_numeric_vendor_is_dropped_rather_than_stringified(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", vendor=7)])
        assert load_judgments(p)[0].vendor is None

    def test_vendor_and_model_are_stripped(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", vendor="  Anthropic  ", model=" c ")])
        j = load_judgments(p)[0]
        assert (j.vendor, j.model) == ("Anthropic", "c")

    def test_an_empty_vendor_becomes_none_not_an_empty_group(self, tmp_path: Path) -> None:
        """An empty string would otherwise become a group named "" in the report."""
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", vendor="   ")])
        assert load_judgments(p)[0].vendor is None

    def test_a_non_string_reasoning_is_dropped(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", reasoning=["a", "b"])])
        assert load_judgments(p)[0].reasoning is None

    def test_an_integer_confidence_survives(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", confidence=4)])
        assert load_judgments(p)[0].confidence == 4

    def test_a_boolean_confidence_is_rejected_not_coerced_to_one(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", confidence=True)])
        assert load_judgments(p)[0].confidence is None

    def test_an_integral_float_confidence_is_accepted(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", confidence=4.0)])
        assert load_judgments(p)[0].confidence == 4

    def test_a_fractional_confidence_is_dropped_rather_than_rounded(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", confidence=4.5)])
        assert load_judgments(p)[0].confidence is None

    def test_a_bad_optional_field_does_not_lose_the_row(self, tmp_path: Path) -> None:
        """id and label are what the statistics need; a junk annotation is not fatal."""
        p = _write(tmp_path / "r.jsonl", [_row("f1", "TP", vendor={"x": 1}, confidence="high")])
        j = load_judgments(p)[0]
        assert (j.finding_id, j.label) == ("f1", "TP")


class TestTruthThatYieldsNothing:
    """A `truth.json` that exists but parses to nothing used to vanish silently.

    `load_truth` is deliberately tolerant, so a top-level list or a key spelled
    `verdict` returns `{}`. But `{}` is falsy, so downstream it was
    indistinguishable from "there is no truth file", and the validity,
    independence and coincident-error sections disappeared from the report on a
    zero exit with nothing said. Tolerance is right; silence was not.
    """

    def _panel(self, tmp_path: Path, truth: str) -> Path:
        _write(tmp_path / "a.jsonl", [_row("f1", "TP")])
        (tmp_path / "truth.json").write_text(truth, encoding="utf-8")
        return tmp_path

    def test_a_misspelled_verdicts_key_warns(self, tmp_path: Path) -> None:
        p = self._panel(tmp_path, json.dumps({"verdict": [{"findingId": "f1", "label": "TP"}]}))
        with pytest.warns(UserWarning, match="no usable verdicts"):
            load_panel(p)

    def test_a_top_level_list_warns(self, tmp_path: Path) -> None:
        p = self._panel(tmp_path, json.dumps([{"findingId": "f1", "label": "TP"}]))
        with pytest.warns(UserWarning, match="no usable verdicts"):
            load_panel(p)

    def test_a_well_formed_truth_file_does_not_warn(self, tmp_path: Path) -> None:
        p = self._panel(tmp_path, json.dumps({"verdicts": [{"findingId": "f1", "label": "TP"}]}))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert load_panel(p).truth == {"f1": "TP"}

    def test_no_truth_file_at_all_is_silent(self, tmp_path: Path) -> None:
        """Absence is not a data problem, so it must not warn."""
        _write(tmp_path / "a.jsonl", [_row("f1", "TP")])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert load_panel(tmp_path).truth is None
