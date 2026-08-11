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
from pathlib import Path

import pytest

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
        assert load_panel(tmp_path).rater_names == ("alpha",)

    def test_a_directory_with_no_raters_raises(self, tmp_path: Path) -> None:
        (tmp_path / "truth.json").write_text(json.dumps({"verdicts": []}), encoding="utf-8")
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
