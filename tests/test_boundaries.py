"""Threshold boundaries the real panel cannot reach.

Every rule in this package has a comparison operator, and the real panel never
lands exactly on one: no rater sits at precisely 80% skew or 40% abstention,
and with seven voters `> voters/2` and `>= voters/2` are the same predicate
(both mean at least four). Mutation testing confirmed that flipping any of
those operators went undetected.

These tests pin the boundary itself, constructed so that flipping the operator
changes the answer.
"""

from __future__ import annotations

import math

from judgecheck.agreement import cohens_kappa
from judgecheck.consensus import consensus
from judgecheck.triage import (
    ABSTENTION_THRESHOLD,
    LEAN_THRESHOLD,
    REDUNDANT_KAPPA,
    SKEW_THRESHOLD,
    split_leaning,
    triage,
)
from judgecheck.validity import accuracy


def _labels(spec: str) -> dict[str, str]:
    """Build a label map from a compact spec like 'TTTF' (T=TP, F=FP, N=NI)."""
    mapping = {"T": "TP", "F": "FP", "N": "NEEDS_INVESTIGATION", "O": "OUT_OF_SCOPE"}
    return {f"i{n}": mapping[c] for n, c in enumerate(spec)}


class TestConsensusMajorityBoundary:
    """An exact half-split is SPLIT, not MAJORITY. Needs an even panel to show."""

    def test_even_panel_tied_two_two_is_split(self) -> None:
        raters = {
            "a": {"x": "TP"},
            "b": {"x": "TP"},
            "c": {"x": "FP"},
            "d": {"x": "FP"},
        }
        entry = consensus(raters)[0]
        assert entry.consensus == "SPLIT", "2 of 4 is not a majority"
        assert entry.consensus_label is None

    def test_even_panel_three_of_four_is_majority(self) -> None:
        raters = {
            "a": {"x": "TP"},
            "b": {"x": "TP"},
            "c": {"x": "TP"},
            "d": {"x": "FP"},
        }
        entry = consensus(raters)[0]
        assert entry.consensus == "MAJORITY"
        assert entry.consensus_label == "TP"

    def test_all_agreeing_is_unanimous_not_majority(self) -> None:
        raters = {"a": {"x": "TP"}, "b": {"x": "TP"}}
        assert consensus(raters)[0].consensus == "UNANIMOUS"

    def test_a_lone_voter_is_trivially_unanimous(self) -> None:
        raters = {"a": {"x": "TP"}, "b": {}}
        assert consensus(raters)[0].consensus == "UNANIMOUS"


class TestSkewBoundary:
    """SKEWED fires strictly above the threshold, not at it."""

    def test_exactly_at_threshold_is_not_flagged(self) -> None:
        assert SKEW_THRESHOLD == 0.8
        # 8 TP of 10 labels == exactly 0.80
        t = triage({"r": _labels("TTTTTTTTFF")})["r"]
        assert t.distribution["TP"] == 8
        assert t.abstention == 0.0
        assert not any(f.startswith("SKEWED") for f in t.flags), "0.80 is not > 0.80"

    def test_just_above_threshold_is_flagged(self) -> None:
        # 9 TP of 10 == 0.90
        t = triage({"r": _labels("TTTTTTTTTF")})["r"]
        assert any(f.startswith("SKEWED") for f in t.flags)


class TestAbstentionBoundary:
    """ABSTAINS fires strictly above the threshold, not at it."""

    def test_exactly_at_threshold_is_not_flagged(self) -> None:
        assert ABSTENTION_THRESHOLD == 0.4
        # 4 NI of 10 == exactly 0.40
        t = triage({"r": _labels("NNNNTTTTFF")})["r"]
        assert t.abstention == 0.4
        assert not any(f.startswith("ABSTAINS") for f in t.flags), "0.40 is not > 0.40"

    def test_just_above_threshold_is_flagged(self) -> None:
        # 5 NI of 10 == 0.50
        t = triage({"r": _labels("NNNNNTTTFF")})["r"]
        assert any(f.startswith("ABSTAINS") for f in t.flags)


class TestLeaningBoundary:
    """LENIENT and STRICT fire at the threshold, inclusive."""

    def test_exactly_seventy_percent_tp_is_lenient(self) -> None:
        assert LEAN_THRESHOLD == 0.7
        rater = {"r": _labels("TTTTTTTFFF")}  # 7 TP of 10
        splits = tuple(f"i{n}" for n in range(10))
        counts, leaning = split_leaning(rater, splits)["r"]
        assert counts["TP"] == 7
        assert leaning == "LENIENT (over-calls TP)", "0.70 is >= 0.70"

    def test_exactly_seventy_percent_strict_is_strict(self) -> None:
        rater = {"r": _labels("FFFFNNNTTT")}  # 7 strict of 10
        splits = tuple(f"i{n}" for n in range(10))
        _, leaning = split_leaning(rater, splits)["r"]
        assert leaning == "STRICT (FP/NI)"

    def test_below_both_thresholds_is_balanced(self) -> None:
        rater = {"r": _labels("TTTTTFFFFF")}  # 50/50
        splits = tuple(f"i{n}" for n in range(10))
        _, leaning = split_leaning(rater, splits)["r"]
        assert leaning == "balanced"

    def test_no_split_items_is_balanced_not_a_crash(self) -> None:
        _, leaning = split_leaning({"r": _labels("TTTT")}, ())["r"]
        assert leaning == "balanced"


class TestRedundancyBoundary:
    """REDUNDANT fires at the threshold, inclusive."""

    def test_identical_raters_are_flagged_redundant(self) -> None:
        assert REDUNDANT_KAPPA == 0.85
        shared = _labels("TTTFFFNNN")
        got = triage({"a": shared, "b": dict(shared)})
        for name in ("a", "b"):
            flags = got[name].flags
            assert any(f.startswith("REDUNDANT") for f in flags), f"{name}: {flags}"

    def test_independent_raters_are_not_flagged_redundant(self) -> None:
        got = triage({"a": _labels("TTTFFF"), "b": _labels("FFFTTT")})
        for name in ("a", "b"):
            assert not any(f.startswith("REDUNDANT") for f in got[name].flags)


class TestRecommendationLadder:
    def test_zero_flags_is_keep(self) -> None:
        # balanced distribution, no truth basis so no accuracy flag
        t = triage({"r": _labels("TTTTFFFNNN")})["r"]
        assert t.flags == ()
        assert t.recommendation == "KEEP"

    def test_one_flag_is_review(self) -> None:
        t = triage({"r": _labels("TTTTTTTTTF")})["r"]  # skewed only
        assert len(t.flags) == 1
        assert t.recommendation == "REVIEW"

    def test_two_flags_is_drop(self) -> None:
        # 9 NI of 10: skewed AND abstaining
        t = triage({"r": _labels("NNNNNNNNNT")})["r"]
        assert len(t.flags) == 2
        assert t.recommendation == "DROP / DOWN-WEIGHT"


class TestRedundancyThresholdIsInclusive:
    """Exactly at the redundancy threshold counts as redundant."""

    def test_a_pair_sitting_exactly_on_the_threshold_is_flagged(self) -> None:
        a = _labels("TTTTFFFFNN")
        b = _labels("TTTFFFFFNN")  # one disagreement, so kappa < 1
        k = cohens_kappa(a, b).kappa
        assert 0.0 < k < 1.0, k
        got = triage({"a": a, "b": b}, redundant_kappa=k)
        for name in ("a", "b"):
            flags = got[name].flags
            assert any(f.startswith("REDUNDANT") for f in flags), f"{name} at exactly {k}: {flags}"

    def test_a_pair_just_below_the_threshold_is_not_flagged(self) -> None:
        a = _labels("TTTTFFFFNN")
        b = _labels("TTTFFFFFNN")
        k = cohens_kappa(a, b).kappa
        got = triage({"a": a, "b": b}, redundant_kappa=math.nextafter(k, 1.0))
        for name in ("a", "b"):
            assert not any(f.startswith("REDUNDANT") for f in got[name].flags)


class TestAccuracyIgnoresUnratedItems:
    """A rater is scored on what it labeled, not punished for what it skipped.

    The real panel is complete, so no reproduction test reaches this branch.
    """

    def test_a_skipped_truth_item_is_not_evaluated(self) -> None:
        truth = {"f1": "TP", "f2": "FP", "f3": "TP"}
        labels = {"f1": "TP", "f2": "FP"}  # f3 skipped
        assert accuracy(labels, truth) == (2, 2)

    def test_a_wrong_label_is_evaluated_and_counted_wrong(self) -> None:
        truth = {"f1": "TP", "f2": "FP"}
        assert accuracy({"f1": "TP", "f2": "TP"}, truth) == (1, 2)

    def test_labels_outside_the_truth_basis_are_ignored(self) -> None:
        truth = {"f1": "TP"}
        assert accuracy({"f1": "TP", "f9": "FP"}, truth) == (1, 1)

    def test_no_overlap_evaluates_nothing(self) -> None:
        assert accuracy({"f9": "TP"}, {"f1": "TP"}) == (0, 0)

    def test_triage_does_not_penalise_a_rater_for_skipping(self) -> None:
        truth = {"f1": "TP", "f2": "FP", "f3": "TP", "f4": "FP", "f5": "TP"}
        # Labeled two items, both right. Perfect on what it answered.
        t = triage({"r": {"f1": "TP", "f2": "FP"}}, truth=truth)["r"]
        assert t.agreement_with_truth == 1.0
        assert not any(f.startswith("LOW ACCURACY") for f in t.flags)

    def test_triage_reports_no_accuracy_when_nothing_overlaps(self) -> None:
        t = triage({"r": {"f9": "TP"}}, truth={"f1": "TP"})["r"]
        assert t.agreement_with_truth is None
        assert not any(f.startswith("LOW ACCURACY") for f in t.flags)
