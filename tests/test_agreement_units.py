"""Unit tests for paths the real panel cannot exercise.

The reproduction suite proves we match the reference implementation on real
data, but that panel has complete coverage: all seven raters labeled all
twenty-three findings. So it never exercises abstention, partial overlap, or
the degenerate cases. Mutation testing confirmed that gap, and these tests
close it.
"""

from __future__ import annotations

import pytest

from judgecheck import cohens_kappa, fleiss_kappa, interpret_kappa, krippendorff_alpha

TWO = ("TP", "FP")


class TestCohensKappa:
    def test_perfect_agreement_is_one(self) -> None:
        a = {"1": "TP", "2": "FP", "3": "TP", "4": "FP"}
        r = cohens_kappa(a, a, TWO)
        assert r.n == 4
        assert r.agreement == 1.0
        assert r.kappa == 1.0

    def test_matches_hand_computed_half(self) -> None:
        # observed agreement 3/4 = 0.75; expected by chance 0.5; kappa = 0.5
        a = {"1": "TP", "2": "TP", "3": "FP", "4": "FP"}
        b = {"1": "TP", "2": "FP", "3": "FP", "4": "FP"}
        r = cohens_kappa(a, b, TWO)
        assert r.agreement == pytest.approx(0.75)
        assert r.kappa == pytest.approx(0.5)

    def test_counts_only_items_both_raters_labeled(self) -> None:
        a = {"1": "TP", "2": "FP", "3": "TP"}
        b = {"1": "TP", "2": "FP"}
        assert cohens_kappa(a, b, TWO).n == 2

    def test_ignores_labels_outside_the_set(self) -> None:
        a = {"1": "TP", "2": "SOMETHING_ELSE"}
        b = {"1": "TP", "2": "TP"}
        assert cohens_kappa(a, b, TWO).n == 1

    def test_no_overlap_returns_zero(self) -> None:
        r = cohens_kappa({"1": "TP"}, {"2": "TP"}, TWO)
        assert r.n == 0
        assert r.kappa == 0.0

    def test_rubber_stamping_scores_near_zero_where_accuracy_would_look_high(self) -> None:
        """The whole reason to use kappa instead of accuracy."""
        # 9 of 10 items are truly TP. A rater that always says TP agrees with a
        # discerning rater 90% of the time, which "accuracy" would call excellent.
        truthful = {str(i): ("TP" if i < 9 else "FP") for i in range(10)}
        rubber_stamp = {str(i): "TP" for i in range(10)}
        r = cohens_kappa(truthful, rubber_stamp, TWO)
        assert r.agreement == pytest.approx(0.9)
        assert r.kappa == pytest.approx(0.0), "chance-corrected, a rubber stamp adds no signal"


class TestFleissAbstention:
    """Paths the real panel never hits, because its coverage is complete."""

    def test_skips_items_with_fewer_than_two_ratings(self) -> None:
        # "solo" is labeled by exactly one rater, so agreement is undefined on it.
        a = {"x": "TP", "y": "FP", "solo": "TP"}
        b = {"x": "TP", "y": "FP"}
        r = fleiss_kappa([a, b], TWO)
        assert r.n == 2, "the single-rating item must be excluded, not counted"

    def test_single_rating_item_does_not_divide_by_zero(self) -> None:
        # n_i = 1 would make n_i * (n_i - 1) == 0. Guarding on n_i < 2 is what
        # prevents it; this test fails loudly if that guard is ever loosened.
        a = {"only": "TP"}
        b: dict[str, str] = {}
        r = fleiss_kappa([a, b], TWO)
        assert r.n == 0
        assert r.value == 0.0

    def test_uneven_coverage_is_scored_per_item(self) -> None:
        a = {"1": "TP", "2": "TP", "3": "TP"}
        b = {"1": "TP", "2": "TP"}
        c = {"1": "TP"}
        r = fleiss_kappa([a, b, c], TWO)
        assert r.n == 2, "item 3 has one rating and is skipped"
        assert r.raters == 3


class TestKrippendorffAlpha:
    def test_skips_items_with_fewer_than_two_ratings(self) -> None:
        a = {"x": "TP", "y": "FP", "solo": "TP"}
        b = {"x": "TP", "y": "FP"}
        assert krippendorff_alpha([a, b], TWO).n == 2

    def test_perfect_agreement_is_one(self) -> None:
        a = {"1": "TP", "2": "FP", "3": "TP"}
        assert krippendorff_alpha([a, dict(a)], TWO).value == pytest.approx(1.0)

    def test_empty_panel_is_degenerate_not_a_crash(self) -> None:
        r = krippendorff_alpha([{}, {}], TWO)
        assert r.n == 0
        assert r.value == 1.0


class TestInterpretation:
    @pytest.mark.parametrize(
        ("value", "band"),
        [
            (-0.5, "poor"),
            (0.0, "poor"),
            (0.199, "poor"),
            (0.2, "fair"),
            (0.399, "fair"),
            (0.4, "moderate"),
            (0.599, "moderate"),
            (0.6, "substantial"),
            (0.799, "substantial"),
            (0.8, "near perfect"),
            (1.0, "near perfect"),
        ],
    )
    def test_bands(self, value: float, band: str) -> None:
        assert interpret_kappa(value) == band
