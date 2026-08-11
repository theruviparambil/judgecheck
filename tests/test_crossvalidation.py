"""Check the statistics against independent implementations by other authors.

The reproduction suite proves this package matches veriva-eval, which shares an
author with it. If both got a formula wrong in the same way, both would agree
and both would be wrong. That is the one failure mode reproduction cannot see.

So the coefficients are also checked against libraries written by other people:
`statsmodels` for Fleiss' and Cohen's kappa, and the `krippendorff` package for
alpha. Agreement there is real external validation of the math, as distinct
from the port fidelity the reproduction suite establishes.

These need extras that the base install deliberately does not pull in:

    pip install -e ".[crossval]"

Without them the module skips. CI installs them, so the README's claim is
enforced there even when a local run skips.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from judgecheck.agreement import cohens_kappa, fleiss_kappa, krippendorff_alpha
from judgecheck.io import load_panel
from judgecheck.types import LABELS, Panel

np = pytest.importorskip("numpy", reason="pip install -e '.[crossval]'")
inter_rater = pytest.importorskip(
    "statsmodels.stats.inter_rater", reason="pip install -e '.[crossval]'"
)
kd = pytest.importorskip("krippendorff", reason="pip install -e '.[crossval]'")

PANEL_DIR = Path(__file__).parent / "data" / "panel-real"

#: Bit-identical is the bar for Fleiss and Cohen. Krippendorff differs in the
#: last ulp because the two implementations accumulate the coincidence matrix in
#: a different order, which is float non-associativity, not disagreement.
EXACT = 0.0
ULP = 1e-14


@pytest.fixture(scope="module")
def panel() -> Panel:
    return load_panel(PANEL_DIR)


@pytest.fixture(scope="module")
def category_index() -> dict[str, int]:
    return {lbl: i for i, lbl in enumerate(LABELS)}


def _counts_table(panel: Panel, category_index: dict[str, int]) -> Any:
    """items x categories, the shape statsmodels' fleiss_kappa expects."""
    items = sorted(panel.item_ids)
    table = np.zeros((len(items), len(LABELS)), dtype=int)
    for row, item in enumerate(items):
        for rater in sorted(panel.raters):
            lbl = panel.raters[rater].get(item)
            if lbl in category_index:
                table[row, category_index[lbl]] += 1
    return table


def _reliability_matrix(panel: Panel, category_index: dict[str, int]) -> Any:
    """raters x items, NaN where a rater abstained, as krippendorff expects."""
    items = sorted(panel.item_ids)
    raters = sorted(panel.raters)
    matrix = np.full((len(raters), len(items)), np.nan)
    for r, rater in enumerate(raters):
        for c, item in enumerate(items):
            lbl = panel.raters[rater].get(item)
            if lbl in category_index:
                matrix[r, c] = category_index[lbl]
    return matrix


def test_fleiss_kappa_matches_statsmodels(panel: Panel, category_index: dict[str, int]) -> None:
    mine = fleiss_kappa(panel.raters).value
    theirs = float(inter_rater.fleiss_kappa(_counts_table(panel, category_index)))
    assert abs(mine - theirs) <= EXACT, f"judgecheck {mine!r} vs statsmodels {theirs!r}"


def test_krippendorff_alpha_matches_the_krippendorff_package(
    panel: Panel, category_index: dict[str, int]
) -> None:
    mine = krippendorff_alpha(panel.raters).value
    theirs = float(
        kd.alpha(
            reliability_data=_reliability_matrix(panel, category_index),
            level_of_measurement="nominal",
        )
    )
    assert abs(mine - theirs) <= ULP, f"judgecheck {mine!r} vs krippendorff {theirs!r}"


def test_every_pairwise_cohens_kappa_matches_statsmodels(
    panel: Panel, category_index: dict[str, int]
) -> None:
    items = sorted(panel.item_ids)
    mismatches: list[str] = []
    pairs = 0

    for a, b in itertools.combinations(sorted(panel.raters), 2):
        matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        for item in items:
            la, lb = panel.raters[a].get(item), panel.raters[b].get(item)
            if la in category_index and lb in category_index:
                matrix[category_index[la], category_index[lb]] += 1

        mine = cohens_kappa(panel.raters[a], panel.raters[b]).kappa
        theirs = float(inter_rater.cohens_kappa(matrix).kappa)
        pairs += 1
        if abs(mine - theirs) > EXACT:
            mismatches.append(f"{a}/{b}: {mine!r} vs {theirs!r}")

    assert pairs == 21, "expected every pair of the seven raters"
    assert not mismatches, "\n".join(mismatches)


def test_a_rubber_stamp_rater_scores_near_zero_in_both(category_index: dict[str, int]) -> None:
    """The README's central argument, checked against statsmodels rather than us.

    A rater that answers TP to almost everything on a mostly-TP corpus has high
    raw agreement and no discriminating power. If judgecheck and statsmodels
    both put it near zero, the claim is not an artifact of this implementation.
    """
    items = [f"i{n}" for n in range(20)]
    truthy = {item: ("TP" if n < 18 else "FP") for n, item in enumerate(items)}
    rubber_stamp = dict.fromkeys(items, "TP")

    result = cohens_kappa(truthy, rubber_stamp)
    assert result.agreement >= 0.9, "raw agreement is high"
    assert abs(result.kappa) < 1e-9, "kappa sees through it"

    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for item in items:
        matrix[category_index[truthy[item]], category_index[rubber_stamp[item]]] += 1
    theirs = float(inter_rater.cohens_kappa(matrix).kappa)
    assert abs(result.kappa - theirs) <= EXACT
