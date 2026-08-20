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
import random
from pathlib import Path
from typing import Any

import pytest

from judgecheck.agreement import cohens_kappa, fleiss_kappa, krippendorff_alpha
from judgecheck.independence import _largest_eigenvalue
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


#: A deliberately incomplete panel: raters skip different items, so the number
#: of raters per item varies. Module level rather than a class attribute so it
#: is not a mutable default (RUF012).
INCOMPLETE = {
    "a": {"i1": "TP", "i2": "FP", "i3": "TP", "i4": "FP", "i5": "TP"},
    "b": {"i1": "TP", "i2": "TP", "i3": "TP", "i5": "FP"},
    "c": {"i1": "FP", "i3": "TP", "i4": "FP", "i5": "TP"},
    "d": {"i1": "TP", "i2": "FP", "i4": "FP"},
}
TWO_LABELS = ("TP", "FP")


class TestIncompleteData:
    """The abstention-tolerant path, which the real panel never exercises.

    Every rater in `panel-real` labeled every item, so the reproduction and the
    cross-checks above only ever validate the complete-data case. The
    generalizations that tolerate a rater skipping an item are exactly where
    independent implementations legitimately diverge, and until this class they
    were checked only against the TypeScript harness, which shares an author.

    Krippendorff's alpha is designed for missing data, so the `krippendorff`
    package validates that path directly. Fleiss' kappa is not: statsmodels
    refuses unequal raters-per-item rather than computing a generalization, so
    that branch has no third-party check and the README says so.
    """

    def _matrix(self) -> Any:
        """Raters x items, NaN for "this rater did not label this item".

        Returns `Any` rather than `np.ndarray` because numpy is an optional
        extra here and a module-scope import would break the base install.
        """
        import numpy as np

        items = sorted({i for r in INCOMPLETE.values() for i in r})
        index = {lbl: i for i, lbl in enumerate(TWO_LABELS)}
        m = np.full((len(INCOMPLETE), len(items)), np.nan)
        for ri, name in enumerate(sorted(INCOMPLETE)):
            for ci, item in enumerate(items):
                value = INCOMPLETE[name].get(item)
                if value in index:
                    m[ri, ci] = index[value]
        return m

    def test_the_panel_really_is_incomplete(self) -> None:
        """Otherwise this class silently re-tests the complete-data path."""
        items = sorted({i for r in INCOMPLETE.values() for i in r})
        counts = [sum(1 for r in INCOMPLETE.values() if i in r) for i in items]
        assert len(set(counts)) > 1, "raters-per-item must vary for this to mean anything"

    def test_krippendorff_alpha_matches_the_package_on_missing_data(self) -> None:
        krippendorff = pytest.importorskip("krippendorff")
        mine = krippendorff_alpha(INCOMPLETE, TWO_LABELS).value
        theirs = krippendorff.alpha(reliability_data=self._matrix(), level_of_measurement="nominal")
        assert mine == pytest.approx(theirs, abs=1e-12)

    def test_statsmodels_declines_this_input_rather_than_disagreeing(self) -> None:
        """Pinned so the README's stated limit stays true.

        If a future statsmodels grows support for unequal raters-per-item, this
        fails and the README claim needs revisiting rather than silently
        becoming understated.
        """
        np = pytest.importorskip("numpy")
        sm = pytest.importorskip("statsmodels.stats.inter_rater")
        items = sorted({i for r in INCOMPLETE.values() for i in r})
        table = np.array(
            [
                [sum(1 for r in INCOMPLETE.values() if r.get(i) == lbl) for lbl in TWO_LABELS]
                for i in items
            ]
        )
        assert len({int(row.sum()) for row in table}) > 1
        with pytest.raises(AssertionError):
            sm.fleiss_kappa(table)


#: Eigenvalue cases, module level so it is not a mutable class attribute (RUF012).
EIGEN_CASES = {
    "dominant negative eigenvalue": [[-1.0, -3.0], [-3.0, -1.0]],
    "equicorrelation, ones is an eigenvector": [
        [1.0, 0.5, 0.5, 0.5],
        [0.5, 1.0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5],
        [0.5, 0.5, 0.5, 1.0],
    ],
    "identity": [[1.0, 0.0], [0.0, 1.0]],
    "negative equicorrelation": [
        [1.0, -0.3, -0.3],
        [-0.3, 1.0, -0.3],
        [-0.3, -0.3, 1.0],
    ],
    "single rater": [[1.0]],
}


class TestEigenvalue:
    """`_largest_eigenvalue` is hand-rolled, so it gets checked against numpy.

    The package has no runtime dependencies and one eigenvalue does not justify
    acquiring one, but a hand-rolled numerical routine with no external check is
    exactly the thing this file exists to prevent. numpy is already a test-only
    extra here.

    The first two cases are the ones that broke the first implementation: a
    matrix whose dominant eigenvalue is negative, and an equicorrelation matrix
    whose eigenvector is the vector of ones the iteration started from.
    """

    @pytest.mark.parametrize("name", sorted(EIGEN_CASES))
    def test_matches_numpy(self, name: str) -> None:
        np = pytest.importorskip("numpy")
        matrix = EIGEN_CASES[name]
        expected = float(np.linalg.eigvalsh(np.array(matrix)).max())
        assert _largest_eigenvalue(matrix) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("k", [5, 7, 9])
    def test_matches_numpy_on_random_correlation_matrices(self, k: int) -> None:
        np = pytest.importorskip("numpy")
        rng = random.Random(k)
        matrix = [[1.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                matrix[i][j] = matrix[j][i] = rng.uniform(-0.9, 0.9)
        expected = float(np.linalg.eigvalsh(np.array(matrix)).max())
        assert _largest_eigenvalue(matrix) == pytest.approx(expected, abs=1e-9)

    def test_the_real_panel_matrix(self) -> None:
        """The value the report actually publishes."""
        np = pytest.importorskip("numpy")
        from itertools import combinations

        from judgecheck.independence import _error_vectors, _phi

        panel = load_panel(PANEL_DIR)
        assert panel.truth is not None
        names = sorted(panel.raters)
        matrix = [[1.0 if i == j else 0.0 for j in names] for i in names]
        for i, j in combinations(range(len(names)), 2):
            ea, eb = _error_vectors(
                panel.raters[names[i]], panel.raters[names[j]], panel.truth, frozenset(LABELS)
            )
            value = _phi(ea, eb)
            if value is not None:
                matrix[i][j] = matrix[j][i] = value
        expected = float(np.linalg.eigvalsh(np.array(matrix)).max())
        assert _largest_eigenvalue(matrix) == pytest.approx(expected, abs=1e-9)
