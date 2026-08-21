"""How much of each number is real, at 23 items.

Every coefficient in this package is a point estimate from one panel. On a
panel this size the estimates move a lot under resampling, and a report that
prints `0.135` to three places without saying so invites a precision nobody
has earned.

That matters most where a number is used to decide something. `--fail-under`
and `--min-effective` turn an estimate into a build failure, and a threshold
sitting inside the interval means the gate is closer to a coin flip on this
data than to a measurement. The renderer says so when it happens, which is the
whole reason this module exists.

Method: nonparametric bootstrap over **items**, not over raters. Items are the
independent observations here; the raters are fixed and are the thing being
measured. Each draw resamples the item set with replacement, and a duplicate
item is renamed so it counts as a distinct observation rather than collapsing
back onto itself in the label maps.

Percentile intervals, not BCa. The bias correction needs a jackknife over items
that would cost more than it buys at this size, and a percentile interval is
the honest, conventional default. It is also slightly too narrow for skewed
statistics, which is worth knowing: `n_eff` is bounded above by `k`, so its
upper limit piles up at that bound and the interval is asymmetric by
construction rather than by accident.

What this does not fix: 23 items is 23 items. A wider interval is a more honest
report, not a better panel.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .agreement import fleiss_kappa, krippendorff_alpha
from .independence import panel_independence
from .types import LABELS, Interval, Panel

#: Resamples per interval. Enough for a stable 95% percentile interval without
#: making a report noticeably slow; the whole thing runs in well under a second
#: on a panel this size.
BOOTSTRAP_DRAWS = 1000

#: Fixed, because the full report is asserted bit-identical across process
#: hash seeds (`tests/test_determinism.py`). An unseeded bootstrap would make
#: every run differ in the last digits and break that guarantee for no gain.
#: The *seed* is what must not vary; `draws` may, and the determinism test
#: pins the default rather than the only possible value.
BOOTSTRAP_SEED = 20260819

#: Two-sided coverage. Not configurable: a caller-varied level would also break
#: the determinism guarantee, and 95% is the convention this report is read by.
CONFIDENCE = 0.95


@dataclass(frozen=True, kw_only=True)
class PanelIntervals:
    """Bootstrap intervals for the statistics a gate can be set on."""

    draws: int
    items: int
    fleiss: Interval
    krippendorff: Interval
    effective_raters: Interval | None
    """None when independence was not measurable on the panel as given."""
    effective_defined_in: int
    """Draws where `n_eff` was computable. A low count is itself a warning:
    it means the panel is close to having nothing to measure."""


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on a pre-sorted sequence.

    Deliberately not `statistics.quantiles`: this has to return one of the
    observed values so the interval endpoints are always attainable results
    rather than interpolations between them.
    """
    if not values:
        raise ValueError("no values")
    k = round(q * (len(values) - 1))
    return values[max(0, min(len(values) - 1, k))]


def _interval(point: float, draws: Sequence[float]) -> Interval:
    tail = (1.0 - CONFIDENCE) / 2
    ordered = sorted(draws)
    return Interval(
        point=point,
        low=_percentile(ordered, tail),
        high=_percentile(ordered, 1.0 - tail),
    )


def _resample(
    panel: Panel, items: Sequence[str], rng: random.Random
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """One bootstrap draw: items with replacement, duplicates kept distinct.

    Renaming a repeated item matters. Without it the second copy overwrites the
    first in each rater's label map and the draw silently shrinks, which biases
    every interval toward the small-sample end.
    """
    raters: dict[str, dict[str, str]] = {name: {} for name in panel.raters}
    truth: dict[str, str] = {}
    for position in range(len(items)):
        item = items[rng.randrange(len(items))]
        key = f"{item}\x00{position}"
        for name, labels in panel.raters.items():
            value = labels.get(item)
            if value is not None:
                raters[name][key] = value
        if panel.truth is not None:
            verdict = panel.truth.get(item)
            if verdict is not None:
                truth[key] = verdict
    return raters, truth


def bootstrap_intervals(
    panel: Panel,
    labels: Sequence[str] = LABELS,
    draws: int = BOOTSTRAP_DRAWS,
) -> PanelIntervals:
    """Percentile bootstrap intervals over resampled items."""
    label_tuple = tuple(labels)
    items = sorted({item for rater in panel.raters.values() for item in rater})

    point_fleiss = fleiss_kappa(panel.raters, label_tuple).value
    point_alpha = krippendorff_alpha(panel.raters, label_tuple).value
    point_independence = (
        panel_independence(panel.raters, panel.truth, label_tuple) if panel.truth else None
    )

    if not items or draws < 1:
        return PanelIntervals(
            draws=0,
            items=len(items),
            fleiss=Interval(point_fleiss, point_fleiss, point_fleiss),
            krippendorff=Interval(point_alpha, point_alpha, point_alpha),
            effective_raters=None,
            effective_defined_in=0,
        )

    rng = random.Random(BOOTSTRAP_SEED)
    fleiss_draws: list[float] = []
    alpha_draws: list[float] = []
    effective_draws: list[float] = []

    for _ in range(draws):
        raters, truth = _resample(panel, items, rng)
        fleiss_draws.append(fleiss_kappa(raters, label_tuple).value)
        alpha_draws.append(krippendorff_alpha(raters, label_tuple).value)
        if point_independence is not None and truth:
            drawn = panel_independence(raters, truth, label_tuple)
            if drawn.effective is not None:
                effective_draws.append(drawn.effective)

    effective: Interval | None = None
    if (
        point_independence is not None
        and point_independence.effective is not None
        and effective_draws
    ):
        # The conservative estimate, matching what the report and the gate use.
        effective = _interval(point_independence.effective, effective_draws)

    return PanelIntervals(
        draws=draws,
        items=len(items),
        fleiss=_interval(point_fleiss, fleiss_draws),
        krippendorff=_interval(point_alpha, alpha_draws),
        effective_raters=effective,
        effective_defined_in=len(effective_draws),
    )
