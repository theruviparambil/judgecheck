"""How many independent judges a panel actually contains.

A panel vote is only worth what its judges independently contribute. Five
judges that fail on the same items are not five checks on a claim, they are one
check reported five times, and averaging them produces a confident number with
no more evidence behind it than a single call.

The quantity that matters is **error** correlation, not agreement. Agreement
counts "both right" and "both wrong" identically, and only the second one means
the panel is weaker than its size suggests. Two judges who agree because they
are both correct are not redundant in any way that should worry you.

This distinction is not academic; it inverts the answer. An earlier version of
this module built its effective-judge count from mean pairwise Cohen's kappa on
labels, and that statistic is maximized by a panel of judges who share nothing
and minimized by a panel that is uniformly right. The README tells that story
with numbers under "Errors, not agreement"; what matters here is why the code
looks the way it does.

Kohli, "Nine Judges, Two Effective Votes" (arXiv:2605.29800, 2026), the source
of both the formula and the threshold used below, is explicit about this:

    "For binary error vectors, the phi coefficient reduces to the Pearson
    product-moment correlation, which is the quantity the Kish formula requires
    ... Alternative association measures (e.g., Cohen's kappa) conflate
    prevalence with dependence; phi isolates the linear dependence that directly
    degrades majority-vote performance."

So `panel_independence` requires adjudicated truth and correlates error vectors.
There is no label-only fallback, because the honest fallback is to report
nothing: without truth you can measure how much your judges repeat each other
(`agreement.mean_pairwise_kappa`, and the redundancy flag in `triage`) but you
cannot measure whether they fail together, and only the second is what `n_eff`
claims to describe.

What this module provides:

  * `panel_independence` converts mean pairwise error correlation into an
    effective number of judges, by Kish's design effect.
  * `coincident_errors` reports, per pair, how often the two are wrong together
    against what independent errors predict, with a permutation test that
    accounts for the fact that a worst pair is being selected out of many.
  * `group_agreement` reports within-group against between-group agreement for
    whatever grouping you supply. It is a description of your panel, not a rule,
    and nothing gates on it.

On that last point: the obvious feature here is to group judges by developer and
treat cross-family panels as trustworthy. Kohli measured that directly and found
its two same-family pairs correlated at phi = 0.437 and 0.435 against a
cross-family mean of 0.389, a gap of 0.047; the three most correlated pairs in
that study were all *cross-family*; and restricting the panel to one judge per
family made effective independence worse rather than better, n_eff falling from
2.18 to 1.93. Goel et al., "Great Models Think Alike and this Undermines AI
Oversight" (arXiv:2502.04313, 2025), give a mechanism: judges score models higher
the more *functionally* similar those models are to themselves, after
controlling for capability, and functional similarity does not follow company
lines. Hence: measure, do not compose.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from .agreement import cohens_kappa
from .types import LABELS, Interval, Labels, Panel, PanelLabels

#: A rater name -> group name mapping. The group is whatever cut you want to
#: examine: the developer, the base architecture, the serving provider, the
#: prompt template. Deliberately not called `vendor` or `family`. `vendor`
#: already means two other things (who serves the model, and the EU AI Act's
#: "provider"), and `family` presumes a grouping by developer, which is the cut
#: the measured evidence supports least (see the module docstring).
GroupLabels = Mapping[str, str]

#: Below this share of nominally-available judges, treat panel votes with
#: caution. From Kohli (2026), who proposes it as a reporting default rather
#: than a hard rule, and who defines it on error correlation. It is a reading
#: aid with no inferential standing, like the bands in `interpret_kappa`.
CAUTION_EFFICIENCY = 0.5

#: Two-sided coverage for the null band, matching `uncertainty.CONFIDENCE`.
CONFIDENCE = 0.95

#: How far the two effective-judge estimators may differ, as a ratio, before
#: the panel is reported as non-exchangeable and the single-number summary is
#: withheld. Kohli's own panel agrees within 1%; a factor of 1.25 is generous.
EXCHANGEABILITY_TOLERANCE = 1.25

#: Minimum items a pair must jointly score before its error correlation means
#: anything. Two points always correlate perfectly or not at all.
MIN_PAIR_ITEMS = 3

#: Default permutation draws for the coincident-error significance test.
#:
#: The default is fixed and `tests/test_determinism.py` pins it, which is what
#: the reproducibility guarantee actually rests on: same input and same
#: settings give the same bytes. Passing a different count is still
#: deterministic, it just answers a slightly different question, so the
#: parameter exists as an escape hatch on large panels. Cost is roughly
#: `draws * pairs`, and pairs grow as the square of the judge count.
PERMUTATIONS = 2000

#: Seed for that test. Fixed for the same reason.
PERMUTATION_SEED = 20260819

#: Draws used to calibrate the effective-judge figure against its own null.
#: Cheap: 500 takes about a third of a second on a 7x23 panel, and the null
#: median moves less than 0.002 between 200 and 500.
NULL_DRAWS = 500


@dataclass(frozen=True, kw_only=True)
class PanelIndependence:
    """What a panel's judges are jointly worth, in units of independent judges."""

    raters: int
    """Raters contributing at least one comparable pair.

    Not the number handed in. A rater whose every pair is incomparable, such as
    a perfect oracle with no errors to correlate, contributes no measurement,
    and counting it in `k` let it add a whole effective judge to the total.
    """
    excluded_raters: tuple[str, ...]
    """Raters dropped from `raters` for contributing no comparable pair."""
    pairs: int
    """Pairs with a defined error correlation."""
    incomparable_pairs: int
    """Pairs skipped: too little joint coverage, or one rater never varied."""
    mean_phi: float | None
    """Mean pairwise phi over error vectors. `None` when nothing is comparable."""
    phi_sd: float | None
    """Spread of those pairwise correlations, and the reason both estimators
    below are reported. Kish's formula assumes exchangeability, i.e. roughly
    equal pairwise correlations. A large SD next to a small mean means the
    assumption does not hold and the Kish number should not be read alone."""
    effective_raters: float | None
    """Kish design effect. Assumes exchangeability."""
    effective_raters_eigen: float | None
    """`k / lambda_max` of the correlation matrix. Makes no exchangeability
    assumption, and is the robustness check the source paper runs alongside
    its own headline figure."""
    efficiency: float | None
    """The **lower** of the two estimators over `raters`, 0-1.

    Deliberately the conservative one. When the two disagree the panel is not
    exchangeable, and in that case the Kish figure is the optimistic reading;
    a gate should not pass on the strength of an assumption the data violates.
    """
    saturated: bool
    """True when `mean_phi <= 0`, so the clamp pinned the Kish estimate at `k`.

    Worth surfacing: because every non-positive mean maps to exactly zero,
    Kish efficiency is identically 1.0 for any such panel and cannot
    distinguish between them. A saturated reading is a ceiling, not a
    measurement.
    """
    exchangeable: bool
    """Whether the two estimators agree closely enough to report one number."""
    null_efficiency: Interval | None
    """What `efficiency` looks like for judges that are independent by construction.

    This is the number that makes the headline readable, and without it the
    headline is wrong. `k / lambda_max` is biased downward at small item counts:
    the largest eigenvalue of a correlation matrix estimated from few
    observations is inflated by sampling noise alone (the Marchenko-Pastur
    edge), so even perfectly independent judges do not score 100%.

    On a 7-judge, 23-item panel the null median is 0.54, not 1.0. Reporting
    "31% of nominal" against an implied ceiling of 100% therefore overstates the
    dependence by roughly a factor of two, and the 50% caution threshold fires
    on about a quarter of independent panels of this shape.

    Computed by permuting each judge's errors across the items it scored,
    holding its error count fixed, which is the same null `coincident_errors`
    uses. `None` when independence was not measurable.
    """
    p_value: float | None
    """Share of null draws whose efficiency was at or below the observed one.

    Small means the panel is more correlated than independent judges of this
    size and accuracy would be. This is the inferential claim; the raw
    percentage is descriptive.
    """
    interpretation: str

    @property
    def effective(self) -> float | None:
        """The conservative effective judge count. This is what gates read.

        `efficiency` already carries the lower of the two estimators, so this
        is it back in units of judges.
        """
        if self.efficiency is None:
            return None
        return self.efficiency * self.raters


@dataclass(frozen=True, kw_only=True)
class PairCoincidence:
    """How often two raters are wrong on the same item."""

    a: str
    b: str
    n: int
    """Items where both raters and truth all carry an in-set label."""
    a_wrong: int
    b_wrong: int
    both_wrong: int
    observed: float
    """`both_wrong / n`."""
    expected: float
    """What `observed` would be if the two raters' errors were independent."""
    lift: float | None
    """`observed / expected`, or `None` when `expected` is 0.

    Read with care, and do not rank by it. Because `both_wrong` cannot exceed
    `min(a_wrong, b_wrong)`, lift is capped at `n / max(a_wrong, b_wrong)`, so
    the pairs scoring highest are the ones whose more-accurate member has the
    fewest errors. On the panel in this repository the four highest-lift pairs
    all sit exactly at their structural ceiling, which makes that ranking a
    property of the metric rather than of the judges. `excess` is what the
    report sorts on.
    """
    excess: float
    """`both_wrong` minus the count independence predicts. Additive, uncapped."""
    phi: float | None
    """Correlation of the two error vectors. `None` when undefined."""


@dataclass(frozen=True, kw_only=True)
class CoincidentError:
    """Coincident-error analysis across every pair in a panel."""

    pairs: tuple[PairCoincidence, ...]
    mean_phi: float | None
    worst: PairCoincidence | None
    """Largest excess joint error. `None` when no pair is comparable."""
    p_value: float | None
    """Permutation p for `worst`, correcting for selection over all pairs.

    Each judge's errors are reshuffled across items independently, holding its
    error count fixed, and the largest excess anywhere in the panel is recorded.
    `p` is the share of draws whose maximum matched or beat the observed one.
    Testing the maximum rather than one nominated pair is the point: with 21
    pairs, some pair looks alarming in most random panels.
    """


@dataclass(frozen=True, kw_only=True)
class GroupAgreement:
    """Within-group against between-group agreement, for a supplied grouping.

    Descriptive only. Nothing in judgecheck gates on `delta`, and the renderer
    prints a reference point next to it, because a delta measured on one panel
    is not evidence of a general law and has repeatedly been read as one.
    """

    groups: Mapping[str, tuple[str, ...]]
    ungrouped: tuple[str, ...]
    """Raters absent from the mapping. Excluded from both sides, not guessed at."""
    within_pairs: int
    between_pairs: int
    within: float | None
    between: float | None
    delta: float | None
    """`within - between`. `None` when either side has no pairs to average."""


def _named(raters: PanelLabels) -> dict[str, Labels]:
    """Normalize either panel shape to a name -> labels mapping.

    Every public function in this module runs this, so all of them accept both
    shapes. `types.PanelLabels` documents why: taking a mapping in one function
    and a sequence in its neighbour is a trap, because a dict iterates its keys
    and the failure surfaces somewhere unrelated. That warning was written about
    `agreement`, and an earlier version of this module reintroduced the same
    defect by accepting a sequence in one function and not in the other two.
    """
    if isinstance(raters, Mapping):
        values = {name: raters[name] for name in sorted(raters)}
        if any(isinstance(v, str) for v in values.values()):
            # The docstring above has always described this trap; until now it
            # did not check for it, and the failure surfaced as
            # `AttributeError: 'str' object has no attribute 'get'` from inside
            # a statistic, which is verbatim the error it warns about.
            raise TypeError(
                "expected a panel -- {rater: {item: label}} or a sequence of label maps -- "
                "but got one rater's labels, {item: label}. "
                "For a single pair use cohens_kappa(a, b)."
            )
        return values
    return {f"rater{i}": r for i, r in enumerate(raters)}


def _error_vectors(
    a: Labels, b: Labels, truth: Labels, label_set: frozenset[str]
) -> tuple[list[int], list[int]]:
    """Aligned 0/1 error vectors over items both raters and truth scored in-set.

    An item a rater skipped is absent rather than an error, matching
    `cohens_kappa`. Truth items outside the label set are not scoreable.
    """
    ea: list[int] = []
    eb: list[int] = []
    for item in sorted(truth):
        t = truth[item]
        if t not in label_set:
            continue
        va, vb = a.get(item), b.get(item)
        if va not in label_set or vb not in label_set:
            continue
        ea.append(int(va != t))
        eb.append(int(vb != t))
    return ea, eb


def _phi(ea: Sequence[int], eb: Sequence[int]) -> float | None:
    """Pearson correlation of two binary vectors, which for 0/1 data is phi.

    `None` when undefined: too few items, or one vector is constant. A judge
    that was never wrong on the overlap has no error pattern to correlate, and
    substituting 0.0 there would read as "independent" on no evidence.
    """
    n = len(ea)
    if n < MIN_PAIR_ITEMS:
        return None
    ma = sum(ea) / n
    mb = sum(eb) / n
    da = math.sqrt(sum((x - ma) ** 2 for x in ea))
    db = math.sqrt(sum((y - mb) ** 2 for y in eb))
    if da == 0.0 or db == 0.0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ea, eb, strict=True)) / (da * db)


def _largest_eigenvalue(matrix: Sequence[Sequence[float]], iters: int = 1000) -> float:
    """Largest eigenvalue of a symmetric matrix, by power iteration.

    Written out rather than pulled from numpy because this package has no
    runtime dependencies and one eigenvalue does not justify acquiring one.
    Matches `numpy.linalg.eigvalsh(...).max()` to about 1e-12.

    Two details, both of which were wrong in the first version:

    The matrix is shifted so every eigenvalue is non-negative, because power
    iteration converges to the eigenvalue of largest *magnitude* and a
    correlation matrix with negative entries can have a dominant negative one.
    The shift is the Gershgorin bound, the largest absolute row sum, rather
    than a guess at the scale.

    The starting vector must not be an eigenvector. A vector of ones is exactly
    the eigenvector of `1 + (k-1)*rho` for any equicorrelation matrix, so the
    iteration would sit on that eigenvalue forever and return it whether or not
    it was the largest. The ramp below is deterministic, which the report's
    reproducibility guarantee requires, and asymmetric, which the mathematics
    requires.
    """
    n = len(matrix)
    if n == 0:
        return 0.0
    if n == 1:
        return float(matrix[0][0])
    shift = max(sum(abs(value) for value in row) for row in matrix)
    vector = [1.0 + i / n for i in range(n)]
    norm = sum(x * x for x in vector) ** 0.5
    vector = [x / norm for x in vector]
    value = 0.0
    for _ in range(iters):
        product = [
            sum((matrix[i][j] + (shift if i == j else 0.0)) * vector[j] for j in range(n))
            for i in range(n)
        ]
        length = sum(x * x for x in product) ** 0.5
        if length == 0.0:
            return -shift
        vector = [x / length for x in product]
        if abs(length - value) < 1e-13:
            value = length
            break
        value = length
    return value - shift


def interpret_efficiency(efficiency: float) -> str:
    """Readability band for `PanelIndependence.efficiency`.

    A reading aid, not a test. Like the kappa bands, these carry no inferential
    standing; they exist so a number in a CI log is legible at a glance.
    """
    if efficiency >= 0.8:
        return "near independent"
    if efficiency >= CAUTION_EFFICIENCY:
        return "moderately correlated"
    return "highly correlated"


def effective_raters(k: int, mean_phi: float) -> float:
    """Kish's effective sample size for `k` judges whose errors correlate at `mean_phi`.

        n_eff = k / (1 + (k - 1) * rho)

    `rho` must be a Pearson correlation of error vectors. The averaging is exact
    rather than approximate: for `k` unit-variance variables the variance of the
    mean is `(1/k)(1 + (k-1) * rho_bar)` where `rho_bar` is precisely the mean of
    the off-diagonal correlations, so heterogeneous pair correlations are handled
    correctly by averaging them.

    `mean_phi` is clamped to [0, 1]. Negative mean correlation is real and good,
    and Kish's framework does admit a design effect below 1, but reporting
    "worth more than seven of seven judges" invites over-reading a small sample.
    The unclamped value stays on `PanelIndependence.mean_phi` so nothing is
    hidden. NaN is treated as no information and returns `k` unchanged rather
    than propagating into a gate.
    """
    if k < 2:
        return float(k)
    if math.isnan(mean_phi):
        # Callers reach the not-measurable branch instead; this is the library
        # entry point's own guard. Returning k here would send a failed
        # computation through --min-effective as a pass, which is the exact
        # failure this package fixes elsewhere.
        return float("nan")
    rho = min(1.0, max(0.0, mean_phi))
    return k / (1 + (k - 1) * rho)


def _efficiency_from_errors(
    errors: Mapping[str, set[str]], names: Sequence[str], items: Sequence[str]
) -> float | None:
    """Eigenvalue-based efficiency for one configuration of judge errors."""
    k = len(names)
    if k < 2:
        return None
    matrix = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    defined = 0
    for i, j in combinations(range(k), 2):
        ea = [1 if item in errors[names[i]] else 0 for item in items]
        eb = [1 if item in errors[names[j]] else 0 for item in items]
        value = _phi(ea, eb)
        if value is not None:
            matrix[i][j] = matrix[j][i] = value
            defined += 1
    if defined == 0:
        return None
    lambda_max = _largest_eigenvalue(matrix)
    if lambda_max <= 0:
        return 1.0
    return min(1.0, max(1.0 / k, (k / lambda_max) / k))


def _null_efficiency(
    named: Mapping[str, Labels],
    names: Sequence[str],
    truth: Labels,
    label_set: frozenset[str],
    observed: float,
    draws: int = NULL_DRAWS,
) -> tuple[Interval | None, float | None]:
    """Where `observed` sits against judges that are independent by construction.

    `k / lambda_max` is biased downward when the item count is small relative to
    the judge count. The largest eigenvalue of a correlation matrix estimated
    from few observations is inflated by sampling noise alone, so a panel of
    genuinely independent judges does not score 100%. On the panel in this
    repository the null median is 0.54.

    Reporting the raw percentage without this is the same category of error the
    module docstring describes twice already: a number that looks like a
    measurement of the judges and is substantially a measurement of the
    estimator. So the null is computed and printed beside it.

    Same null as `coincident_errors`: each judge keeps its error count and has
    those errors redistributed across the items it scored. Seeded, because the
    report is asserted bit-identical across process hash seeds.
    """
    items = [
        item
        for item in sorted(truth)
        if truth[item] in label_set and any(named[n].get(item) in label_set for n in names)
    ]
    if len(items) < MIN_PAIR_ITEMS or len(names) < 2:
        return None, None

    flags = {
        n: {
            item: int(named[n][item] != truth[item])
            for item in items
            if named[n].get(item) in label_set
        }
        for n in names
    }

    rng = random.Random(PERMUTATION_SEED)
    drawn: list[float] = []
    for _ in range(draws):
        value = _efficiency_from_errors(_permuted(flags, rng), names, items)
        if value is not None:
            drawn.append(value)
    if not drawn:
        return None, None

    drawn.sort()
    tail = (1.0 - CONFIDENCE) / 2

    def q(fraction: float) -> float:
        index = round(fraction * (len(drawn) - 1))
        return drawn[max(0, min(len(drawn) - 1, index))]

    band = Interval(point=q(0.5), low=q(tail), high=q(1.0 - tail))
    # Add-one, as elsewhere: 500 draws cannot support a claim of exactly zero.
    hits = sum(1 for value in drawn if value <= observed)
    return band, (hits + 1) / (len(drawn) + 1)


def panel_independence(
    raters: PanelLabels,
    truth: Labels,
    labels: Sequence[str] = LABELS,
    null_draws: int = 0,
) -> PanelIndependence:
    """Effective judge count from pairwise error correlation.

    Requires adjudicated truth. Without it there is no error vector to
    correlate, and the label-agreement substitute is not merely less precise but
    directionally wrong (see the module docstring).

    `null_draws` calibrates the reported efficiency against judges that are
    independent by construction, which the percentage needs in order to be
    readable: this estimator does not reach 100% on independent judges when the
    item count is small. Off by default because it costs about a third of a
    second; `build_report` enables it wherever a gate or an interval is being
    computed.

    Two estimators are reported, following the source paper: Kish's design
    effect over the mean correlation, and `k / lambda_max` over the correlation
    matrix. The first assumes exchangeability; the second does not. They agree
    on a panel whose judges correlate about equally with one another and
    diverge sharply on one that splits into blocks, which is exactly the case
    a single averaged number cannot describe.
    """
    named = _named(raters)
    label_set = frozenset(labels)

    phi: dict[tuple[str, str], float] = {}
    incomparable = 0
    for a, b in combinations(list(named), 2):
        ea, eb = _error_vectors(named[a], named[b], truth, label_set)
        value = _phi(ea, eb)
        if value is None:
            incomparable += 1
        else:
            phi[(a, b)] = value

    # Only raters that produced at least one comparable pair count toward k.
    # Counting the rest inflates n_eff for free: appending a perfect oracle,
    # whose every pair is undefined because it never erred, previously raised
    # this panel from 7.00 of 7 to 8.00 of 8.
    contributing = sorted({name for pair in phi for name in pair})
    excluded = tuple(n for n in named if n not in contributing)
    k = len(contributing)

    if k < 2 or not phi:
        # Not "independent". Nothing was measurable, and a panel of judges who
        # never scored the same item must not report full independence and then
        # pass a gate on the strength of having no overlap.
        return PanelIndependence(
            raters=k,
            excluded_raters=excluded,
            pairs=0,
            incomparable_pairs=incomparable,
            mean_phi=None,
            phi_sd=None,
            effective_raters=None,
            effective_raters_eigen=None,
            efficiency=None,
            saturated=False,
            exchangeable=True,
            null_efficiency=None,
            p_value=None,
            interpretation="not measurable",
        )

    values = list(phi.values())
    mean_phi = sum(values) / len(values)
    sd = (
        (sum((v - mean_phi) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        if len(values) > 1
        else 0.0
    )

    kish = effective_raters(k, mean_phi)
    if math.isnan(kish):
        return PanelIndependence(
            raters=k,
            excluded_raters=excluded,
            pairs=len(phi),
            incomparable_pairs=incomparable,
            mean_phi=None,
            phi_sd=None,
            effective_raters=None,
            effective_raters_eigen=None,
            efficiency=None,
            saturated=False,
            exchangeable=True,
            null_efficiency=None,
            p_value=None,
            interpretation="not measurable",
        )

    # Undefined pairs enter the matrix as 0: no measured relationship. That is
    # the same convention the mean uses by omitting them, and it keeps the
    # matrix square without inventing a correlation.
    matrix = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    index = {name: i for i, name in enumerate(contributing)}
    for (a, b), value in phi.items():
        i, j = index[a], index[b]
        matrix[i][j] = matrix[j][i] = value
    lambda_max = _largest_eigenvalue(matrix)
    eigen = k / lambda_max if lambda_max > 0 else float(k)
    eigen = min(float(k), max(1.0, eigen))

    # Calibrate the reported figure against judges that are independent by
    # construction. Without this the percentage is read against an implied
    # ceiling of 100%, which k / lambda_max does not reach at small item counts.
    # Off by default: 500 draws cost about a third of a second, which is fine
    # once and not fine on every call in a test suite. `build_report` turns it
    # on wherever a decision is being made, which is the same rule the bootstrap
    # intervals follow.
    null_efficiency, p_value = (
        _null_efficiency(named, contributing, truth, label_set, min(kish, eigen) / k, null_draws)
        if null_draws > 0
        else (None, None)
    )

    lo, hi = min(kish, eigen), max(kish, eigen)
    exchangeable = hi <= lo * EXCHANGEABILITY_TOLERANCE
    efficiency = lo / k
    interpretation = (
        interpret_efficiency(efficiency)
        if exchangeable
        else "not exchangeable (estimators disagree)"
    )

    return PanelIndependence(
        raters=k,
        excluded_raters=excluded,
        pairs=len(phi),
        incomparable_pairs=incomparable,
        mean_phi=mean_phi,
        phi_sd=sd,
        effective_raters=kish,
        effective_raters_eigen=eigen,
        efficiency=efficiency,
        saturated=mean_phi <= 0.0,
        exchangeable=exchangeable,
        null_efficiency=null_efficiency,
        p_value=p_value,
        interpretation=interpretation,
    )


def _excess(a_wrong: int, b_wrong: int, both_wrong: int, n: int) -> float:
    """Joint errors above what independence predicts. Additive, so uncapped."""
    return both_wrong - (a_wrong * b_wrong / n if n else 0.0)


def _permuted(vectors: Mapping[str, Mapping[str, int]], rng: random.Random) -> dict[str, set[str]]:
    """One draw: each judge's errors reshuffled across the items it scored.

    The null is "these judges are exactly as accurate as observed, and fail
    independently of one another". Each judge keeps its error count and its
    coverage; only which items it got wrong varies.

    **This assumes items are exchangeable in difficulty**, and that assumption
    is load-bearing. Where some items are much harder than others, two judges
    failing on the same hard items is expected under independence, and this
    null will charge that co-occurrence to judge dependence instead. Kohli
    avoids the problem by stratifying the same permutation on *human-entropy*
    strata, "preserving per-judge error rates and the difficulty structure".
    That requires an external difficulty measure, from human annotations rather
    than from the judges being tested.

    judgecheck has no such measure, and deriving one from the judges is
    circular: stratifying on how many judges erred per item conditions on the
    dependence under test, and in the limit forces the permutation to reproduce
    the observed matrix and returns p = 1 for a perfectly correlated pair. So
    the null is left unstratified and the assumption is stated rather than
    silently patched. Read a significant result as "correlated beyond what
    equal accuracy alone predicts", not as "correlated beyond item difficulty".
    """
    out: dict[str, set[str]] = {}
    for name, errs in vectors.items():
        scored = sorted(errs)
        count = sum(errs.values())
        out[name] = set(rng.sample(scored, count)) if count else set()
    return out


def _max_excess(
    errors: Mapping[str, set[str]],
    coverage: Mapping[str, set[str]],
    pairs: Sequence[tuple[str, str]],
) -> float:
    """Largest pairwise excess joint error, each pair on its own overlap."""
    best = float("-inf")
    for a, b in pairs:
        shared = coverage[a] & coverage[b]
        n = len(shared)
        if n < MIN_PAIR_ITEMS:
            continue
        a_wrong = len(errors[a] & shared)
        b_wrong = len(errors[b] & shared)
        both = len(errors[a] & errors[b] & shared)
        best = max(best, _excess(a_wrong, b_wrong, both, n))
    return best


def coincident_errors(
    raters: PanelLabels,
    truth: Labels,
    labels: Sequence[str] = LABELS,
    permutations: int = PERMUTATIONS,
) -> CoincidentError:
    """How often each pair of raters is wrong on the same item.

    Agreement counts "both right" and "both wrong" identically, and only the
    second one means the panel's vote is weaker than its size suggests. This
    measures the second directly.

    The panel is ranked by `excess`, not `lift`. Lift is capped by the more
    accurate rater's error count, so ranking by it selects for pairs containing
    an accurate judge rather than for pairs that fail together. `p_value` then
    asks whether the largest excess anywhere in the panel exceeds what
    reshuffling produces, which is the question a reader actually has once they
    have been shown the worst of many pairs.
    """
    if permutations < 1:
        # A public parameter, so it gets checked. `permutations=0` returned
        # p = 1.0 ("consistent with chance") and a negative count returned a
        # negative p, which the renderer printed as "<0.001 (unlikely by
        # chance)" -- the opposite of what it meant.
        raise ValueError(f"permutations must be at least 1, got {permutations}")

    named = _named(raters)
    names = list(named)
    label_set = frozenset(labels)

    # Per-rater coverage and errors over every item that rater could be scored
    # on. The permutation below draws on these, so each pair's null uses the
    # same items as the pair's own statistic. An earlier version drew the null
    # on the complete-case intersection instead, so on a panel with uneven
    # coverage the p value described a different item set, and a pair that was
    # wrong together on every single shared item reported "consistent with
    # chance".
    coverage: dict[str, set[str]] = {}
    error_flags: dict[str, dict[str, int]] = {}
    for name in names:
        scored = {
            item
            for item, verdict in truth.items()
            if verdict in label_set and named[name].get(item) in label_set
        }
        coverage[name] = scored
        error_flags[name] = {item: int(named[name][item] != truth[item]) for item in scored}

    pairs: list[PairCoincidence] = []
    phis: list[float] = []
    for a, b in combinations(names, 2):
        ea, eb = _error_vectors(named[a], named[b], truth, label_set)
        n = len(ea)
        a_wrong, b_wrong = sum(ea), sum(eb)
        both = sum(1 for x, y in zip(ea, eb, strict=True) if x and y)
        observed = both / n if n else 0.0
        expected = (a_wrong / n) * (b_wrong / n) if n else 0.0
        value = _phi(ea, eb)
        if value is not None:
            phis.append(value)
        pairs.append(
            PairCoincidence(
                a=a,
                b=b,
                n=n,
                a_wrong=a_wrong,
                b_wrong=b_wrong,
                both_wrong=both,
                observed=observed,
                expected=expected,
                # expected == 0 means one rater was never wrong on the overlap.
                # There is no error pattern to correlate, so the ratio is
                # undefined rather than infinite or zero.
                lift=(observed / expected if expected > 0 else None),
                excess=_excess(a_wrong, b_wrong, both, n),
                phi=value,
            )
        )

    comparable = [p for p in pairs if p.n >= MIN_PAIR_ITEMS]
    # max() on ties returns the first, and `pairs` is built from sorted names,
    # so the winner is stable across runs.
    worst = max(comparable, key=lambda p: p.excess) if comparable else None
    mean_phi = sum(phis) / len(phis) if phis else None

    p_value: float | None = None
    comparable_names = [(pc.a, pc.b) for pc in comparable]
    if worst is not None and comparable_names:
        observed_max = _max_excess(
            {name: {i for i, e in flags.items() if e} for name, flags in error_flags.items()},
            coverage,
            comparable_names,
        )
        rng = random.Random(PERMUTATION_SEED)
        hits = 0
        for _ in range(permutations):
            drawn = _permuted(error_flags, rng)
            if _max_excess(drawn, coverage, comparable_names) >= observed_max:
                hits += 1
        # Add-one estimator: a p of exactly 0 is not a claim 2000 draws support.
        p_value = (hits + 1) / (permutations + 1)

    return CoincidentError(pairs=tuple(pairs), mean_phi=mean_phi, worst=worst, p_value=p_value)


def group_agreement(
    raters: PanelLabels,
    groups: GroupLabels,
    labels: Sequence[str] = LABELS,
) -> GroupAgreement:
    """Mean pairwise kappa within groups against across them.

    Supplied as a description of a panel, never as a rule for building one. The
    one study that measured a developer-family split directly found a gap of
    0.047 and found its three most correlated pairs on the *cross*-family side,
    so a delta from a single panel should be read as a fact about that panel.

    This one is label agreement rather than error correlation on purpose. It
    describes how similarly groups of judges *behave*, which is answerable
    without truth, and it feeds no gate.

    Raters missing from `groups` are excluded from both sides and listed in
    `ungrouped`. Inventing a singleton group for them would silently move
    every one of their pairs onto the between-group side and shrink the delta.
    """
    named = _named(raters)
    names = list(named)
    ungrouped = tuple(n for n in names if n not in groups)
    grouped = [n for n in names if n in groups]

    members: dict[str, list[str]] = {}
    for name in grouped:
        members.setdefault(groups[name], []).append(name)

    within: list[float] = []
    between: list[float] = []
    for a, b in combinations(grouped, 2):
        result = cohens_kappa(named[a], named[b], labels)
        if result.n == 0:
            continue  # no shared items: not agreement, absence of comparison
        (within if groups[a] == groups[b] else between).append(result.kappa)

    # None, not 0.0, when a side is empty. On a panel where every rater comes
    # from a different developer there are no within-group pairs at all, and
    # "0.0" would read as "within-group agreement is zero", which is a
    # different and false claim.
    within_mean = sum(within) / len(within) if within else None
    between_mean = sum(between) / len(between) if between else None
    delta = (
        within_mean - between_mean if within_mean is not None and between_mean is not None else None
    )

    return GroupAgreement(
        groups={g: tuple(sorted(ms)) for g, ms in sorted(members.items())},
        ungrouped=ungrouped,
        within_pairs=len(within),
        between_pairs=len(between),
        within=within_mean,
        between=between_mean,
        delta=delta,
    )


def rater_groups_from_panel(panel: Panel) -> dict[str, str]:
    """Derive a grouping from the `vendor` field the panel files already carry.

    A convenience default, not a recommendation that vendor is the right cut.
    It is simply the grouping the data hands you for free. If a rater's rows
    disagree on vendor, the first non-empty value wins and the rest are
    ignored: a mixed rater file is a data problem, and picking a winner here
    rather than raising keeps a malformed annotation from taking down a report
    whose statistics do not depend on it.

    Raters with no vendor on any row are left out, which surfaces downstream as
    `GroupAgreement.ungrouped`.
    """
    if panel.judgments is None:
        return {}
    out: dict[str, str] = {}
    for name, judgments in panel.judgments.items():
        for j in judgments:
            if j.vendor:
                out[name] = j.vendor
                break
    return out
