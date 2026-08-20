"""Chance-corrected agreement.

Why kappa and not accuracy: on an imbalanced label set, a rater that always
picks the majority class scores high accuracy while adding zero signal. Kappa
subtracts out the agreement you would get by chance, so a rubber-stamp rater
lands near zero. That is the correct way to validate an LLM judge.

Cohen's kappa is for two raters. For a panel, use Fleiss' kappa or
Krippendorff's alpha; averaging pairwise Cohen's kappa is not a defined
coefficient, though the mean is still a useful "agrees with everyone"
redundancy signal.

These implementations mirror the reference TypeScript harness exactly, and the
test suite proves it by reproducing that harness's published numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import LABELS, KappaResult, Labels, MultiRaterResult, PanelLabels


def _rater_list(raters: PanelLabels) -> list[Labels]:
    """Normalize either panel shape to a list of raters.

    `Panel.raters` is a name -> labels mapping and the report functions take it
    directly, so the panel statistics must too. Without this, passing the
    obvious thing iterates the dict's *keys* and fails several frames later with
    `'str' object has no attribute 'get'`, which points nowhere useful.

    Rater order does not affect Fleiss or Krippendorff; sorting only makes the
    traversal deterministic.
    """
    if isinstance(raters, Mapping):
        values = [raters[name] for name in sorted(raters)]
        if any(isinstance(v, str) for v in values):
            raise TypeError(
                "expected a panel -- {rater: {item: label}} or a sequence of label maps -- "
                "but got one rater's labels, {item: label}. "
                "For a single pair use cohens_kappa(a, b)."
            )
        return values
    return list(raters)


#: Interpretation for a coefficient computed over no comparable items. Not a
#: band: it marks the absence of a measurement, so that a degenerate panel does
#: not read as "poor" (which sounds like a finding) or "near perfect" (which
#: sounds like a pass).
UNDEFINED = "undefined (no comparable items)"

#: The other degenerate case, and a different one: there were plenty of items,
#: but every rating fell in a single category, so chance agreement is total and
#: the coefficient is 0/0. Kept distinct from UNDEFINED because the two call
#: for different fixes: one is a coverage problem, the other is a panel of
#: raters that never discriminated.
UNDEFINED_NO_VARIANCE = "undefined (all ratings in one category)"


def interpret_kappa(k: float) -> str:
    """Readability bands, matching the reference implementation exactly.

    Landis and Koch (1977), with their 0.00-0.20 `slight` band folded into
    `poor` and `almost perfect` renamed `near perfect`. These are a reading aid,
    not a statistical test: they carry no inferential standing and say nothing
    about whether a value differs significantly from zero.
    """
    if k < 0.2:
        return "poor"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "near perfect"


def cohens_kappa(a: Labels, b: Labels, labels: Sequence[str] = LABELS) -> KappaResult:
    """Cohen's kappa between two raters.

    Scored over items present in *both* raters' maps whose labels are both
    in-set. An item one rater skipped is not a disagreement, it is absent.
    """
    label_list = list(labels)
    index = {lbl: i for i, lbl in enumerate(label_list)}
    k = len(label_list)
    matrix = [[0] * k for _ in range(k)]

    n = 0
    for item, la in a.items():
        lb = b.get(item)
        if lb is None:
            continue
        ia, ib = index.get(la), index.get(lb)
        if ia is None or ib is None:
            continue
        matrix[ia][ib] += 1
        n += 1

    if n == 0:
        # No shared items. Not "poor" agreement, which reads as a finding; the
        # two raters were never compared. `group_agreement` and
        # `mean_pairwise_kappa` both skip these pairs for the same reason.
        return KappaResult(n=0, agreement=0.0, kappa=0.0, interpretation=UNDEFINED)

    agreed = sum(matrix[i][i] for i in range(k))
    p_o = agreed / n

    p_e = 0.0
    for i in range(k):
        row = sum(matrix[i])
        col = sum(matrix[j][i] for j in range(k))
        p_e += (row / n) * (col / n)

    if p_e >= 1:
        # Every rating fell in one category, so chance agreement is total and
        # kappa is 0/0. Reporting 1.0 "near perfect" here is the single worst
        # thing this module can do: two raters who both say TP to everything
        # are the textbook case kappa exists to catch, and 1.0 clears any
        # --fail-under. No variance is not perfect agreement.
        return KappaResult(n=n, agreement=p_o, kappa=0.0, interpretation=UNDEFINED_NO_VARIANCE)
    kappa = (p_o - p_e) / (1 - p_e)
    return KappaResult(n=n, agreement=p_o, kappa=kappa, interpretation=interpret_kappa(kappa))


def fleiss_kappa(raters: PanelLabels, labels: Sequence[str] = LABELS) -> MultiRaterResult:
    """Fleiss' kappa for a panel labeling the same items.

    Generalized to tolerate abstention: each item is scored over however many
    raters actually labeled it, and items with fewer than two ratings are
    skipped because agreement is undefined on them.
    """
    rater_list = _rater_list(raters)
    label_list = list(labels)
    index = {lbl: i for i, lbl in enumerate(label_list)}
    k = len(label_list)

    item_set: set[str] = set()
    for r in rater_list:
        item_set.update(r)
    # Sorted, not raw set order: string hashing is randomized per process, so
    # set iteration order varies between runs. Float addition is not
    # associative, which moved this result by ~2e-16 run to run. Irrelevant
    # numerically, unacceptable for a package that claims exact reproduction.
    items = sorted(item_set)

    category_totals = [0] * k
    total_assignments = 0
    p_bar_sum = 0.0
    used_items = 0

    for item in items:
        counts = [0] * k
        n_i = 0
        for r in rater_list:
            lbl = r.get(item)
            if lbl is None:
                continue
            idx = index.get(lbl)
            if idx is None:
                continue
            counts[idx] += 1
            n_i += 1
        if n_i < 2:
            continue
        sum_sq = 0
        for j in range(k):
            sum_sq += counts[j] * counts[j]
            category_totals[j] += counts[j]
        total_assignments += n_i
        p_bar_sum += (sum_sq - n_i) / (n_i * (n_i - 1))
        used_items += 1

    if used_items == 0 or total_assignments == 0:
        return MultiRaterResult(n=0, raters=len(rater_list), value=0.0, interpretation=UNDEFINED)

    p_bar = p_bar_sum / used_items
    p_e = sum((c / total_assignments) ** 2 for c in category_totals)
    if p_e >= 1:
        # As in cohens_kappa: one category used everywhere means kappa is 0/0,
        # not 1.0. A panel of rubber stamps scored "near perfect" and passed
        # --fail-under 0.9 while triage flagged every one of its raters DROP.
        return MultiRaterResult(
            n=used_items, raters=len(rater_list), value=0.0, interpretation=UNDEFINED_NO_VARIANCE
        )
    value = (p_bar - p_e) / (1 - p_e)
    return MultiRaterResult(
        n=used_items, raters=len(rater_list), value=value, interpretation=interpret_kappa(value)
    )


def krippendorff_alpha(raters: PanelLabels, labels: Sequence[str] = LABELS) -> MultiRaterResult:
    """Krippendorff's alpha, nominal metric.

    Like Fleiss it scores the whole panel, but it handles missing data
    correctly, so it is the safer choice when coverage is uneven.
    alpha = 1 - Do/De, from the coincidence matrix of every ordered pair of
    ratings within each item.
    """
    rater_list = _rater_list(raters)
    label_list = list(labels)
    index = {lbl: i for i, lbl in enumerate(label_list)}
    k = len(label_list)
    o = [[0.0] * k for _ in range(k)]

    item_set: set[str] = set()
    for r in rater_list:
        item_set.update(r)
    # Sorted, not raw set order: string hashing is randomized per process, so
    # set iteration order varies between runs. Float addition is not
    # associative, which moved this result by ~2e-16 run to run. Irrelevant
    # numerically, unacceptable for a package that claims exact reproduction.
    items = sorted(item_set)

    used_items = 0
    for item in items:
        vals: list[int] = []
        for r in rater_list:
            lbl = r.get(item)
            if lbl is None:
                continue
            idx = index.get(lbl)
            if idx is None:
                continue
            vals.append(idx)
        m_u = len(vals)
        if m_u < 2:
            continue
        used_items += 1
        for x in range(m_u):
            for y in range(m_u):
                if x == y:
                    continue
                o[vals[x]][vals[y]] += 1 / (m_u - 1)

    n_c = [sum(o[c]) for c in range(k)]
    n = sum(n_c)
    if used_items == 0 or n <= 1:
        # 0.0 with an explicit UNDEFINED, not 1.0. Alpha's limit here really is
        # 1.0 by its own algebra, but Fleiss reports 0.0 on the identical input,
        # so the two coefficients disagreed by a full unit on no data, printed
        # three lines apart. Worse, an alpha of 1.0 clears any --fail-under, so
        # a panel with nothing in it passed on one coefficient. Neither number
        # is a measurement; saying so is better than picking a side.
        return MultiRaterResult(
            n=used_items, raters=len(rater_list), value=0.0, interpretation=UNDEFINED
        )

    do_sum = 0.0
    de_sum = 0.0
    for c in range(k):
        for j in range(k):
            if c == j:
                continue
            do_sum += o[c][j]
            de_sum += n_c[c] * n_c[j]

    if de_sum == 0:
        # Expected disagreement is zero because only one category appears. The
        # `krippendorff` package raises here ("There has to be more than one
        # value in the domain") rather than returning 1.0, which is the same
        # judgement reached a different way.
        return MultiRaterResult(
            n=used_items, raters=len(rater_list), value=0.0, interpretation=UNDEFINED_NO_VARIANCE
        )
    value = 1 - (n - 1) * (do_sum / de_sum)
    return MultiRaterResult(
        n=used_items, raters=len(rater_list), value=value, interpretation=interpret_kappa(value)
    )


def pairwise_kappa(
    raters: Mapping[str, Labels], labels: Sequence[str] = LABELS
) -> dict[tuple[str, str], KappaResult]:
    """Cohen's kappa for every unordered pair, keyed by (a, b) with a < b."""
    names = sorted(raters)
    out: dict[tuple[str, str], KappaResult] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[(a, b)] = cohens_kappa(raters[a], raters[b], labels)
    return out


def mean_pairwise_kappa(
    raters: Mapping[str, Labels], labels: Sequence[str] = LABELS
) -> dict[str, float | None]:
    """Per-rater mean of its Cohen's kappa against every other rater.

    Not a panel statistic. It answers "does this rater agree with everyone?",
    which flags a rater adding little independent signal.

    Pairs with no shared items are skipped rather than averaged in as 0.0.
    Including them was a real defect: a rater that agrees perfectly with the one
    other rater it overlaps reported 0.5 because a third, disjoint rater
    contributed an undefined 0.0. `None` when a rater has no comparable pair at
    all, because "agrees with nobody" and "was never compared" are different
    facts and only one of them is about the rater.
    """
    pairs = pairwise_kappa(raters, labels)
    sums: dict[str, list[float]] = {name: [] for name in raters}
    for (a, b), res in pairs.items():
        if res.n == 0:
            continue
        sums[a].append(res.kappa)
        sums[b].append(res.kappa)
    return {name: (sum(v) / len(v) if v else None) for name, v in sums.items()}
