"""One computed view of a panel, and two ways to render it.

`build_report()` does every calculation once and returns a frozen result.
`render_text()` and `to_dict()` are pure formatters over that result. Keeping
them apart matters: if the human-readable output and the JSON output each
recomputed their own numbers, they could disagree, and a report that disagrees
with itself is worse than no report.

The reference harness spreads this across three separate entry points that each
reload the panel. This is the same content, computed once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agreement import (
    fleiss_kappa,
    krippendorff_alpha,
    mean_pairwise_kappa,
    pairwise_kappa,
)
from .consensus import MAJORITY, SPLIT, UNANIMOUS, UNSCORED, consensus, split_items
from .independence import (
    CAUTION_EFFICIENCY,
    CoincidentError,
    GroupAgreement,
    GroupLabels,
    PanelIndependence,
    coincident_errors,
    group_agreement,
    panel_independence,
    rater_groups_from_panel,
)
from .triage import BALANCED, UNKNOWN, RaterTriage, split_leaning, triage
from .types import LABELS, ConsensusEntry, KappaResult, MultiRaterResult, Panel, RaterValidity
from .uncertainty import CONFIDENCE, PanelIntervals, bootstrap_intervals
from .validity import POSITIVE, validity


@dataclass(frozen=True)
class PanelReport:
    """Everything judgecheck knows about one panel."""

    panel: str
    raters: tuple[str, ...]
    items: int
    labels: tuple[str, ...]
    positive_label: str

    fleiss: MultiRaterResult
    krippendorff: MultiRaterResult
    pairwise: Mapping[tuple[str, str], KappaResult]
    mean_pairwise: Mapping[str, float | None]

    consensus: tuple[ConsensusEntry, ...]
    triage: Mapping[str, RaterTriage]
    leaning: Mapping[str, tuple[Mapping[str, int], str]]

    validity: Mapping[str, RaterValidity] | None
    """None when the panel has no adjudicated truth."""

    independence: PanelIndependence | None
    """None when the panel has no adjudicated truth.

    Effective judge count is defined on error correlation, so without truth
    there is nothing to compute. The earlier label-agreement version of this
    statistic was directionally wrong; see `independence` module docstring.
    """
    coincidence: CoincidentError | None
    """None when the panel has no adjudicated truth. Coincident error is
    defined against truth, not against the other raters."""
    groups: GroupAgreement | None
    """None when no grouping was supplied and none could be read off the
    panel's `vendor` fields."""
    intervals: PanelIntervals | None
    """Bootstrap intervals, when asked for. Off by default because they cost a
    second, and always on when a gate is set: a threshold decided from a point
    estimate should be shown next to the range that estimate actually spans."""

    @property
    def consensus_counts(self) -> dict[str, int]:
        counts = {UNANIMOUS: 0, MAJORITY: 0, SPLIT: 0, UNSCORED: 0}
        for entry in self.consensus:
            counts[entry.consensus] += 1
        return counts

    @property
    def splits(self) -> tuple[ConsensusEntry, ...]:
        return split_items(self.consensus)


def build_report(
    panel: Panel,
    labels: Sequence[str] = LABELS,
    positive_label: str = POSITIVE,
    groups: GroupLabels | None = None,
    intervals: bool = False,
) -> PanelReport:
    """Compute every statistic for a panel, once.

    `groups` is an optional rater -> group mapping for the group-agreement
    section. When omitted it is read off the panel's `vendor` fields, which is
    a convenience rather than an endorsement of vendor as the right cut; pass
    your own to group by base architecture, prompt template, or anything else.
    """
    label_tuple = tuple(labels)
    if positive_label not in label_tuple:
        # The CLI checks this; a library caller used to get a validity table of
        # zeros with no indication that the label it was scoring did not exist.
        raise ValueError(
            f"positive_label {positive_label!r} is not in the label set ({', '.join(label_tuple)})"
        )
    entries = consensus(panel.raters, label_tuple)
    resolved_groups = rater_groups_from_panel(panel) if groups is None else dict(groups)

    return PanelReport(
        panel=panel.name,
        raters=panel.rater_names,
        items=len(panel.item_ids),
        labels=label_tuple,
        positive_label=positive_label,
        fleiss=fleiss_kappa(panel.raters, label_tuple),
        krippendorff=krippendorff_alpha(panel.raters, label_tuple),
        pairwise=pairwise_kappa(panel.raters, label_tuple),
        mean_pairwise=mean_pairwise_kappa(panel.raters, label_tuple),
        consensus=entries,
        triage=triage(panel.raters, panel.truth, label_tuple),
        leaning=split_leaning(
            panel.raters, tuple(e.finding_id for e in split_items(entries)), label_tuple
        ),
        validity=(validity(panel.raters, panel.truth, positive_label) if panel.truth else None),
        independence=(
            panel_independence(panel.raters, panel.truth, label_tuple) if panel.truth else None
        ),
        coincidence=(
            coincident_errors(panel.raters, panel.truth, label_tuple) if panel.truth else None
        ),
        groups=(
            group_agreement(panel.raters, resolved_groups, label_tuple) if resolved_groups else None
        ),
        intervals=(bootstrap_intervals(panel, label_tuple) if intervals else None),
    )


@dataclass(frozen=True)
class Gate:
    """The result of checking a panel against the thresholds CI asked for.

    Two independent checks, either of which may be left off:

    `threshold` gates on agreement, and checks both defined panel coefficients,
    failing if either is short. Gating on one alone lets a panel pass on
    whichever happens to be kinder, and the two disagreeing is itself worth
    stopping for.

    `min_effective` gates on how many independent judges the panel is actually
    worth. That is a different question from agreement, and in one direction
    the answers invert: a panel can clear a kappa floor precisely because its
    judges are redundant. Reporting both keeps that visible.

    judgecheck has no default for either. What counts as enough depends on what
    the panel decides and what a wrong call costs, so the numbers have to be
    supplied deliberately.
    """

    threshold: float | None
    fleiss: float
    krippendorff: float
    coefficients_defined: bool = True
    """False when either panel coefficient was computed over nothing.

    Both degenerate branches in `agreement` return 0.0, which is the right
    value but is not a measurement. Without this flag `--fail-under 0.0` passed
    a panel of rubber stamps: the gate read `.value` and never `.interpretation`,
    so a report printing "undefined (all ratings in one category)" three
    sections above still exited 0.
    """
    min_effective: float | None = None
    effective: float | None = None
    """None when independence was not measurable (no truth, or no overlap)."""
    raters: int = 0

    @property
    def failures(self) -> tuple[str, ...]:
        short: list[str] = []
        if self.threshold is not None and not self.coefficients_defined:
            short.append(
                "agreement is undefined (no comparable items, or every rating in "
                "one category), so no threshold can be met"
            )
        elif self.threshold is not None:
            if self.fleiss < self.threshold:
                short.append(f"Fleiss' kappa {self.fleiss:.3f} < {self.threshold:.3f}")
            if self.krippendorff < self.threshold:
                short.append(f"Krippendorff's alpha {self.krippendorff:.3f} < {self.threshold:.3f}")
        if self.min_effective is not None:
            if self.effective is None:
                # Fail, do not skip. A panel whose independence cannot be
                # measured has not met the floor, and treating "unmeasurable"
                # as "passed" is how a gate silently stops gating.
                short.append(
                    f"effective judges not measurable, so a floor of "
                    f"{self.min_effective:.2f} cannot be met "
                    f"(needs adjudicated truth and overlapping items)"
                )
            elif self.effective < self.min_effective:
                short.append(
                    f"effective judges {self.effective:.2f} of {self.raters} "
                    f"< {self.min_effective:.2f}"
                )

        return tuple(short)

    @property
    def passed(self) -> bool:
        return not self.failures


def check_gate(
    report: PanelReport,
    threshold: float | None = None,
    min_effective: float | None = None,
) -> Gate:
    """Compare a panel against an agreement floor, an independence floor, or both.

    Passing neither yields a Gate that trivially passes. The CLI never does
    that: it builds a gate only when at least one threshold was requested, so
    "no gate" and "a gate with nothing to check" stay distinguishable.
    """
    return Gate(
        threshold=threshold,
        fleiss=report.fleiss.value,
        krippendorff=report.krippendorff.value,
        coefficients_defined=not (
            report.fleiss.interpretation.startswith("undefined")
            or report.krippendorff.interpretation.startswith("undefined")
        ),
        min_effective=min_effective,
        effective=(report.independence.effective if report.independence else None),
        raters=(report.independence.raters if report.independence else len(report.raters)),
    )


def to_dict(report: PanelReport, gate: Gate | None = None) -> dict[str, Any]:
    """JSON-shaped view. Floats are left unrounded on purpose.

    Rounding here would make the output useless for the thing it most needs to
    support: checking that another implementation produces the same number.
    """
    out: dict[str, Any] = {
        "panel": report.panel,
        "raters": list(report.raters),
        "items": report.items,
        "labels": list(report.labels),
        "positiveLabel": report.positive_label,
        "agreement": {
            "fleissKappa": {
                "value": report.fleiss.value,
                "n": report.fleiss.n,
                "raters": report.fleiss.raters,
                "interpretation": report.fleiss.interpretation,
            },
            "krippendorffAlpha": {
                "value": report.krippendorff.value,
                "n": report.krippendorff.n,
                "raters": report.krippendorff.raters,
                "interpretation": report.krippendorff.interpretation,
            },
            "pairwiseKappa": [
                {
                    "a": a,
                    "b": b,
                    "kappa": r.kappa,
                    "agreement": r.agreement,
                    "n": r.n,
                    "interpretation": r.interpretation,
                }
                for (a, b), r in sorted(report.pairwise.items())
            ],
            "meanPairwiseKappa": dict(sorted(report.mean_pairwise.items())),
        },
        "consensus": {
            "counts": report.consensus_counts,
            "items": [
                {
                    "findingId": e.finding_id,
                    "consensus": e.consensus,
                    "consensusLabel": e.consensus_label,
                    "labels": dict(e.labels),
                }
                for e in report.consensus
            ],
        },
        "triage": {
            name: {
                "labeled": t.labeled,
                "distribution": dict(t.distribution),
                "abstention": t.abstention,
                "agreementWithTruth": t.agreement_with_truth,
                "meanRedundancy": t.mean_redundancy,
                "maxRedundancy": t.max_redundancy,
                "maxRedundancyWith": t.max_redundancy_with,
                "flags": list(t.flags),
                "recommendation": t.recommendation,
                "splitLeaning": report.leaning[name][1],
                "splitVotes": dict(report.leaning[name][0]),
            }
            for name, t in sorted(report.triage.items())
        },
    }

    if report.intervals is not None:
        iv = report.intervals
        out["intervals"] = {
            "method": "percentile bootstrap over items",
            "draws": iv.draws,
            "confidence": CONFIDENCE,
            "fleissKappa": {
                "point": iv.fleiss.point,
                "low": iv.fleiss.low,
                "high": iv.fleiss.high,
            },
            "krippendorffAlpha": {
                "point": iv.krippendorff.point,
                "low": iv.krippendorff.low,
                "high": iv.krippendorff.high,
            },
            "effectiveRaters": (
                {
                    "point": iv.effective_raters.point,
                    "low": iv.effective_raters.low,
                    "high": iv.effective_raters.high,
                    "definedInDraws": iv.effective_defined_in,
                }
                if iv.effective_raters is not None
                else None
            ),
        }

    if report.independence is not None:
        ind = report.independence
        out["independence"] = {
            "basis": "error correlation vs adjudicated truth",
            "raters": ind.raters,
            "excludedRaters": list(ind.excluded_raters),
            "comparablePairs": ind.pairs,
            "incomparablePairs": ind.incomparable_pairs,
            "meanPhi": ind.mean_phi,
            "phiSd": ind.phi_sd,
            "effectiveRatersKish": ind.effective_raters,
            "effectiveRatersEigen": ind.effective_raters_eigen,
            "effectiveRaters": ind.effective,
            "efficiency": ind.efficiency,
            "exchangeable": ind.exchangeable,
            "saturated": ind.saturated,
            "interpretation": ind.interpretation,
            "cautionBelow": CAUTION_EFFICIENCY,
        }

    if report.groups is not None:
        g = report.groups
        out["groups"] = {
            "members": {name: list(ms) for name, ms in g.groups.items()},
            "ungrouped": list(g.ungrouped),
            "withinPairs": g.within_pairs,
            "betweenPairs": g.between_pairs,
            "withinKappa": g.within,
            "betweenKappa": g.between,
            "delta": g.delta,
        }

    if report.coincidence is not None:
        c = report.coincidence
        out["coincidentError"] = {
            "meanPhi": c.mean_phi,
            "permutationP": c.p_value,
            "worstPair": (
                {
                    "a": c.worst.a,
                    "b": c.worst.b,
                    "excess": c.worst.excess,
                    "lift": c.worst.lift,
                    "phi": c.worst.phi,
                }
                if c.worst is not None
                else None
            ),
            "pairs": [
                {
                    "a": pc.a,
                    "b": pc.b,
                    "n": pc.n,
                    "aWrong": pc.a_wrong,
                    "bWrong": pc.b_wrong,
                    "bothWrong": pc.both_wrong,
                    "observed": pc.observed,
                    "expected": pc.expected,
                    "lift": pc.lift,
                    "excess": pc.excess,
                    "phi": pc.phi,
                }
                for pc in c.pairs
            ],
        }

    if gate is not None:
        out["gate"] = {
            "threshold": gate.threshold,
            "fleissKappa": gate.fleiss,
            "krippendorffAlpha": gate.krippendorff,
            "minEffectiveRaters": gate.min_effective,
            "effectiveRaters": gate.effective,
            "passed": gate.passed,
            "failures": list(gate.failures),
        }

    if report.validity is not None:
        out["validity"] = {
            name: {
                "caught": v.caught,
                "truthPositives": v.truth_positives,
                "called": v.called,
                "correctCalls": v.correct_calls,
                "recall": v.recall,
                "precision": v.precision,
                "f1": v.f1,
            }
            for name, v in sorted(report.validity.items())
        }
    return out


def to_json(report: PanelReport, gate: Gate | None = None, indent: int | None = 2) -> str:
    return json.dumps(to_dict(report, gate), indent=indent, sort_keys=False)


def _bar(label: str, width: int = 74) -> str:
    return f"{label} " + "─" * max(0, width - len(label) - 1)


def render_text(report: PanelReport, gate: Gate | None = None) -> str:
    """Human-readable report. Percentages are display only; JSON keeps full precision."""
    lines: list[str] = []
    add = lines.append

    add(f"panel: {report.panel}   {len(report.raters)} raters x {report.items} items")
    add("")

    add(_bar("PANEL AGREEMENT"))
    f, k = report.fleiss, report.krippendorff
    if report.intervals is not None:
        fi, ki = report.intervals.fleiss, report.intervals.krippendorff
        add(
            f"  Fleiss' kappa        {f.value:+.3f}  [{fi.low:+.3f}, {fi.high:+.3f}]  "
            f"({f.interpretation})   n={f.n}, raters={f.raters}"
        )
        add(
            f"  Krippendorff's alpha {k.value:+.3f}  [{ki.low:+.3f}, {ki.high:+.3f}]  "
            f"({k.interpretation})"
        )
        add("")
        add(
            f"  Brackets are {CONFIDENCE:.0%} percentile bootstrap intervals over "
            f"{report.intervals.draws} resamples"
        )
        add(f"  of the {report.intervals.items} items scored by this panel.")
    else:
        add(
            f"  Fleiss' kappa        {f.value:+.3f}  ({f.interpretation})   "
            f"n={f.n}, raters={f.raters}"
        )
        add(f"  Krippendorff's alpha {k.value:+.3f}  ({k.interpretation})")
    add("")
    add("  Accuracy would flatter these judges; kappa subtracts the agreement")
    add("  they would reach by chance. Low kappa means the panel is not")
    add("  measuring the same thing, whatever its accuracy says.")
    add("")

    if report.independence is not None:
        ind = report.independence
        add(_bar("PANEL INDEPENDENCE"))
        if ind.mean_phi is None or ind.efficiency is None or ind.effective is None:
            add(f"  not measurable  ({ind.incomparable_pairs} pairs had nothing to compare)")
            add("")
            add("  A pair is incomparable when it scored too few items in common, or when")
            add("  one of the two was never wrong on the items it did score. Either way")
            add("  there is no error pattern to correlate. That is absence of evidence,")
            add("  not independence, so it is reported as such rather than as a number")
            add("  that would pass a gate.")
        else:
            add(
                f"  mean error correlation  {ind.mean_phi:+.3f}  (sd {ind.phi_sd:.3f})"
                f"   over {ind.pairs} pairs"
            )
            add(f"  effective judges, Kish  {ind.effective_raters:.2f} of {ind.raters}")
            add(f"  effective judges, eigen {ind.effective_raters_eigen:.2f} of {ind.raters}")
            add(
                f"  reported                {ind.effective:.2f} of {ind.raters}"
                f"   ({ind.efficiency * 100:.0f}% of nominal, {ind.interpretation})"
            )
            if report.intervals is not None and report.intervals.effective_raters is not None:
                band = report.intervals.effective_raters
                add(f"  {'':<23} {CONFIDENCE:.0%} interval [{band.low:.2f}, {band.high:.2f}]")
            if ind.excluded_raters:
                add(
                    f"  excluded                {', '.join(ind.excluded_raters)}"
                    "  (no comparable pair)"
                )
            if ind.incomparable_pairs:
                add(f"  pairs not comparable    {ind.incomparable_pairs}")
            add("")
            add("  Kish's design effect, n_eff = k / (1 + (k-1) * phi), over the correlation")
            add("  between judges' binary error vectors. It answers how many independent")
            add("  checks the panel's vote rests on, not how many judges cast it.")
            add("")
            if not ind.exchangeable:
                add("  The two estimators disagree, so no single number describes this panel.")
                add("  Kish averages the pairwise correlations, which assumes they are")
                add("  roughly equal; the eigenvalue form assumes nothing and is the")
                add("  robustness check the source paper runs beside its own headline.")
                add("  When they part company the panel has structure an average hides,")
                add("  usually blocks of judges that fail on different items. The lower")
                add("  figure is the one reported, because a gate should not pass on an")
                add("  assumption the data violates.")
                add("")
            if ind.saturated:
                add("  Mean correlation is at or below zero, so the Kish figure is pinned")
                add("  at its ceiling by the clamp rather than measured. Every panel with")
                add("  a non-positive mean reports the same Kish number, which is why the")
                add("  eigenvalue form matters here.")
                add("")
            add("  Errors, not labels. Agreement counts 'both right' and 'both wrong' the")
            add("  same way, and only the second one makes a panel weaker than its size.")
            add("")
            add("  This is not a quality score. Judges can be independently wrong, which")
            add("  reads as high independence and is still a bad panel. Read it next to")
            add("  the validity table, never instead of it.")
            if ind.efficiency < CAUTION_EFFICIENCY:
                add("")
                add(
                    f"  Below {CAUTION_EFFICIENCY:.0%} of nominal. Treat panel votes with caution "
                    "(Kohli 2026)."
                )
        add("")

    if report.groups is not None:
        grp = report.groups
        add(_bar("JUDGE GROUPS"))
        for name, members in grp.groups.items():
            add(f"  {name:<14} {', '.join(members)}")
        if grp.ungrouped:
            add(f"  {'(ungrouped)':<14} {', '.join(grp.ungrouped)}")
        add("")
        add(f"  within-group pairs   {grp.within_pairs}")
        add(f"  between-group pairs  {grp.between_pairs}")
        if grp.within is None or grp.between is None:
            add("")
            if grp.within_pairs == 0:
                add("  Every judge is in a different group, so there is no within-group")
                add("  agreement to compare against. That is worth seeing rather than")
                add("  hiding: a panel can be maximally group-diverse and still contain")
                add("  a strongly correlated pair, which is what the section below")
                add("  measures directly.")
        elif grp.delta is not None:
            # within and between are both non-None here, so delta is too; the
            # explicit check is for the type checker, not for the logic.
            add(
                f"  within  {grp.within:+.3f}   between  {grp.between:+.3f}   "
                f"delta {grp.delta:+.3f}"
            )
            add("")
            add("  Descriptive, and nothing gates on it. The one study to measure a")
            add("  developer-family split directly found a gap of 0.047, and its three")
            add("  most correlated pairs were on the cross-family side (Kohli 2026).")
            add("  A delta from one panel is a fact about that panel.")
        add("")

    if report.coincidence is not None and report.coincidence.pairs:
        coin = report.coincidence
        add(_bar("COINCIDENT ERROR vs adjudicated truth"))
        if coin.worst is None:
            add("  no pair has enough comparable items to assess")
        else:
            w = coin.worst
            add(
                f"  worst pair    {w.a} + {w.b}   "
                f"both wrong on {w.both_wrong}/{w.n}, "
                f"{w.excess:+.1f} above independent"
            )
            if coin.p_value is not None:
                verdict = "unlikely by chance" if coin.p_value <= 0.05 else "consistent with chance"
                # The add-one estimator keeps p strictly positive precisely so
                # the report never claims p = 0, and rounding to three places
                # would put it back. Below the resolution of the test, say so.
                shown = f"{coin.p_value:.3f}" if coin.p_value >= 0.0005 else "<0.001"
                add(
                    f"  permutation p {shown}   ({verdict}, "
                    f"corrected for picking the worst of {len(coin.pairs)} pairs)"
                )
            if w.phi is not None:
                add(f"  error correlation  {w.phi:+.3f}")
        add("")
        add("  Ranked by excess joint errors, not by ratio. The ratio is capped by the")
        add("  more accurate judge's error count, so ranking on it picks out pairs that")
        add("  contain an accurate judge rather than pairs that fail together.")
        add("")
        add("  With many pairs, some pair always looks alarming. The p value is the")
        add("  share of reshuffled panels whose worst pair was at least this bad, so it")
        add("  already accounts for the fact that a maximum was selected.")
        add("")

    if report.validity is not None:
        add(_bar(f"VALIDITY vs adjudicated truth  (positive = {report.positive_label})"))
        add(f"  {'rater':<10} {'recall':>16} {'precision':>16} {'F1':>8}")
        for name in report.raters:
            v = report.validity[name]
            # A dash, not 0%. No truth positives to catch, or no calls made, is
            # an empty question rather than a failed one.
            rec = (
                f"{v.caught}/{v.truth_positives} ({v.recall * 100:.0f}%)"
                if v.recall is not None
                else f"{v.caught}/{v.truth_positives} (-)"
            )
            pre = (
                f"{v.correct_calls}/{v.called} ({v.precision * 100:.0f}%)"
                if v.precision is not None
                else f"{v.correct_calls}/{v.called} (-)"
            )
            f1 = f"{v.f1:>8.2f}" if v.f1 is not None else f"{'-':>8}"
            add(f"  {name:<10} {rec:>16} {pre:>16} {f1}")
        add("")

    add(_bar("CONSENSUS"))
    c = report.consensus_counts
    total = len(report.consensus)
    for kind in (UNANIMOUS, MAJORITY, SPLIT, UNSCORED):
        if kind == UNSCORED and not c[kind]:
            continue  # the normal case; do not add a zero row to every report
        share = (c[kind] / total * 100) if total else 0.0
        add(f"  {kind:<10} {c[kind]:>3}  ({share:.0f}%)")
    if report.splits:
        add("")
        add(f"  contested items ({len(report.splits)}): these need human adjudication")
        for e in report.splits:
            votes = ", ".join(f"{n}={lbl}" for n, lbl in sorted(e.labels.items()))
            add(f"    {e.finding_id}: {votes}")
    add("")

    add(_bar("RATER TRIAGE"))
    add(f"  {'rater':<10} {'mean k':>7} {'abstain':>8}  recommendation")
    for name in report.raters:
        t = report.triage[name]
        abstain = f"{t.abstention * 100:>7.0f}%" if t.abstention is not None else "      -"
        # A dash, not 0.000. A rater with no comparable pair was never measured
        # against the panel, which is different from disagreeing with it.
        mean_k = report.mean_pairwise[name]
        shown_k = f"{mean_k:>7.3f}" if mean_k is not None else f"{'-':>7}"
        add(f"  {name:<10} {shown_k} {abstain}  {t.recommendation}")
        for flag in t.flags:
            add(f"             - {flag}")
    add("")
    add("  'mean k' is agreement with the rest of the panel, not correctness.")
    add("  High mean kappa can mean redundant rather than good.")

    leaning_lines = [
        f"  {name:<10} {report.leaning[name][1]}"
        for name in report.raters
        if report.leaning[name][1] not in (BALANCED, UNKNOWN)
    ]
    if leaning_lines:
        add("")
        add(_bar("THRESHOLD ON CONTESTED ITEMS"))
        lines.extend(leaning_lines)

    if gate is not None:
        add("")
        add(_bar("GATE"))
        if gate.passed:
            # Name the checks that actually ran. Either threshold may be absent,
            # and reporting "both coefficients passed" for a run that only gated
            # on independence would be a false statement about what was checked.
            checks: list[str] = []
            if gate.threshold is not None:
                checks.append(f"both coefficients are at or above {gate.threshold:.3f}")
            if gate.min_effective is not None:
                checks.append(
                    f"the panel is worth {gate.effective:.2f} judges, "
                    f"at or above {gate.min_effective:.2f}"
                )
            add(f"  PASS  {'; '.join(checks) if checks else 'nothing was checked'}")
        else:
            for failure in gate.failures:
                add(f"  FAIL  {failure}")

        # A threshold inside the interval is the case worth naming. The gate
        # still passes or fails as asked, but a reader deciding whether to
        # trust that verdict should know it turned on a number this panel
        # cannot resolve. Silence here is what makes a green badge misleading.
        if report.intervals is not None:
            straddles: list[str] = []
            iv = report.intervals
            if gate.threshold is not None:
                for name, band in (
                    ("Fleiss' kappa", iv.fleiss),
                    ("Krippendorff's alpha", iv.krippendorff),
                ):
                    if band.contains(gate.threshold):
                        straddles.append(
                            f"{gate.threshold:.3f} falls inside the {name} interval "
                            f"[{band.low:+.3f}, {band.high:+.3f}]"
                        )
            if (
                gate.min_effective is not None
                and iv.effective_raters is not None
                and iv.effective_raters.contains(gate.min_effective)
            ):
                straddles.append(
                    f"{gate.min_effective:.2f} falls inside the effective-judges "
                    f"interval [{iv.effective_raters.low:.2f}, "
                    f"{iv.effective_raters.high:.2f}]"
                )
            for note in straddles:
                add(f"  NOTE  {note};")
                add("        this panel cannot resolve that threshold, so the verdict")
                add("        is closer to a coin flip than to a measurement.")

    return "\n".join(lines)
