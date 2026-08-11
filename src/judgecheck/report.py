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
from .consensus import MAJORITY, SPLIT, UNANIMOUS, consensus, split_items
from .triage import RaterTriage, split_leaning, triage
from .types import LABELS, ConsensusEntry, KappaResult, MultiRaterResult, Panel, RaterValidity
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
    mean_pairwise: Mapping[str, float]

    consensus: tuple[ConsensusEntry, ...]
    triage: Mapping[str, RaterTriage]
    leaning: Mapping[str, tuple[Mapping[str, int], str]]

    validity: Mapping[str, RaterValidity] | None
    """None when the panel has no adjudicated truth."""

    @property
    def consensus_counts(self) -> dict[str, int]:
        counts = {UNANIMOUS: 0, MAJORITY: 0, SPLIT: 0}
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
) -> PanelReport:
    """Compute every statistic for a panel, once."""
    label_tuple = tuple(labels)
    entries = consensus(panel.raters, label_tuple)

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
    )


@dataclass(frozen=True)
class Gate:
    """The result of checking a panel against a minimum agreement threshold.

    Checks both defined panel coefficients and fails if either is short.
    Gating on one alone lets a panel pass on whichever happens to be kinder,
    and the two disagreeing is itself worth stopping for.

    judgecheck has no default threshold. What counts as enough agreement
    depends on what the panel decides and what a wrong call costs, so the
    number has to be supplied deliberately.
    """

    threshold: float
    fleiss: float
    krippendorff: float

    @property
    def failures(self) -> tuple[str, ...]:
        short: list[str] = []
        if self.fleiss < self.threshold:
            short.append(f"Fleiss' kappa {self.fleiss:.3f} < {self.threshold:.3f}")
        if self.krippendorff < self.threshold:
            short.append(f"Krippendorff's alpha {self.krippendorff:.3f} < {self.threshold:.3f}")
        return tuple(short)

    @property
    def passed(self) -> bool:
        return not self.failures


def check_gate(report: PanelReport, threshold: float) -> Gate:
    """Compare a panel's agreement against a minimum threshold."""
    return Gate(
        threshold=threshold,
        fleiss=report.fleiss.value,
        krippendorff=report.krippendorff.value,
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

    if gate is not None:
        out["gate"] = {
            "threshold": gate.threshold,
            "fleissKappa": gate.fleiss,
            "krippendorffAlpha": gate.krippendorff,
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
    add(f"  Fleiss' kappa        {f.value:+.3f}  ({f.interpretation})   n={f.n}, raters={f.raters}")
    add(f"  Krippendorff's alpha {k.value:+.3f}  ({k.interpretation})")
    add("")
    add("  Accuracy would flatter these judges; kappa subtracts the agreement")
    add("  they would reach by chance. Low kappa means the panel is not")
    add("  measuring the same thing, whatever its accuracy says.")
    add("")

    if report.validity is not None:
        add(_bar(f"VALIDITY vs adjudicated truth  (positive = {report.positive_label})"))
        add(f"  {'rater':<10} {'recall':>16} {'precision':>16} {'F1':>8}")
        for name in report.raters:
            v = report.validity[name]
            rec = f"{v.caught}/{v.truth_positives} ({v.recall * 100:.0f}%)"
            pre = f"{v.correct_calls}/{v.called} ({v.precision * 100:.0f}%)"
            add(f"  {name:<10} {rec:>16} {pre:>16} {v.f1:>8.2f}")
        add("")

    add(_bar("CONSENSUS"))
    c = report.consensus_counts
    total = len(report.consensus)
    for kind in (UNANIMOUS, MAJORITY, SPLIT):
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
        add(
            f"  {name:<10} {report.mean_pairwise[name]:>7.3f} "
            f"{t.abstention * 100:>7.0f}%  {t.recommendation}"
        )
        for flag in t.flags:
            add(f"             - {flag}")
    add("")
    add("  'mean k' is agreement with the rest of the panel, not correctness.")
    add("  High mean kappa can mean redundant rather than good.")

    leaning_lines = [
        f"  {name:<10} {report.leaning[name][1]}"
        for name in report.raters
        if report.leaning[name][1] != "balanced"
    ]
    if leaning_lines:
        add("")
        add(_bar("THRESHOLD ON CONTESTED ITEMS"))
        lines.extend(leaning_lines)

    if gate is not None:
        add("")
        add(_bar("GATE"))
        if gate.passed:
            add(f"  PASS  both coefficients are at or above {gate.threshold:.3f}")
        else:
            for failure in gate.failures:
                add(f"  FAIL  {failure}")

    return "\n".join(lines)
