"""Which raters to keep, and which add noise.

The statistics say how much a panel agrees. This says what to do about it.
Each rater collects flags, and the flag count becomes a recommendation:

    0 flags   KEEP
    1 flag    REVIEW
    2+ flags  DROP / DOWN-WEIGHT

The flags are deliberately about *discriminating power*, not correctness alone.
A rater can be perfectly accurate and still worthless: one that says TP to
everything on a corpus that is mostly TP scores well and tells you nothing.

Note the direction of the redundancy signal. HIGH pairwise kappa between two
raters means they are near-identical, so one of them is enough. It is not a
quality score, and it is not the panel-agreement statistic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from .agreement import pairwise_kappa
from .types import LABELS, Labels
from .validity import accuracy

#: The most-used label may not exceed this share of a rater's labels.
SKEW_THRESHOLD = 0.8
#: NEEDS_INVESTIGATION may not exceed this share.
ABSTENTION_THRESHOLD = 0.4
#: Exact-match agreement with truth below this is flagged.
ACCURACY_THRESHOLD = 0.6
#: Pairwise kappa at or above this means the pair is redundant.
REDUNDANT_KAPPA = 0.85

KEEP = "KEEP"
REVIEW = "REVIEW"
DROP = "DROP / DOWN-WEIGHT"


@dataclass(frozen=True)
class RaterTriage:
    rater: str
    labeled: int
    distribution: Mapping[str, int]
    abstention: float
    """Share of this rater's labels that are NEEDS_INVESTIGATION."""
    agreement_with_truth: float | None
    """Exact-match share, or None when no truth basis covers this rater."""
    mean_redundancy: float
    max_redundancy: float
    max_redundancy_with: str | None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def recommendation(self) -> str:
        if not self.flags:
            return KEEP
        return DROP if len(self.flags) >= 2 else REVIEW


def triage(
    raters: Mapping[str, Labels],
    truth: Labels | None = None,
    labels: tuple[str, ...] = LABELS,
    redundant_kappa: float = REDUNDANT_KAPPA,
) -> dict[str, RaterTriage]:
    """Flag and rank every rater in the panel."""
    names = sorted(raters)
    allowed = set(labels)

    pairs = pairwise_kappa(raters, labels)
    redundancy: dict[str, list[tuple[str, float]]] = {n: [] for n in names}
    for (a, b), res in pairs.items():
        redundancy[a].append((b, res.kappa))
        redundancy[b].append((a, res.kappa))

    # A rater is redundant if it appears in any pair at or above the threshold.
    redundant_with: dict[str, str] = {}
    for (a, b), res in pairs.items():
        if res.kappa >= redundant_kappa:
            redundant_with.setdefault(a, b)
            redundant_with.setdefault(b, a)

    out: dict[str, RaterTriage] = {}
    for name in names:
        rater_labels = {k: v for k, v in raters[name].items() if v in allowed}
        labeled = len(rater_labels)
        dist = Counter({lbl: 0 for lbl in labels})
        dist.update(rater_labels.values())

        abstention = (dist["NEEDS_INVESTIGATION"] / labeled) if labeled else 0.0

        agree: float | None = None
        if truth:
            agreed, evaluated = accuracy(rater_labels, truth)
            agree = (agreed / evaluated) if evaluated else None

        others = redundancy[name]
        kappas = [k for _, k in others]
        mean_red = sum(kappas) / len(kappas) if kappas else 0.0
        max_red, max_with = (0.0, None)
        if others:
            max_with, max_red = max(others, key=lambda x: x[1])

        flags: list[str] = []
        max_count = max((dist[lbl] for lbl in labels), default=0)
        max_label = next((lbl for lbl in labels if dist[lbl] == max_count), None)
        max_share = (max_count / labeled) if labeled else 0.0
        if max_share > SKEW_THRESHOLD:
            flags.append(
                f"SKEWED ({max_share * 100:.0f}% {max_label}): low discriminating power; "
                "likely rubber-stamping the description"
            )
        if abstention > ABSTENTION_THRESHOLD:
            flags.append(
                f"ABSTAINS ({abstention * 100:.0f}% NI): its NI votes carry no TP/FP signal; "
                "treat NI as abstention, or drop"
            )
        if agree is not None and agree < ACCURACY_THRESHOLD:
            basis = "adjudicated truth" if truth else "majority"
            flags.append(
                f"LOW ACCURACY ({agree * 100:.0f}% vs {basis}): diverges from the truth basis; "
                "down-weight"
            )
        if name in redundant_with:
            flags.append(
                f"REDUNDANT with {redundant_with[name]} (pairwise kappa >= {redundant_kappa}): "
                "near-identical; one of the pair is enough"
            )

        out[name] = RaterTriage(
            rater=name,
            labeled=labeled,
            distribution=dict(dist),
            abstention=abstention,
            agreement_with_truth=agree,
            mean_redundancy=mean_red,
            max_redundancy=max_red,
            max_redundancy_with=max_with,
            flags=tuple(flags),
        )
    return out


LENIENT = "LENIENT (over-calls TP)"
STRICT = "STRICT (FP/NI)"
BALANCED = "balanced"
LEAN_THRESHOLD = 0.7


def split_leaning(
    raters: Mapping[str, Labels],
    split_ids: tuple[str, ...],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, tuple[dict[str, int], str]]:
    """How each rater voted on the contested items: the TP-threshold signal.

    On the items the panel could not agree on, a rater that mostly says TP is
    running a lenient threshold; one that mostly says FP or NI is running a
    strict one. That is a more useful characterisation than accuracy, because
    on genuinely contested items there may be no single right answer.
    """
    allowed = set(labels)
    contested = set(split_ids)
    out: dict[str, tuple[dict[str, int], str]] = {}

    for name in sorted(raters):
        counts = dict.fromkeys(labels, 0)
        total = 0
        for item, lbl in raters[name].items():
            if item in contested and lbl in allowed:
                counts[lbl] += 1
                total += 1

        leaning = BALANCED
        if total > 0:
            tp_share = counts.get("TP", 0) / total
            strict_share = (counts.get("FP", 0) + counts.get("NEEDS_INVESTIGATION", 0)) / total
            if tp_share >= LEAN_THRESHOLD:
                leaning = LENIENT
            elif strict_share >= LEAN_THRESHOLD:
                leaning = STRICT
        out[name] = (counts, leaning)
    return out
