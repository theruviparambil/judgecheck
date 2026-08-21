"""Core types.

Results are frozen dataclasses rather than dicts so the API is typed and
introspectable. Inputs are plain `Mapping`s so callers can pass ordinary dicts
without adopting our types.

`frozen=True` here is shallow, as it is everywhere in Python. It stops you
rebinding a field; it does not stop you mutating what a field points at, so
`panel.raters["claude"]["new-item"] = "TP"` changes the panel and every
statistic computed from it afterwards. These types are also unhashable for the
same reason, because the generated `__hash__` reaches the `Mapping` inside.
Treat them as read-only by convention rather than by enforcement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: The canonical label set. Order does not affect any computation, but the
#: membership does: labels outside this set are excluded from every statistic.
#:
#: NEEDS_INVESTIGATION is an abstention, not a verdict: the rater is saying the
#: evidence does not settle it. OUT_OF_SCOPE is a valid observation that falls
#: outside what is being graded.
LABELS: tuple[str, ...] = ("TP", "FP", "NEEDS_INVESTIGATION", "OUT_OF_SCOPE")

#: A rater's labels, keyed by item id. Raters may abstain by omitting an item.
Labels = Mapping[str, str]

#: A whole panel, either as a name -> labels mapping (what `Panel.raters` holds,
#: and what the report functions take) or as a bare sequence of raters when the
#: names do not matter. The panel statistics accept both: requiring one shape
#: here and the other in `pairwise_kappa` is a trap, because a dict iterates its
#: keys and the failure surfaces as an unrelated AttributeError.
PanelLabels = Sequence[Labels] | Mapping[str, Labels]


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range resampling puts around it.

    Deliberately not `kw_only`, unlike every other result type here. Three
    fields in one natural order read better positionally, and there is nowhere
    to insert a fourth that would silently change meaning.
    """

    point: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, kw_only=True)
class Judgment:
    """One rater's verdict on one item, as read from a `.jsonl` panel file."""

    finding_id: str
    label: str
    confidence: int | None = None
    reasoning: str | None = None
    model: str | None = None
    vendor: str | None = None


@dataclass(frozen=True, kw_only=True)
class Panel:
    """A set of raters that labeled the same items, with optional truth."""

    name: str
    raters: Mapping[str, Labels]
    truth: Labels | None = None
    judgments: Mapping[str, tuple[Judgment, ...]] | None = None

    @property
    def rater_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.raters))

    @property
    def item_ids(self) -> tuple[str, ...]:
        ids: set[str] = set()
        for labels in self.raters.values():
            ids.update(labels)
        return tuple(sorted(ids))


@dataclass(frozen=True, kw_only=True)
class KappaResult:
    """Two-rater agreement."""

    n: int
    """Items both raters labeled with an in-set label."""
    agreement: float
    """Observed raw agreement, 0-1."""
    kappa: float
    """Cohen's kappa, -1..1."""
    interpretation: str


@dataclass(frozen=True, kw_only=True)
class MultiRaterResult:
    """Panel-level agreement across more than two raters."""

    n: int
    """Items with at least two ratings, which is what agreement is defined over."""
    raters: int
    value: float
    """Fleiss' kappa or Krippendorff's alpha, roughly -1..1."""
    interpretation: str


@dataclass(frozen=True, kw_only=True)
class RaterValidity:
    """How well one rater tracks adjudicated truth, for the positive label.

    Deliberately binary rather than multi-class. `NEEDS_INVESTIGATION` is an
    abstention, so folding it into a macro average would reward a rater for
    declining to decide.
    """

    rater: str
    positive_label: str
    caught: int
    """Truth-positive items this rater also called positive."""
    truth_positives: int
    called: int
    """Items this rater called positive."""
    correct_calls: int
    """Of those calls, how many were truly positive."""

    @property
    def recall(self) -> float | None:
        """`None` when the truth basis holds no positives to catch.

        Not 0.0. A rater that missed everything and a truth set with nothing to
        miss are different situations, and `--positive-label OUT_OF_SCOPE` on a
        panel with no OUT_OF_SCOPE verdicts printed `0/0 (0%)` for every judge,
        which reads as total failure rather than as an empty question. The same
        rule the agreement coefficients follow.
        """
        return self.caught / self.truth_positives if self.truth_positives else None

    @property
    def precision(self) -> float | None:
        """`None` when the rater never used the positive label."""
        return self.correct_calls / self.called if self.called else None

    @property
    def f1(self) -> float | None:
        """`None` when either component is undefined, or both are zero."""
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True, kw_only=True)
class ConsensusEntry:
    """Where the panel landed on a single item."""

    finding_id: str
    labels: Mapping[str, str]
    consensus: str
    """UNANIMOUS, MAJORITY, or SPLIT."""
    consensus_label: str | None
    """None when SPLIT."""
