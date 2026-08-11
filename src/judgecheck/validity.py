"""How well each rater tracks adjudicated truth.

Two different questions, deliberately kept apart:

  `validity()`  binary, for one positive label. Of the truly-positive items,
                how many did this rater catch (recall)? Of the ones it called
                positive, how many were right (precision)?

  `accuracy()`  exact label match across all labels. A cohesion measure, and
                the input to the LOW ACCURACY triage flag.

They are not interchangeable. Recall and precision are deliberately *not*
macro-averaged across labels: `NEEDS_INVESTIGATION` is an abstention, so folding
it into a macro average would reward a rater for declining to decide.

A truth-positive item the rater skipped and one it labeled negative both count
as misses. Silence is not a free pass.
"""

from __future__ import annotations

from collections.abc import Mapping

from .types import Labels, RaterValidity

POSITIVE = "TP"


def rater_validity(labels: Labels, truth: Labels, positive_label: str = POSITIVE) -> RaterValidity:
    """Binary recall and precision for one rater against truth."""
    truth_positive_ids = [item for item, lbl in truth.items() if lbl == positive_label]

    caught = sum(1 for item in truth_positive_ids if labels.get(item) == positive_label)

    called = 0
    correct_calls = 0
    for item, lbl in labels.items():
        if lbl != positive_label:
            continue
        called += 1
        if truth.get(item) == positive_label:
            correct_calls += 1

    return RaterValidity(
        rater="",
        positive_label=positive_label,
        caught=caught,
        truth_positives=len(truth_positive_ids),
        called=called,
        correct_calls=correct_calls,
    )


def validity(
    raters: Mapping[str, Labels], truth: Labels, positive_label: str = POSITIVE
) -> dict[str, RaterValidity]:
    """Binary recall and precision for every rater."""
    out: dict[str, RaterValidity] = {}
    for name in sorted(raters):
        v = rater_validity(raters[name], truth, positive_label)
        out[name] = RaterValidity(
            rater=name,
            positive_label=v.positive_label,
            caught=v.caught,
            truth_positives=v.truth_positives,
            called=v.called,
            correct_calls=v.correct_calls,
        )
    return out


def accuracy(labels: Labels, truth: Labels) -> tuple[int, int]:
    """Exact-label agreement with the truth basis.

    Returns (agreed, evaluated). Only items with a truth label AND a rater
    label are evaluated, so a rater is neither rewarded nor punished for items
    outside the truth basis.
    """
    evaluated = 0
    agreed = 0
    for item, truth_label in truth.items():
        rater_label = labels.get(item)
        if rater_label is None:
            continue
        evaluated += 1
        if rater_label == truth_label:
            agreed += 1
    return agreed, evaluated
