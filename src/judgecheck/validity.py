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

from collections.abc import Mapping, Sequence

from .types import Labels, RaterValidity

POSITIVE = "TP"


def rater_validity(
    labels: Labels,
    truth: Labels,
    positive_label: str = POSITIVE,
    rater: str = "",
) -> RaterValidity:
    """Binary recall and precision for one rater against truth.

    Both are scored over the adjudicated items only. That is not a detail:
    precision used to count every positive call the rater made, including calls
    on items truth never covered, and score them against `truth.get(item)`,
    which is `None` there. So an unadjudicated item silently became a false
    positive, recall and precision were computed over two different
    populations, and F1 combined them anyway.

    Partial adjudication is the normal case, since you adjudicate what you can
    afford to. On a panel with truth over 25 of 60 items this reported two
    judges at identical precision when one was twice as precise as the other,
    which is enough to make you down-weight your best judge.
    """
    truth_positive_ids = [item for item, lbl in truth.items() if lbl == positive_label]

    caught = sum(1 for item in truth_positive_ids if labels.get(item) == positive_label)

    called = 0
    correct_calls = 0
    for item, lbl in labels.items():
        if lbl != positive_label or item not in truth:
            continue
        called += 1
        if truth[item] == positive_label:
            correct_calls += 1

    return RaterValidity(
        rater=rater,
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
    return {
        name: rater_validity(raters[name], truth, positive_label, rater=name)
        for name in sorted(raters)
    }


def accuracy(
    labels: Labels, truth: Labels, allowed: Sequence[str] | None = None
) -> tuple[int, int]:
    """Exact-label agreement with the truth basis.

    Returns (agreed, evaluated). Only items with a truth label AND a rater
    label are evaluated, so a rater is neither rewarded nor punished for items
    outside the truth basis.

    `allowed` restricts scoring to a label set, which every other statistic in
    the package already does. Without it, running `--labels TP,FP` on a panel
    whose truth holds NEEDS_INVESTIGATION verdicts scored those 7 items as
    failures here while kappa dropped them, so triage flagged LOW ACCURACY off
    a denominator no other number in the report used.
    """
    label_set = frozenset(allowed) if allowed is not None else None
    evaluated = 0
    agreed = 0
    for item, truth_label in truth.items():
        if label_set is not None and truth_label not in label_set:
            continue
        rater_label = labels.get(item)
        if rater_label is None:
            continue
        if label_set is not None and rater_label not in label_set:
            continue
        evaluated += 1
        if rater_label == truth_label:
            agreed += 1
    return agreed, evaluated
