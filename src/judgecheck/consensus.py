"""Where a panel landed on each item.

Three outcomes:

    UNANIMOUS  every rater who voted picked the same label
    MAJORITY   one label held strictly more than half the votes
    SPLIT      no label held a majority; `consensus_label` is None

SPLIT items are the interesting ones. They are where the panel genuinely
disagreed, which is the signal a human adjudication step exists to resolve.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .types import LABELS, ConsensusEntry, Labels

UNANIMOUS = "UNANIMOUS"
MAJORITY = "MAJORITY"
SPLIT = "SPLIT"


def consensus(
    raters: Mapping[str, Labels], labels: tuple[str, ...] = LABELS
) -> tuple[ConsensusEntry, ...]:
    """Classify every item the panel touched.

    Only in-set labels count as votes. A rater that skipped an item is not a
    voter on it, so a single-voter item is trivially UNANIMOUS.
    """
    allowed = set(labels)
    item_ids: set[str] = set()
    for rater_labels in raters.values():
        item_ids.update(rater_labels)

    out: list[ConsensusEntry] = []
    for item in sorted(item_ids):
        votes: dict[str, str] = {}
        for name in sorted(raters):
            lbl = raters[name].get(item)
            if lbl is not None and lbl in allowed:
                votes[name] = lbl

        counts = Counter(votes.values())
        voters = len(votes)
        top_label, top_n = (None, 0)
        if counts:
            top_label, top_n = counts.most_common(1)[0]

        if voters > 0 and top_n == voters:
            kind, label = UNANIMOUS, top_label
        elif top_n > voters / 2:
            kind, label = MAJORITY, top_label
        else:
            kind, label = SPLIT, None

        out.append(
            ConsensusEntry(finding_id=item, labels=votes, consensus=kind, consensus_label=label)
        )
    return tuple(out)


def split_items(entries: tuple[ConsensusEntry, ...]) -> tuple[ConsensusEntry, ...]:
    """Just the contested ones."""
    return tuple(e for e in entries if e.consensus == SPLIT)
