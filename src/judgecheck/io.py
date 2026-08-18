"""Reading a rater panel off disk.

A panel directory looks like:

    <dir>/truth.json        {"verdicts": [{"findingId": ..., "label": ...}]}
    <dir>/<rater>.jsonl     one {"findingId": ..., "label": ...} per line

`truth.json` is optional: agreement statistics do not need it, only validity
does. Malformed `.jsonl` lines are skipped rather than fatal, matching the
reference implementation, because a panel is usually assembled from several
model runs and one bad line should not lose the other twenty-two.

Types are normalized here rather than trusted. `json.loads` returns `Any`, so
the annotations on `Judgment` are not enforced at runtime and `mypy --strict`
cannot see through the boundary. A row like `{"findingId": 1, "label": "TP"}`
therefore used to load happily and put an `int` into a dict typed `str`, which
only surfaced later as `TypeError: '<' not supported between instances of 'str'
and 'int'` from the `sorted()` calls in `fleiss_kappa`, `krippendorff_alpha`,
`consensus`, and `Panel.item_ids`.

That class of bug is invisible to the mutation sweep by construction: mutation
testing perturbs lines that exist, so it can never find validation that was
never written. Hence a validated boundary rather than more mutants.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Judgment, Panel

TRUTH_FILENAME = "truth.json"


def _as_id(value: object) -> str | None:
    """Normalize a finding id to `str`, or `None` if it is unusable.

    Scalars are coerced, because `{"findingId": 1}` unambiguously means item 1
    and rejecting it would be pedantic. Anything structured is rejected: a dict
    or list is a malformed row, not an id. `bool` is excluded explicitly since
    it subclasses `int` and `"True"` is not an id anyone meant to write.
    """
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _as_label(value: object) -> str | None:
    """Labels must be strings; a numeric or structured label is a malformed row."""
    if isinstance(value, str):
        return value.strip() or None
    return None


def load_labels(path: str | Path) -> dict[str, str]:
    """Read one rater's `.jsonl` into {finding_id: label}."""
    return {j.finding_id: j.label for j in load_judgments(path)}


def load_judgments(path: str | Path) -> tuple[Judgment, ...]:
    """Read one rater's `.jsonl`, preserving confidence and reasoning."""
    out: list[Judgment] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed line, keep the rest of the rater
            finding_id = _as_id(obj.get("findingId"))
            label = _as_label(obj.get("label"))
            if finding_id is None or label is None:
                continue
            out.append(
                Judgment(
                    finding_id=finding_id,
                    label=label,
                    confidence=obj.get("confidence"),
                    reasoning=obj.get("reasoning"),
                    model=obj.get("model"),
                    vendor=obj.get("vendor"),
                )
            )
    return tuple(out)


def load_truth(path: str | Path) -> dict[str, str]:
    """Read an adjudicated `truth.json` into {finding_id: label}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    out: dict[str, str] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        finding_id = _as_id(v.get("findingId"))
        label = _as_label(v.get("label"))
        if finding_id is None or label is None:
            continue
        out[finding_id] = label
    return out


def load_panel(directory: str | Path) -> Panel:
    """Load every `<rater>.jsonl` in a directory, plus `truth.json` if present.

    Raises:
        FileNotFoundError: if the directory has no `.jsonl` rater files.
    """
    d = Path(directory)
    rater_files = sorted(p for p in d.glob("*.jsonl"))
    if not rater_files:
        raise FileNotFoundError(f"no rater .jsonl files in {d}")

    raters: dict[str, dict[str, str]] = {}
    judgments: dict[str, tuple[Judgment, ...]] = {}
    for p in rater_files:
        name = p.stem
        js = load_judgments(p)
        judgments[name] = js
        raters[name] = {j.finding_id: j.label for j in js}

    truth_path = d / TRUTH_FILENAME
    truth = load_truth(truth_path) if truth_path.exists() else None

    return Panel(name=d.name, raters=raters, truth=truth, judgments=judgments)
