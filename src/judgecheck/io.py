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

Every field is normalized, not only the two the statistics read directly. The
optional ones are annotations until something starts grouping by them, and
`vendor` now is.
"""

from __future__ import annotations

import json
import warnings
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


def _as_text(value: object) -> str | None:
    """Normalize an optional free-text field, dropping anything non-string.

    Unlike `_as_id`, nothing is coerced. These fields are descriptive, so a
    number or an object in one is a sign the row means something other than
    what we think, and inventing `"1"` from it would hide that.

    This matters more than it looks. `model` and `vendor` are annotations
    today, but `vendor` is what `rater_groups_from_panel` groups raters by, so
    an unvalidated value here becomes a grouping key later and fails inside a
    statistic instead of at the line that produced it.
    """
    if isinstance(value, str):
        return value.strip() or None
    return None


def _as_confidence(value: object) -> int | None:
    """Confidence is declared `int | None`; keep it that way.

    `True` is rejected before the `int` branch for the same reason as in
    `_as_id`: `bool` subclasses `int`, and `confidence=True` would silently
    become `1`. A float is truncated only when it is integral, so `4.0` loads
    and `4.5` is dropped rather than quietly rounded.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
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
            if not isinstance(obj, dict):
                # Valid JSON, wrong shape. `[1,2]` and `null` parse fine and then
                # `obj.get` raises AttributeError, which escaped the CLI as a
                # traceback and exit 1 -- the code reserved for "the gate
                # failed", so CI read a crash as an agreement result.
                continue
            finding_id = _as_id(obj.get("findingId"))
            label = _as_label(obj.get("label"))
            if finding_id is None or label is None:
                continue
            out.append(
                Judgment(
                    finding_id=finding_id,
                    label=label,
                    confidence=_as_confidence(obj.get("confidence")),
                    reasoning=_as_text(obj.get("reasoning")),
                    model=_as_text(obj.get("model")),
                    vendor=_as_text(obj.get("vendor")),
                )
            )
    return tuple(out)


def load_truth(path: str | Path) -> dict[str, str]:
    """Read an adjudicated `truth.json` into {finding_id: label}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("verdicts", []) if isinstance(data, dict) else []
    verdicts = raw if isinstance(raw, list) else []
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
    truth = None
    if truth_path.exists():
        truth = load_truth(truth_path)
        if not truth:
            # An empty dict is falsy, so downstream this looked identical to
            # "there is no truth file" and the validity, independence and
            # coincident-error sections disappeared from the report with a
            # zero exit and no explanation. A file that exists and yielded
            # nothing is a data problem worth naming.
            warnings.warn(
                f"{truth_path} exists but yielded no usable verdicts; "
                "expected {'verdicts': [{'findingId': ..., 'label': ...}]}. "
                "Sections that need truth will be missing.",
                UserWarning,
                stacklevel=2,
            )

    return Panel(name=d.name, raters=raters, truth=truth, judgments=judgments)
