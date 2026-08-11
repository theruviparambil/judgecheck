"""Reading a rater panel off disk.

A panel directory looks like:

    <dir>/truth.json        {"verdicts": [{"findingId": ..., "label": ...}]}
    <dir>/<rater>.jsonl     one {"findingId": ..., "label": ...} per line

`truth.json` is optional: agreement statistics do not need it, only validity
does. Malformed `.jsonl` lines are skipped rather than fatal, matching the
reference implementation, because a panel is usually assembled from several
model runs and one bad line should not lose the other twenty-two.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Judgment, Panel

TRUTH_FILENAME = "truth.json"


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
            finding_id = obj.get("findingId")
            label = obj.get("label")
            if not finding_id or not label:
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
    return {
        v["findingId"]: v["label"]
        for v in verdicts
        if isinstance(v, dict) and v.get("findingId") and v.get("label")
    }


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
