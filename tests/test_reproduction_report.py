"""Reproduction suite, part two: validity, consensus, and triage.

Every expected value was produced by the `veriva-eval` TypeScript harness on
this same committed panel and captured on 2026-08-10:

  * validity          `npm run replay:real`
  * distributions,    `npm run reliability:real`
    accuracy, triage
  * consensus         `data/panel-real/panel-comparison.json` (committed fixture)

Nothing here is computed by judgecheck and then asserted against itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgecheck import load_panel
from judgecheck.consensus import consensus, split_items
from judgecheck.triage import split_leaning, triage
from judgecheck.types import Panel
from judgecheck.validity import validity

PANEL_DIR = Path(__file__).parent / "data" / "panel-real"


@pytest.fixture(scope="module")
def panel() -> Panel:
    return load_panel(PANEL_DIR)


# ── validity: `npm run replay:real` ────────────────────────────────────────
# rater -> (caught, truth_positives, correct_calls, called)
EXPECTED_VALIDITY = {
    "claude": (12, 15, 12, 13),
    "deepseek": (3, 15, 3, 3),
    "gemini": (15, 15, 15, 18),
    "glm": (15, 15, 15, 21),
    "gpt": (10, 15, 10, 10),
    "grok": (2, 15, 2, 2),
    "qwen": (9, 15, 9, 12),
}


def test_reproduces_recall_and_precision(panel: Panel) -> None:
    assert panel.truth is not None
    got = validity(dict(panel.raters), panel.truth)
    assert set(got) == set(EXPECTED_VALIDITY)

    for rater, (caught, truth_pos, correct, called) in EXPECTED_VALIDITY.items():
        v = got[rater]
        assert v.caught == caught, f"{rater} caught"
        assert v.truth_positives == truth_pos, f"{rater} truth positives"
        assert v.correct_calls == correct, f"{rater} correct calls"
        assert v.called == called, f"{rater} called"
        assert v.recall == pytest.approx(caught / truth_pos)
        assert v.precision == pytest.approx(correct / called)


def test_abstention_counts_as_a_miss_not_an_exemption(panel: Panel) -> None:
    """grok labeled only 2 findings TP, so it misses 13 of 15 truth positives."""
    assert panel.truth is not None
    v = validity(dict(panel.raters), panel.truth)["grok"]
    assert v.precision == 1.0, "never wrong when it does fire"
    assert v.recall == pytest.approx(2 / 15), "but silence is still a miss"


# ── consensus: committed panel-comparison.json ─────────────────────────────
def test_reproduces_all_23_consensus_classifications(panel: Panel) -> None:
    fixture = json.loads((PANEL_DIR / "panel-comparison.json").read_text(encoding="utf-8"))
    expected = {e["findingId"]: e for e in fixture["consensus"]["entries"]}
    assert len(expected) == 23

    got = {e.finding_id: e for e in consensus(dict(panel.raters))}
    assert set(got) == set(expected)

    for finding_id, exp in expected.items():
        entry = got[finding_id]
        assert entry.consensus == exp["consensus"], f"{finding_id} consensus"
        assert entry.consensus_label == exp["consensusLabel"], f"{finding_id} label"
        assert dict(entry.labels) == exp["labels"], f"{finding_id} votes"


def test_split_count_matches(panel: Panel) -> None:
    """`reliability:real` reports: 23 total, 3 SPLIT."""
    assert len(split_items(consensus(dict(panel.raters)))) == 3


# ── triage: `npm run reliability:real` ─────────────────────────────────────
# rater -> (TP, FP, NI, OOS, ni_pct, accuracy_pct, mean_red, max_red, max_with, rec)
EXPECTED_TRIAGE = {
    "claude": (13, 0, 10, 0, 43, 83, 0.26, 0.42, "gpt", "REVIEW"),
    "deepseek": (3, 8, 12, 0, 52, 48, 0.21, 0.34, "gpt", "DROP / DOWN-WEIGHT"),
    "gemini": (18, 1, 4, 0, 17, 87, 0.29, 0.52, "glm", "KEEP"),
    "glm": (21, 0, 2, 0, 9, 74, 0.15, 0.52, "gemini", "REVIEW"),
    "gpt": (10, 1, 12, 0, 52, 78, 0.29, 0.42, "claude", "REVIEW"),
    "grok": (2, 1, 20, 0, 87, 43, 0.13, 0.31, "gpt", "DROP / DOWN-WEIGHT"),
    "qwen": (12, 10, 1, 0, 4, 48, 0.08, 0.20, "deepseek", "REVIEW"),
}


def test_reproduces_triage(panel: Panel) -> None:
    got = triage(dict(panel.raters), panel.truth)
    assert set(got) == set(EXPECTED_TRIAGE)

    for rater, exp in EXPECTED_TRIAGE.items():
        tp, fp, ni, oos, ni_pct, acc_pct, mean_red, max_red, max_with, rec = exp
        t = got[rater]
        assert t.labeled == 23, f"{rater} labeled"
        assert t.distribution["TP"] == tp, f"{rater} TP"
        assert t.distribution["FP"] == fp, f"{rater} FP"
        assert t.distribution["NEEDS_INVESTIGATION"] == ni, f"{rater} NI"
        assert t.distribution["OUT_OF_SCOPE"] == oos, f"{rater} OOS"
        assert round(t.abstention * 100) == ni_pct, f"{rater} NI%"
        assert t.agreement_with_truth is not None
        assert round(t.agreement_with_truth * 100) == acc_pct, f"{rater} accuracy"
        assert round(t.mean_redundancy, 2) == mean_red, f"{rater} mean redundancy"
        assert round(t.max_redundancy, 2) == max_red, f"{rater} max redundancy"
        assert t.max_redundancy_with == max_with, f"{rater} max redundancy partner"
        assert t.recommendation == rec, f"{rater} recommendation: flags={t.flags}"


def test_no_rater_is_flagged_redundant_on_this_panel(panel: Panel) -> None:
    """Highest pairwise kappa here is 0.52, well under the 0.85 threshold."""
    for t in triage(dict(panel.raters), panel.truth).values():
        assert not any(f.startswith("REDUNDANT") for f in t.flags)


def test_glm_is_flagged_skewed_the_rubber_stamp_case(panel: Panel) -> None:
    """glm called 21 of 23 findings TP: high recall, low discriminating power."""
    t = triage(dict(panel.raters), panel.truth)["glm"]
    assert any(f.startswith("SKEWED") for f in t.flags)
    assert "91% TP" in t.flags[0]


# ── SPLIT leaning: `npm run reliability:real` ──────────────────────────────
EXPECTED_LEANING = {
    "claude": (2, 0, 1, 0, "balanced"),
    "deepseek": (0, 2, 1, 0, "STRICT (FP/NI)"),
    "gemini": (3, 0, 0, 0, "LENIENT (over-calls TP)"),
    "glm": (3, 0, 0, 0, "LENIENT (over-calls TP)"),
    "gpt": (1, 0, 2, 0, "balanced"),
    "grok": (0, 0, 3, 0, "STRICT (FP/NI)"),
    "qwen": (0, 3, 0, 0, "STRICT (FP/NI)"),
}


def test_reproduces_split_leaning(panel: Panel) -> None:
    splits = tuple(e.finding_id for e in split_items(consensus(dict(panel.raters))))
    got = split_leaning(dict(panel.raters), splits)

    for rater, (tp, fp, ni, oos, leaning) in EXPECTED_LEANING.items():
        counts, lean = got[rater]
        assert counts["TP"] == tp, f"{rater} TP on splits"
        assert counts["FP"] == fp, f"{rater} FP on splits"
        assert counts["NEEDS_INVESTIGATION"] == ni, f"{rater} NI on splits"
        assert counts["OUT_OF_SCOPE"] == oos, f"{rater} OOS on splits"
        assert lean == leaning, f"{rater} leaning"
