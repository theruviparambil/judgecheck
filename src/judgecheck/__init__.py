"""judgecheck: validate LLM-as-judge reliability with agreement, not accuracy."""

from __future__ import annotations

from .agreement import (
    cohens_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
    mean_pairwise_kappa,
    pairwise_kappa,
)
from .consensus import consensus, split_items
from .io import load_judgments, load_labels, load_panel, load_truth
from .report import Gate, PanelReport, build_report, check_gate, render_text, to_dict, to_json
from .triage import RaterTriage, split_leaning, triage
from .types import (
    LABELS,
    ConsensusEntry,
    Judgment,
    KappaResult,
    Labels,
    MultiRaterResult,
    Panel,
    RaterValidity,
)
from .validity import accuracy, rater_validity, validity

__version__ = "0.1.0"

__all__ = [
    "LABELS",
    "ConsensusEntry",
    "Gate",
    "Judgment",
    "KappaResult",
    "Labels",
    "MultiRaterResult",
    "Panel",
    "PanelReport",
    "RaterTriage",
    "RaterValidity",
    "accuracy",
    "build_report",
    "check_gate",
    "cohens_kappa",
    "consensus",
    "fleiss_kappa",
    "interpret_kappa",
    "krippendorff_alpha",
    "load_judgments",
    "load_labels",
    "load_panel",
    "load_truth",
    "mean_pairwise_kappa",
    "pairwise_kappa",
    "rater_validity",
    "render_text",
    "split_items",
    "split_leaning",
    "to_dict",
    "to_json",
    "triage",
    "validity",
]
