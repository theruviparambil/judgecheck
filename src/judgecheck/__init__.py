"""judgecheck: validate LLM-as-judge reliability with agreement, not accuracy."""

from __future__ import annotations

from .agreement import (
    UNDEFINED,
    UNDEFINED_NO_VARIANCE,
    cohens_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
    mean_pairwise_kappa,
    pairwise_kappa,
)
from .consensus import MAJORITY, SPLIT, UNANIMOUS, UNSCORED, consensus, split_items
from .independence import (
    CAUTION_EFFICIENCY,
    CoincidentError,
    GroupAgreement,
    GroupLabels,
    PairCoincidence,
    PanelIndependence,
    coincident_errors,
    effective_raters,
    group_agreement,
    interpret_efficiency,
    panel_independence,
    rater_groups_from_panel,
)
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
from .uncertainty import (
    BOOTSTRAP_DRAWS,
    CONFIDENCE,
    Interval,
    PanelIntervals,
    bootstrap_intervals,
)
from .validity import accuracy, rater_validity, validity

__version__ = "0.2.0"

__all__ = [
    "BOOTSTRAP_DRAWS",
    "CAUTION_EFFICIENCY",
    "CONFIDENCE",
    "LABELS",
    "MAJORITY",
    "SPLIT",
    "UNANIMOUS",
    "UNDEFINED",
    "UNDEFINED_NO_VARIANCE",
    "UNSCORED",
    "CoincidentError",
    "ConsensusEntry",
    "Gate",
    "GroupAgreement",
    "GroupLabels",
    "Interval",
    "Judgment",
    "KappaResult",
    "Labels",
    "MultiRaterResult",
    "PairCoincidence",
    "Panel",
    "PanelIndependence",
    "PanelIntervals",
    "PanelReport",
    "RaterTriage",
    "RaterValidity",
    "accuracy",
    "bootstrap_intervals",
    "build_report",
    "check_gate",
    "cohens_kappa",
    "coincident_errors",
    "consensus",
    "effective_raters",
    "fleiss_kappa",
    "group_agreement",
    "interpret_efficiency",
    "interpret_kappa",
    "krippendorff_alpha",
    "load_judgments",
    "load_labels",
    "load_panel",
    "load_truth",
    "mean_pairwise_kappa",
    "pairwise_kappa",
    "panel_independence",
    "rater_groups_from_panel",
    "rater_validity",
    "render_text",
    "split_items",
    "split_leaning",
    "to_dict",
    "to_json",
    "triage",
    "validity",
]
