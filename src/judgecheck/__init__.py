"""judgecheck: validate LLM-as-judge reliability with agreement, not accuracy.

Everything a caller needs is exported here, including the result strings and
thresholds, because the submodules are not reliably reachable by name. Four
functions share a name with the module that defines them (`triage`, `validity`,
`consensus`, `accuracy`), so after this module runs, `judgecheck.triage` is the
*function*, not the module, and `import judgecheck.triage as t; t.KEEP` raises
AttributeError. That is standard Python and not worth a rename that would break
every caller; the fix is that you never need the module. Use
`from judgecheck import KEEP` and the rest.
"""

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
    EXCHANGEABILITY_TOLERANCE,
    MIN_PAIR_ITEMS,
    NULL_DRAWS,
    PERMUTATION_SEED,
    PERMUTATIONS,
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
from .triage import (
    ABSTENTION_THRESHOLD,
    ACCURACY_THRESHOLD,
    BALANCED,
    DROP,
    KEEP,
    LEAN_THRESHOLD,
    LENIENT,
    REDUNDANT_KAPPA,
    REVIEW,
    SKEW_THRESHOLD,
    STRICT,
    UNKNOWN,
    RaterTriage,
    split_leaning,
    triage,
)
from .types import (
    LABELS,
    ConsensusEntry,
    Interval,
    Judgment,
    KappaResult,
    Labels,
    MultiRaterResult,
    Panel,
    RaterValidity,
)
from .uncertainty import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    PanelIntervals,
    bootstrap_intervals,
)
from .validity import POSITIVE, accuracy, rater_validity, validity

__version__ = "0.2.0"

__all__ = [
    "ABSTENTION_THRESHOLD",
    "ACCURACY_THRESHOLD",
    "BALANCED",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CAUTION_EFFICIENCY",
    "CONFIDENCE",
    "DROP",
    "EXCHANGEABILITY_TOLERANCE",
    "KEEP",
    "LABELS",
    "LEAN_THRESHOLD",
    "LENIENT",
    "MAJORITY",
    "MIN_PAIR_ITEMS",
    "NULL_DRAWS",
    "PERMUTATIONS",
    "PERMUTATION_SEED",
    "POSITIVE",
    "REDUNDANT_KAPPA",
    "REVIEW",
    "SKEW_THRESHOLD",
    "SPLIT",
    "STRICT",
    "UNANIMOUS",
    "UNDEFINED",
    "UNDEFINED_NO_VARIANCE",
    "UNKNOWN",
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
