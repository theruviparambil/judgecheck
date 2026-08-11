# judgecheck

[![CI](https://github.com/theruviparambil/judgecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/theruviparambil/judgecheck/actions/workflows/ci.yml)

Validate an LLM-as-judge panel the way you would validate a panel of human
graders: with chance-corrected inter-rater agreement, not accuracy.

Typed Python, no runtime dependencies, a CLI, and a test suite whose core
reproduces, number for number, the output of veriva-eval: my earlier TypeScript
harness for the same job.

```bash
git clone https://github.com/theruviparambil/judgecheck
cd judgecheck
pip install -e .

judgecheck report tests/data/panel-real
```

## The number this repo exists to prove

The seven-model panel in `tests/data/panel-real/` was scored by
[veriva-eval](https://github.com/theruviparambil/veriva-eval), my earlier
TypeScript harness for this job. Its output is committed here as a fixture,
judgecheck recomputes it in Python, and the test suite asserts every value
matches:

| reproduced | count |
| --- | --- |
| Pairwise Cohen's κ | 21 (every model pair) |
| Per-rater recall and precision | 14 |
| Mean pairwise κ | 7 |
| Consensus classification per finding | 23 |
| Fleiss' κ and Krippendorff's α | 2 |

```
Fleiss' kappa        +0.135  (poor)   n=23, raters=7
Krippendorff's alpha +0.141  (poor)
```

Both to full float precision: `0.13541263908579276` and `0.14078274691755766`.

The fixture is committed as-is and judgecheck never regenerates it. That is the
whole point. If this package could rewrite its own expected values, matching
them would prove nothing.

Be clear about what this does and does not establish. Both implementations
share an author, so this is not third-party validation, and the Python is a
deliberate port of the TypeScript rather than an independent derivation. What
the reproduction pins down is narrower: that the port is faithful across a
language boundary, measured against expected values that predate it. The
fixture has exactly one commit in veriva-eval's history, dated 2026-07-01,
before judgecheck existed. Narrow, and still enough to catch a real bug, which
it did.

## Why κ and not accuracy

Seven frontier models labeled the same 23 code-review findings. Adjudicated
truth is 15 TP, 7 NEEDS_INVESTIGATION, 1 FP, so a rater that blindly answers TP
to everything scores 65% accuracy while contributing no information at all.

Look at what that does to `glm`, alongside two of the other six raters:

| rater | accuracy vs truth | recall | mean pairwise κ | label mix |
| --- | --- | --- | --- | --- |
| gemini | 87% | 15/15 | 0.285 | 18 TP, 1 FP, 4 NI |
| glm | 74% | 15/15 | 0.146 | 21 TP, 2 NI |
| grok | 43% | 2/15 | 0.134 | 2 TP, 1 FP, 20 NI |

`glm` catches every true positive and beats the always-TP baseline on accuracy.
It also says TP to 91% of everything it sees. Accuracy cannot tell those two
facts apart; κ can, because it subtracts the agreement the panel would reach by
chance. judgecheck flags `glm` as SKEWED and recommends REVIEW.

Across all 21 pairs, κ ranges from -0.087 to 0.517. This panel is not measuring
one thing, whatever its accuracy column says.

Read that table as one panel, not a model ranking. It is 23 findings from a
single code-review task with one rubric, which is enough to demonstrate the
method and nowhere near enough to rank frontier models against each other. A
different rubric or corpus would reorder it. What generalizes is the gap
between the accuracy column and the κ column, not the row order.

A note on the `interpretation` labels. They follow the Landis and Koch (1977)
bands with two changes, kept deliberately so they match the reference
implementation: their `slight` band (0.00 to 0.20) is folded into `poor`, and
their `almost perfect` is named `near perfect`. Those bands are a reading aid
with no inferential standing, and at n=23 the point estimates carry real
uncertainty. judgecheck reports no confidence intervals, so read every number
here as describing this panel, not as an estimate of a population value.

## What `report` gives you

Four sections, from one pass over the panel:

**Panel agreement.** Fleiss' κ and Krippendorff's α, both defined for more than
two raters. Averaging pairwise Cohen's κ is not a defined panel coefficient, so
judgecheck reports that mean separately and labels it what it is. Both are
computed in a form that tolerates a rater skipping an item; on a complete panel
like this one that reduces to the standard definitions.

**Validity.** Recall and precision against adjudicated truth, binary for one
positive label rather than macro-averaged. `NEEDS_INVESTIGATION` is an
abstention, and folding an abstention into a macro average pays a rater for
refusing to decide. A truth-positive the rater skipped and one it labeled
negative both count as misses.

**Consensus.** Every finding classified UNANIMOUS, MAJORITY, or SPLIT. The SPLIT
items are the ones worth a human's time. This panel has 3.

**Triage.** Per-rater flags and a recommendation. A rater collects flags for
being skewed (>80% one label), abstaining (>40% NI), diverging from truth (<60%
exact match), or being redundant with another rater (pairwise κ >= 0.85). Zero
flags is KEEP, one is REVIEW, two or more is DROP / DOWN-WEIGHT.

Note the direction of the redundancy signal. High agreement between two raters
means one of them is surplus, not that either is good.

`--json` prints the same numbers unrounded, for diffing against another
implementation.

## Gating CI

`--fail-under` turns the report into a check:

```bash
judgecheck report path/to/panel --fail-under 0.6
```

It exits 1 unless Fleiss' κ **and** Krippendorff's α are both at or above the
threshold. Checking one alone lets a panel through on whichever coefficient
happens to be kinder, and the two disagreeing is the borderline case a gate
exists to catch. On the panel in this repo, `--fail-under 0.14` fails on Fleiss
(0.1354) while Krippendorff (0.1408) clears it.

```
GATE ─────────────────────────────────────────────────────────────────────
  FAIL  Fleiss' kappa 0.135 < 0.600
  FAIL  Krippendorff's alpha 0.141 < 0.600
```

| exit | meaning |
| --- | --- |
| 0 | the report ran, and any requested gate passed |
| 1 | `--fail-under` was given and the panel fell short |
| 2 | usage or input error |

The two failure codes stay separate so CI can distinguish "this panel does not
agree enough" from "you invoked the tool wrong". A missing panel directory is
always exit 2, even with `--fail-under` set: you cannot gate what you cannot
read.

There is no default threshold, and omitting the flag never fails the build.
How much agreement is enough depends on what the panel decides and what a wrong
call costs. `--json` still prints the full report alongside a `gate` block, so
a failed run is diagnosable rather than just red.

## Library

```python
from judgecheck import load_panel, build_report, fleiss_kappa, cohens_kappa

panel = load_panel("tests/data/panel-real")

fleiss_kappa(panel.raters).value  # 0.13541263908579276
cohens_kappa(panel.raters["claude"], panel.raters["gpt"]).kappa

report = build_report(panel)
report.consensus_counts  # {'UNANIMOUS': 1, 'MAJORITY': 19, 'SPLIT': 3}
[t.rater for t in report.triage.values() if t.recommendation != "KEEP"]
```

A panel directory holds one `<rater>.jsonl` per rater, each line
`{"findingId": ..., "label": ...}`, plus an optional `truth.json`. Without
truth you still get agreement, consensus, and triage; only validity needs it.

Results are frozen dataclasses. Inputs are plain mappings, so you can pass
ordinary dicts without adopting any types from this package.

## How it is verified

```
pytest          150 tests on 3.10, 3.11, 3.12, 3.13
mypy --strict   clean across src, tests, and scripts
ruff            check and format clean
mutation sweep  36/36 mutants killed
cross-check     Fleiss, Krippendorff, 21 Cohen pairs vs third-party libraries

Four of the 150 are the cross-validation tests and need the `crossval` extra;
they skip without it, and CI installs it in a dedicated job.
```

The reproduction tests prove judgecheck agrees with the reference
implementation on real data. They cannot prove the tests would catch a defect,
so `scripts/mutation_sweep.py` breaks the source on purpose, one edit at a
time, and checks the suite goes red for each one:

```bash
python scripts/mutation_sweep.py
```

It earned its place. Run against a suite that was already 82 tests and fully
green, it exposed four real defects:

- **Fleiss' κ was nondeterministic.** It iterated a `set[str]`. Python
  randomizes string hashing per process and float addition is not associative,
  so the published value moved between `0.13541263908579276` and `...293`
  depending on `PYTHONHASHSEED`. Meaningless numerically, disqualifying for a
  package that claims exact reproduction. `tests/test_determinism.py` now
  spawns subprocesses across five hash seeds and asserts bit-identical output.
- **The panel statistics rejected the obvious call.** `fleiss_kappa` took a
  sequence of raters while everything else took a name-to-labels mapping, so
  `fleiss_kappa(panel.raters)` iterated the dict's keys and died inside the
  counting loop with `'str' object has no attribute 'get'`, an error naming
  neither the function called nor the argument passed to it.
- **Every threshold comparison was unprotected.** No rater in the real panel
  sits at exactly 80% skew or 40% abstention, and with seven voters
  `> voters/2` and `>= voters/2` are the same predicate, so flipping any of
  those operators changed nothing the tests could see.
- **Every malformed-input branch in the loader was unprotected**, because the
  real panel is complete and well-formed.

That is the honest limit of reproduction-as-a-test-suite: it proves you match
the reference on the data you have, not that you handle the data you do not.

## Independently cross-checked

Reproducing veriva-eval proves the port is faithful. It cannot catch a formula
that both implementations get wrong in the same way, because they share an
author. So the coefficients are also checked against libraries written by other
people:

| coefficient | checked against | difference |
| --- | --- | --- |
| Fleiss' κ | `statsmodels` | 0.0, bit-identical |
| Cohen's κ, all 21 pairs | `statsmodels` | 0.0, bit-identical |
| Krippendorff's α | `krippendorff` | 4.4e-16, one ulp |

The alpha difference is float non-associativity: the two implementations
accumulate the coincidence matrix in a different order. It is not disagreement.

`tests/test_crossvalidation.py` runs these on every CI build. They need extras
the base install leaves out:

```bash
pip install -e ".[crossval]"
pytest tests/test_crossvalidation.py
```

## Scope

v1 covers the agreement and reliability statistics the reference implementation
computes, plus the `--fail-under` gate. It is not a port of that project's live
judge or model benchmark. Also not included: weighted or ordinal κ, bootstrap confidence intervals, or
any default threshold. A tool that ships a built-in idea of "enough agreement"
makes that call on your behalf and hides it in a default.

## Data

`tests/data/panel-real/` is the public, git-tracked panel from veriva-eval: 23
candidate findings raised against merged pull requests in public open-source
repositories (Cal.com, Discourse), labeled independently by seven frontier
models, with truth set by a separate full-context adjudication pass. Internal
finding ids and rule tags were removed before publication.

These are model-generated *candidate* findings, not confirmed defects in those
projects. Adjudication marked 15 of 23 as true positives and the rest as false
positives or unresolved, which is the entire reason the panel is interesting.
Nothing here is a vulnerability report, and nothing here is newly disclosed:
this is the same panel veriva-eval already publishes.

The private corpus behind that project is not in this repository and is never
read by this package.

## Requirements

Python 3.10+, tested on 3.10 through 3.13. No runtime dependencies; a clean
install pulls in nothing but judgecheck itself. Development extras are pytest,
mypy, and ruff:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
