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
uncertainty. `--intervals` puts a bootstrap range around them; read every number
here as describing this panel, not as an estimate of a population value.

## How many judges do you actually have?

A panel vote is only worth what its judges independently contribute. Judges that
fail on the same items are one check reported several times, and a majority over
them sounds more confident than its evidence.

```
PANEL INDEPENDENCE ───────────────────────────────────────────────────────
  mean error correlation  -0.013  (sd 0.402)   over 21 pairs
  effective judges, Kish  7.00 of 7
  effective judges, eigen 2.19 of 7
  reported                2.19 of 7   (31% of nominal, not exchangeable (estimators disagree))
  independent-panel null  54% of nominal   95% [43%, 65%]
  p                       0.002   (more correlated than independent judges of this size)
```

**Read the percentage against that null, not against 100%.** `k / λ_max` is
biased downward when items are few relative to judges: the top eigenvalue of a
correlation matrix estimated from 23 observations is inflated by sampling noise
alone, so judges that are *independent by construction* score about 54% here,
never above 71%. An earlier version of this section reported "31% of nominal"
with no null beside it, which overstated the dependence by roughly a factor of
two and implied a ceiling the estimator cannot reach.

Two consequences worth stating. The 50% caution line
[Kohli](https://arxiv.org/abs/2605.29800) proposes was set for a panel where
the estimator is unbiased; here the null's own 95% band straddles it, so about a
quarter of *independent* panels of this shape would trip it. And the claim that
survives is the p value, not the percentage: 0.312 sits below every one of 500
null draws, so **these judges are measurably not independent** — but "worth 2.19
of 7 judges" is a description of one panel through a biased estimator, not a
measurement to two decimals.

Two estimators, and on this panel they disagree by a factor of three. That
disagreement is the finding.

**Kish's design effect**, `n_eff = k / (1 + (k-1) * φ̄)`, averages the pairwise
correlations between judges' binary **error** vectors. It is the standard
correction for correlated observations, and it assumes *exchangeability*: that
the pairwise correlations are roughly equal. Here they run from **-0.68 to
+0.74**, with a standard deviation thirty times the size of the mean. The
average is near zero because two blocks of judges cancel, not because the judges
are independent.

**The eigenvalue form**, `k / λ_max`, assumes nothing about the shape of the
correlation matrix. It is the robustness check
[Kohli (2026)](https://arxiv.org/abs/2605.29800) runs beside his own headline
figure, in the same sentence where he states the exchangeability assumption:

> "This formula assumes exchangeability (approximately equal pairwise
> correlations); **we validate this assumption against the eigenvalue method
> below.**"

On his panel the two agree within 1% (2.18 against 2.16). On this one they are
3.2× apart, so judgecheck reports the lower figure and says why. A gate should
not pass on an assumption the data violates.

It is not a quality score either. These same seven judges have a Fleiss' κ of
0.135 and six of the seven are flagged REVIEW or DROP. Judges can be
independently wrong. Read this next to the validity table, never instead of it.

### Errors, not agreement

The measurement has to be on errors. Agreement counts "both right" and "both
wrong" identically, and only the second one makes a panel weaker than its size.
Two judges who agree because they are both correct are not redundant in any way
that should worry you.

**Two earlier versions of this package got that wrong in different ways**, and
both are worth keeping in the README because the second one is subtler than the
first.

The first computed `n_eff` from mean pairwise Cohen's κ on *labels*:

```
seven judges that all match truth exactly    ->  14% of nominal, "highly correlated"
seven judges labeling uniformly at random    ->  89% of nominal, "near independent"
```

Backwards. Minimized by the ideal panel, maximized by the worthless one, and a
rubber-stamp rater carrying no information at all *raised* measured
independence, because a constant rater has κ ≈ 0 against everybody.
[Kohli](https://arxiv.org/abs/2605.29800) is explicit about the cause:

> "For binary error vectors, the phi coefficient reduces to the Pearson
> product-moment correlation, **which is the quantity the Kish formula
> requires**... Alternative association measures (**e.g., Cohen's kappa**)
> **conflate prevalence with dependence**."

The second version fixed the statistic and kept the pathology. Moving to φ over
error vectors made the reductio come out right, but `k` still counted every
rater handed in while the mean only averaged *comparable* pairs. So a perfect
oracle, whose every pair is undefined because it never errs, still added a full
effective judge:

```
baseline, k=7                      7.00 of 7
+ perfect oracle (zero errors)     8.00 of 8      <- 7 pairs incomparable, k grew anyway
+ always-TP rubber stamp           8.00 of 8
+ exact clone of gemini            7.07 of 8      <- a duplicate raised the count
```

Only raters contributing at least one comparable pair count toward `k` now, and
the reported figure is the conservative estimator. Every one of those additions
lowers measured independence, which is what adding a judge that carries no
information should do.

There is no label-only fallback. Without truth there is no error vector to
correlate, and the honest fallback is to report nothing: you can still measure
how much your judges *repeat* each other, which `mean_pairwise_kappa` and the
triage redundancy flag already do, but you cannot measure whether they *fail
together*, and only the second is what `n_eff` claims to describe.


### One pair is not independent, and it is measurable

```
COINCIDENT ERROR vs adjudicated truth ────────────────────────────────────
  worst pair    deepseek + grok   both wrong on 11/23, +4.2 above independent
  permutation p 0.002   (unlikely by chance, corrected for picking the worst of 21 pairs)
  error correlation  +0.740
```

All eleven of those items are findings adjudicated **TP**, real issues, and
neither judge confirmed any of them: `deepseek` called seven of them FP outright
and `grok` abstained. That is a shared failure mode rather than a shared mistake,
and it is the kind a panel vote cannot rescue you from, because on those items no
amount of additional votes from these two surfaces anything.

Two details about how that is reported, both of which changed after review:

**Ranked by excess, not by ratio.** The obvious statistic is the ratio of
observed joint errors to what independence predicts. But joint errors cannot
exceed the more accurate judge's error count, so that ratio is capped at
`n / max(a_wrong, b_wrong)` and ranking on it selects for pairs *containing an
accurate judge*. On this panel the four highest-ratio pairs all sit exactly at
their structural ceiling, so that ranking reports a property of the metric.

**With a p value that accounts for the selection.** With 21 pairs, some pair
looks alarming in most random panels. The reported p is the share of reshuffled
panels, each judge keeping its error count and having those errors redistributed
independently, whose *worst* pair was at least this bad.

### Why there is no same-family / cross-family split

The obvious feature is to group judges by developer and treat cross-family
panels as trustworthy. judgecheck deliberately ships no such gate.

The panel above has **zero** within-family pairs, so the statistic is undefined
on the only real data in this repository. And where it has been measured the
effect is small: Kohli scored 9 judges across 7 families and found its two
same-family pairs correlated at φ = 0.437 and φ = 0.435, against a cross-family
mean of φ = 0.389. That is a gap of **0.047**. The three most correlated pairs in
that study were all *cross*-family, and restricting the panel to one judge per
family made effective independence **worse**, not better: n_eff fell from 2.18 to
1.93. [Goel et al. (2025)](https://arxiv.org/abs/2502.04313) suggest why. Judges
favor models that are functionally similar to themselves after controlling for
capability, and functional similarity does not follow company lines.

The panel here is the same point in miniature: seven judges from seven different
developers, and the one strongly correlated pair in it, `deepseek` and `grok`, is
cross-family too.

None of which says two judges from one lab cannot be redundant. They often are.
The claim is narrower: the developer label does not predict redundancy well
enough to substitute for measuring it. So grouping is reported as a description
of the panel you brought, under whatever grouping you supply, and nothing gates
on it:

```bash
judgecheck report path/to/panel --groups groups.json
```

Group by developer, base architecture, serving provider, prompt template,
whatever you think the shared cause might be. The default reads the `vendor`
field the panel files already carry, which is a convenience and not a claim that
vendor is the right cut.

### What this cannot tell you

23 items. Every number in this section describes one panel, and the pairwise
ones rest on 23 comparisons each.

`--intervals` puts a range around the ones a gate can be set on:

```
PANEL AGREEMENT ──────────────────────────────────────────────────────────
  Fleiss' kappa        +0.135  [+0.003, +0.245]  (poor)   n=23, raters=7
  Krippendorff's alpha +0.141  [+0.009, +0.249]  (poor)
```

95% percentile bootstrap over 1000 resamples of the items. The effective judge
count gets one too: **7.00, interval [5.22, 7.00]**.

Those are wide, and they should be. The one conclusion this panel robustly
supports is that its agreement is poor and probably above zero. It does not
support ranking these seven models, and it does not support reading `n_eff` to
two decimals.

It is not free. Measured on this machine, synthetic panels:

| panel | `report` | with `--intervals` |
| --- | --- | --- |
| 7 raters x 23 items | 0.11s | 0.6s |
| 7 x 200 | 0.92s | 3.8s |
| 12 x 500 | 6.9s | 22s |
| 20 x 1000 | 34s | 116s |

Fine as a CI gate at the top of that table and not fine at the bottom. Cost
grows with the square of the judge count, because both the permutation test and
the bootstrap run over every pair. `coincident_errors(..., permutations=N)` and
`bootstrap_intervals(..., draws=N)` are the knobs; the defaults are what
`tests/test_determinism.py` pins, and lowering them stays reproducible while
widening the p value's resolution.

**A gate implies intervals**, because that is where it matters most:

```
GATE ─────────────────────────────────────────────────────────────────────
  PASS  the panel is worth 7.00 judges, at or above 6.00
  NOTE  6.00 falls inside the effective-judges interval [5.22, 7.00];
        this panel cannot resolve that threshold, so the verdict
        is closer to a coin flip than to a measurement.
```

The gate still passes or fails exactly as asked. The note is information, not a
veto. But a green build that turned on a number the data cannot resolve is worth
saying out loud, and `--min-effective 5` on this panel is a real check while
`--min-effective 6` is not.


## What `report` gives you

Eight sections, from one pass over the panel:

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

**Panel independence.** Mean pairwise error correlation and the effective judge
count it implies. Needs adjudicated truth; absent otherwise rather than guessed.
See above.

**Judge groups.** Within-group against between-group agreement, for a supplied
grouping or the `vendor` field. Reported, never gated.

**Coincident error.** The pair with the most joint errors above what independence
predicts, with a permutation p that corrects for having selected a worst pair.

**Intervals.** With `--intervals`, or automatically whenever a gate is set,
bootstrap ranges around the coefficients a threshold can be applied to.

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

`--min-effective` gates on the other question, and the two genuinely disagree on
the panel in this repo: it has poor agreement (κ 0.135) and good independence
(7.00 of 7). Setting only one of them would have told you the opposite thing.

```bash
judgecheck report path/to/panel --min-effective 5
```

Worth setting both. A panel can clear an agreement floor precisely *because* its
judges are redundant, and a panel of genuinely independent judges will score
lower on agreement than a panel of near-copies. Gating on agreement alone rewards
the wrong thing.

A panel whose independence cannot be measured **fails** this gate rather than
skipping it. That happens when there is no adjudicated truth, or when the judges
have too few items in common to compare:

```
GATE ─────────────────────────────────────────────────────────────────────
  FAIL  effective judges not measurable, so a floor of 3.00 cannot be met
        (needs adjudicated truth and overlapping items)
```

Treating "unmeasurable" as "passed" is how a gate silently stops gating, and an
earlier version of this code did exactly that: three judges who had rated no
items in common reported 100% independence and passed.

| exit | meaning |
| --- | --- |
| 0 | the report ran, and any requested gate passed |
| 1 | a gate was given and the panel fell short |
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
report.consensus_counts  # {'UNANIMOUS': 1, 'MAJORITY': 19, 'SPLIT': 3, 'UNSCORED': 0}
[t.rater for t in report.triage.values() if t.recommendation != "KEEP"]
```

A panel directory holds one `<rater>.jsonl` per rater, each line
`{"findingId": ..., "label": ...}`, plus an optional `truth.json`. Without
truth you still get agreement, consensus, and triage; only validity needs it.

Results are frozen dataclasses. Inputs are plain mappings, so you can pass
ordinary dicts without adopting any types from this package.

## How it is verified

```
pytest          394 tests on 3.10, 3.11, 3.12, 3.13
mypy --strict   clean across src, tests, and scripts
ruff            check and format clean
mutation sweep  113/113 mutants killed
cross-check     Fleiss, Krippendorff, 21 Cohen pairs vs third-party libraries

16 of the 394 are the cross-validation tests and need the `crossval` extra;
they skip without it, and CI installs it in a dedicated job.
```

Read "113/113" with one caveat, because it is softer than a mutation score usually
is. The mutants are a hand-written list of string substitutions rather than
AST-generated, so **I chose them**, and a mutant nobody thought to write cannot
survive. It is a checklist of the invariants I believe matter, executed
honestly; it is not the unbiased score `mutmut` or `cosmic-ray` would give you.
The list is in `scripts/mutation_sweep.py` and worth skimming for that reason.

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

A fifth defect arrived later, from an outside reviewer rather than from the
sweep, and it is the most interesting one. A row like
`{"findingId": 1, "label": "TP"}` loaded without complaint, because
`json.loads` returns `Any` and the `str` annotation on `Judgment` is not
enforced at runtime, so `mypy --strict` cannot see through that boundary. An
`int` then sat in a mapping typed `str` until `sorted()` raised `TypeError` from
four separate call sites. **Mutation testing was structurally incapable of
finding it**: the sweep perturbs lines that exist, and what was missing was
validation nobody had written. Types are now normalized at the loader, with the
boundary tested directly. The lesson generalizes past this package: a mutation
score says nothing about the code you forgot to write.

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
| Krippendorff's α, **incomplete panel** | `krippendorff` | 0.0, bit-identical |

The alpha difference is float non-associativity: the two implementations
accumulate the coincidence matrix in a different order. It is not disagreement.

**One branch has no third-party check, and it is worth naming.** Every rater in
`panel-real` labeled every item, so the first three rows above only validate the
complete-data path. The abstention-tolerant generalizations, where a rater skips
an item and each item is scored over however many raters actually rated it, are
where independent implementations legitimately diverge.

Krippendorff's α is built for missing data, so the fourth row covers that path
directly on a deliberately incomplete panel. **Fleiss' κ is not.** statsmodels
declines unequal raters-per-item with an `AssertionError` rather than computing a
generalization, so judgecheck's generalized Fleiss is checked only against the
TypeScript harness, which shares an author. A test pins statsmodels' refusal, so
if a future version grows that support this claim gets revisited rather than
quietly becoming understated.

`tests/test_crossvalidation.py` runs these on every CI build. They need extras
the base install leaves out:

```bash
pip install -e ".[crossval]"
pytest tests/test_crossvalidation.py
```

## Scope

v1 covers the agreement and reliability statistics the reference implementation
computes, plus the `--fail-under` gate. It is not a port of that project's live
judge or model benchmark. Also not included: weighted or ordinal κ, BCa intervals, or
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
