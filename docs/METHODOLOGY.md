# Methodology

The derivations behind the independence numbers in the [README](../README.md). Split out so the README stays readable in one
sitting. Nothing here is needed to use the tool, and everything here is
needed to trust it.

Every claim in this file is pinned by `tests/test_readme_claims.py`, which
reads the README and this document together.

---

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
