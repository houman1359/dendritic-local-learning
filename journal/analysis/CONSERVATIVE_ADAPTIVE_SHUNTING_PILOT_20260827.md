# Conservative adaptive-shunting pilot

Defined 27 August 2026 before inspecting the new pilot seeds.

## Question

The completed adaptive reliability experiment estimated branch-local signal
and noise but attenuated the resulting credit by the full plug-in gain.  At
maximum reliability heterogeneity, this recovered the branch ordering but
over-attenuated the update: the adaptive local shunt was worse than the noisy
unshunted rule.  This exploratory pilot asks whether conservative attenuation
toward the unshunted update can retain denoising while avoiding that loss.

## Rule

Let the existing paired-probe estimator produce the plug-in gain

```text
a_hat_b = clip(S_hat_b / (c * (S_hat_b + N_hat_b)), a_min, 1).
```

For a fixed attenuation dose `lambda`, use

```text
a_b(lambda) = 1 - lambda * (1 - a_hat_b).
```

Thus `lambda=0` is exactly the unshunted update and `lambda=1` is the
previous adaptive rule.  The pilot scans
`lambda in {0, 0.1, 0.25, 0.5, 0.75, 1}` without changing the task, paired
noise, estimator, step size, state clamp or training horizon.

## Seeds and decision rule

- Artifact-only canary seeds: 8298--8299.
- Exploratory selection seeds: 8300--8319.
- The independent simulation seed is the paired unit.
- The primary endpoint is final test loss at reliability heterogeneity 2.
- A positive dose is eligible for a later fresh-seed confirmation only if it
  improves on `lambda=0` by at least 0.001 mean loss and in at least 15/20
  paired seeds.
- Every dose and outcome is retained.  No pilot result is confirmatory and no
  result enters the manuscript without a separately frozen fresh-seed run.

## Interpretation boundary

This test still uses paired noisy teaching probes, a known step fraction and
an exact forward-state clamp.  A benefit would show that positive conductance
can realize a conservatively calibrated local denoising gain.  It would not
show a dendrite-exclusive operation, autonomous inhibitory plasticity or
benefit on a natural task.  Exact agreement with the supplied point gate
remains a required numerical control.

## Completed result

The canary and all 20 exploratory seeds passed the numerical gates.  No
positive attenuation dose improved mean final loss over the unshunted update.
The least adverse positive dose was `lambda=0.1`, with mean final loss
`0.361079` versus `0.360925` without shunting (paired difference
`+0.000154`, bootstrap 95% interval `[-0.000297, +0.000716]`); it improved
only 10 of 20 paired seeds.  The full plug-in dose was worse by `+0.003161`
mean loss and improved only 3 of 20 seeds.  Because the predeclared gate
required a loss reduction of at least `0.001` in at least 15 of 20 seeds, no
dose is eligible for fresh-seed confirmation.

This is an informative boundary result: in this paired-probe task, shrinking
the adaptive gain toward the unshunted update removes most of the penalty but
does not reveal a positive conductance benefit.  It therefore should not be
presented as positive manuscript evidence or motivate a larger autonomous
shunting study without a new mechanistic hypothesis.
