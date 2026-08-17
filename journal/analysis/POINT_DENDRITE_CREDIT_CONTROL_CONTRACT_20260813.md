# Point--dendrite and BP--local-credit control contract

Frozen: 13 August 2026, before any outcome from the new control arms was
opened. The reference BP and LocalCA outcomes for seeds 10200--10209 already
existed and motivated the reviewer-requested controls; this is therefore a
prospective extension of a completed cohort, not an independently selected
replication of its reference arms.

## Fixed task and operating point

All arms reuse the confirmatory hierarchical gain--load task without
recalibration: signal delta 0.80, train/test hierarchical log-gain SD 0.25,
child conductance 16, identity reactivation, mechanism-neutral
initialization, 180-epoch maximum, the same early-stopping rule, and paired
seeds 10200--10209. D1 `[8]`, D2 `[2,3]` and D3 `[2,1,2]` retain the fixed
eight-branch 4/2/2 input inventory.

## New arms (140 fits)

1. **All-active grouped-point/star, standard BP (60):** D1--D3 in aligned and
   reversed-placement regimes. It retains the tree's modules, masks,
   parameters, active contacts and initialization, but computes each level
   independently and pools the retained child aggregators once at the soma.
2. **Unstructured point MLP, standard BP (20):** one MLP matched to active
   reference parameters and one matched to total trainable parameters, both in
   the aligned task. The D3 reference config determines the count; the three
   tree morphologies are resource identical in the original cohort.
3. **Soma-broadcast autograd (60):** D1--D3 in aligned and reversed-placement
   regimes. The forward tree and ordinary autograd eligibility derivatives are
   unchanged, but every non-somatic output derivative is replaced by its
   owning soma derivative. This isolates teaching-coordinate restriction from
   LocalCA's explicit eligibility rule.

The completed exact-BP, shared-soma LocalCA and exact-path LocalCA arms on the
same seeds are reused as frozen references. They are not rerun or pooled with
different seeds.

## Primary estimands

All contrasts are paired by seed.

1. Serial composition at D3: hierarchical BP minus star BP, aligned.
2. Composition-by-alignment interaction: estimand 1 minus the same contrast
   after reversed placement.
3. D1 equality: hierarchical BP minus star BP at D1; expected zero up to
   floating-point/training nondeterminism.
4. Restricted-autograd cost by depth: soma-broadcast BP minus exact BP at D1,
   D2 and D3.
5. Eligibility residual: shared-soma LocalCA minus soma-broadcast BP.
6. Path-transport recovery: exact-path LocalCA minus shared-soma LocalCA,
   interpreted beside exact BP rather than as superiority to BP.
7. Unstructured point baselines versus D3 hierarchical BP, reported for both
   active- and total-parameter matching without selecting the better baseline.

Secondary endpoints are test AUC, train--test gap, D2 composition, and all
aligned-minus-reversed interactions.

## Gates and decision rules

- Every new config must finish with finite outcomes and no fallback pathway.
- Hierarchy/star comparisons must match state-dictionary keys and values at
  initialization, trainable parameters, active contacts, candidate slots and
  fixed input inventory exactly.
- Forward logits must match exactly between exact and soma-broadcast autograd
  before training; D1 hierarchical/star logits must agree to numerical
  tolerance.
- A positive claim requires a paired bootstrap 95% interval excluding zero
  and at least 8/10 seed signs. Architecture specificity requires the
  aligned-minus-reversed interaction.
- No architecture will be selected or relabeled after outcomes. A star tie
  means the effect belongs to grouped divisive computation rather than serial
  dendritic composition. A point-MLP win is reported as such.

## Claim boundary

These experiments can identify whether the paper's largest effect requires
serial tree composition, grouped nonlinear state, unstructured parameter
capacity, exact path credit or only a neuron-level teaching coordinate. They
cannot prove that a biological dendrite implements the assumed error signal,
that the calibrated task is naturalistic, or that point networks cannot emulate
the computation when explicitly supplied equivalent state and routing.

## Post-outcome optimizer-matching amendment

After the 140 primary controls completed, a configuration audit showed that
the soma-broadcast autograd arm inherited the standard-BP optimizer groups
(global learning rate 0.003, unsplit parameters), whereas the frozen LocalCA
reference used its stability-selected optimizer groups (global learning rate
0.0008, split parameters, branch-weight rate 0.0007 and decoder rate 0.0015).
The primary BP-versus-broadcast comparison remains valid because those arms
are optimizer matched. The broadcast-versus-LocalCA contrast cannot, however,
be attributed specifically to eligibility while this optimizer difference is
present.

Before viewing any outcome from the amendment, we therefore froze 60 additional
diagnostic fits: aligned and reversed D1--D3 soma-broadcast autograd, ten paired
seeds each, using the LocalCA parameter groups and optimizer rates exactly.
This is transparently post-outcome and cannot upgrade or rescue the primary
claim. It asks only whether the observed BP--LocalCA separation survives
optimizer matching. Both the original and matched-optimizer contrasts will be
reported.
