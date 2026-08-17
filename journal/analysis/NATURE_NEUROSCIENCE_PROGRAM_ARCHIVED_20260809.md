# Archived Nature Neuroscience expansion program

Archived on 9 August 2026 when Nature Communications became the sole active
editorial target. This file is retained only as a decision record; its format
limits and evidence gates do not govern the current manuscript.

## Objective

The expanded project will test one biological theory of local credit
assignment:

> Neuron-specific instructive signals provide a first level of credit. A
> dendritic tree expands those signals into branch-addressed routes,
> conductance controls route gain, and experience aligns co-learning synapses
> with the available routes.

The present manuscript establishes exact factorization, communication limits,
anatomical routing capacity, and modeled descendant-selective shunting. The
expansion must establish how these quantities relate to measured synaptic or
dendritic changes during animal learning. Additional model-generated route
fields alone do not satisfy this goal.

## Scope for this phase

Included:

1. longitudinal animal-learning data at spine, dendrite, neuron, or behavioral
   resolution;
2. expanded MICrONS structure--function analyses;
3. synapse-level inhibitory-placement analyses in reconstructed trees;
4. a formal credit-covariance alignment theory;
5. quantitative models of branch-specific learning and task interference;
6. extensive examples, schematics, and cell-level visualizations; and
7. reuse of the exact validated regular-tree panels and source data from the
   earlier credit-assignment work, with explicit disclosure.

Deferred:

1. new causal animal experiments;
2. a time-dependent or spiking adjoint derivation; and
3. claims of state-of-the-art machine-learning performance.

## Two-paper structure

### Paper A: focused theory

Working title: **Credit covariance and routing in dendritic trees**

Purpose: isolate the mathematical problem of representing a high-dimensional
synaptic credit field with a small number of neuron- or branch-indexed teaching
coordinates.

Core results:

1. exact eligibility--error factorization in directed conductance trees;
2. the route-subspace formulation and capture/descent bounds;
3. a covariance objective for task distributions;
4. conditions under which topology-matched routing exceeds scalar, low-rank,
   and depth-based feedback;
5. identifiability and communication-budget limits.

This paper must remain concise and should not claim that a biological animal
implements the rule. Its natural audience is theoretical neuroscience,
biological physics, or neural computation rather than a general physics
letters journal.

### Paper B: comprehensive computational neuroscience

Working title: **Dendritic routes organize local credit during learning**

Purpose: test the theory against regular networks, reconstructed anatomy,
measured visual responses, inhibitory synapse placement, and animal-learning
data.

The headline is conditional and biological: topology provides addresses,
conductance regulates address gain, and learning depends on task alignment.
The paper should explain both where the mechanism works and where anatomy alone
is insufficient.

## Evidence hierarchy

Every result must be labeled as one of:

1. exact mathematical identity;
2. numerical verification;
3. controlled synthetic necessity or sufficiency test;
4. model perturbation on reconstructed anatomy;
5. retrospective association in measured biological data;
6. prospective or held-out biological prediction; or
7. causal biological intervention.

This phase can reach level 6. Level 7 is deferred. Oracle gradients,
projection coefficients, or placements must never be described as biologically
available learning signals.

## Proposed main-results sequence

1. **Local credit equation.** Exact eligibility and transported error, with
   the earlier regular-tree panels retained as validated foundations.
2. **Credit has an address problem.** Scalar feedback loses neuron identity;
   neuron-indexed feedback and exact transport isolate the missing information.
3. **Task distributions define a credit covariance.** Introduce route-subspace
   alignment and derive expected capture under a wiring budget.
4. **Real dendritic trees provide sparse route dictionaries.** Original and
   expanded MICrONS cohorts, with many cell-level examples and matched
   structural controls.
5. **Visual function tests route alignment.** Expand beyond the seven-target
   pilot and use held-out stimulus-derived credit.
6. **Inhibition is positioned to regulate routes.** Test whether observed
   dendrite-targeting inhibitory synapses occupy predicted credit-control
   locations.
7. **The theory predicts animal learning.** Test spine or dendritic changes in
   longitudinal learning data and model branch-specific task interference.

## Submission gate

Nature Neuroscience is considered only if at least one independent,
multi-animal learning dataset supports a pre-specified, held-out prediction
that is specific to branch-routed credit and cannot be reduced to distance,
activity, or generic Hebbian coincidence. Expanded MICRONS and synthetic
learning results are supporting evidence, not substitutes for this gate.

If that gate is not met, the comprehensive paper remains appropriate for
Nature Communications or a strong computational-neuroscience journal. The
focused theory paper remains independently useful.

## Primary external datasets

1. Wright, Hedrick and Komiyama, *Science* (2025), DOI
   `10.1126/science.ads4706`: longitudinal single-spine activity and plasticity
   during motor learning. The Zenodo record `10.5281/zenodo.14549129` is
   restricted and requires an access request or collaboration.
2. MICrONS Consortium, *Nature* (2025), DOI
   `10.1038/s41586-025-08790-w`: reconstructed anatomy, connectivity, visual
   responses, coregistration, and digital-twin responses from one mouse.
3. Schneider-Mizell et al., *Nature* (2025), DOI
   `10.1038/s41586-024-07780-8`: synapse-level inhibitory specificity and
   target-compartment annotations in MICrONS.
4. Francioni et al., *Nature* (2026), DOI
   `10.1038/s41586-026-10190-7`: neuron-specific signed dendritic instructive
   signals during a BCI learning task.
5. Cichon and Gan, *Nature* (2015), DOI `10.1038/nature14251`:
   task-specific branch calcium spikes, spine plasticity, SST perturbation, and
   behavioral interference.
