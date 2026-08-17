# Prospective shunting-dose and topology-alignment contract

Date frozen: 2026-08-02

These cohorts follow the primary depth-by-feedback study. They will not replace
or alter its configurations, and their canaries are software checks only.

## A. Inhibitory conductance dose

### Design

The frozen noise-resilience task is trained with a `[2,2,2]` tree, one
population of 128 neurons, 40 excitatory inputs per dendritic compartment, and
0, 5, 10, 20, or 40 inhibitory inputs. Shunting and raw additive neurons are
run separately with explicit architecture-specific reactivation calibration.
The local cohort crosses each dose with scalar, ancestry-shared, and exact
path-transported feedback. Matched BP controls use each dose once. The
confirmatory cohort uses seeds 42--51.

### Questions

1. Does inhibition have a monotonic or intermediate optimum for learning?
2. Is the dose-response different under shunting and subtractive inhibition?
3. Does any shunting-dose benefit remain under exact feedback and BP, or is it
   specific to restricted credit routing?
4. Do measured `G_I/G_tot`, input resistance, path-gain dispersion, gradient
   cosine, and one-step progress predict the learning response?

The primary contrast is the core-by-dose interaction within each feedback
condition. The scalar-minus-exact difference across dose tests whether
inhibition mainly conditions restricted feedback. The BP dose response is a
forward/optimization ceiling, not a local-learning baseline duplicated across
feedback labels.

Changing the inhibitory dose changes the complete trained operating point. It
does not by itself isolate the backward role of inhibition. That causal role is
assessed only by fixed-forward counterfactual diagnostics.

## B. Task-aligned fixed dendritic topology

### Design

All networks use compact fixed-index synapses, a `[2,2,2,2]` tree with 16
distal leaves, and 21 excitatory plus 11 inhibitory inputs per non-somatic
compartment. This gives 960 active input synapses per modeled neuron. The
topology axis changes only how the fixed indices are assigned:

- **spatial:** image coordinates are recursively assigned to branches by
  alternating height and width splits;
- **random:** the same compact indexed layer samples fixed input indices
  without the spatial map.

The two topologies are compared in shunting and additive neurons on ordinary
MNIST and on the frozen random-projection noise task. Local learning uses the
three-factor rule with scalar, ancestry-shared, or exact transported feedback;
matched BP controls are also run. The confirmatory cohort uses seeds 42--51.

### Prediction

Spatial topology should help only when coordinate proximity reflects task
structure. It may improve MNIST learning but should lose that advantage after
the frozen random projection destroys image locality. The pre-specified
topology-by-task interaction is therefore more informative than a uniform
topology win.

We will report exact trainable parameter counts and active-synapse counts for
every condition. A spatial advantage shared by BP and local learning is a
forward inductive-bias effect. An additional advantage under restricted local
feedback is evidence that task-aligned morphology also organizes credit.

## C. Ancestry-map necessity at matched feedback bandwidth

For both tasks and cores, depth-2 and depth-4 trees receive either the correct
ancestry-shared soma coordinate or a fixed derangement that sends each
coordinate to another neuron's subtree. The permutation preserves feedback
bandwidth, the per-example value distribution, the forward network, and the
training code path. The confirmatory cohort uses seeds 42--51.

The paired correct-minus-shuffled learning difference is the primary topology
contrast. It tests whether neuron identity must be routed along the correct
tree, not whether more feedback coordinates are better than one scalar. A
depth interaction tests whether wrong routing becomes more costly as a credit
signal must cover more compartments.

## Common analysis and stopping rules

- Seeds are paired and are the inferential unit.
- The full ten-seed cohort is retained irrespective of sign.
- Effect sizes and paired bootstrap intervals are primary; exact Wilcoxon
  signed-rank tests accompany pre-specified paired contrasts.
- Dendritic and somatic diagnostics are never pooled for cross-core claims.
- Canaries must complete without NaNs, shape fallbacks, missing stages, or
  missing artifacts before confirmatory submission.
- No task, topology, feedback, learning-rate, or seed selection is made from
  test-set performance.
