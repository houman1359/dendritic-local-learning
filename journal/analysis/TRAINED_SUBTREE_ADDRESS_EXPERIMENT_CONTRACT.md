# Frozen contract: trained within-neuron subtree addressing

Status: phase-1 primary contrast and full architecture-by-bandwidth matrix completed
Target claim: estimate the learning value of task-dependent within-neuron
addresses at matched forward and feedback resources.

## Phase-1 execution record (2026-08-10)

The frozen `phase1_confirmatory.json` executes the central K=2 contrast on a
two-sibling credit-reversal task. Canary seed 4099 passed all prespecified
artifact and nondegeneracy gates. Without changing the executable or
configuration, confirmatory seeds 4100--4109 completed eight conditions (80
runs). Correct routing exceeded fixed within-neuron derangement by 61.00
percentage points and one shared neuronal coordinate by 4.39 points; both
comparisons favored correct routing in 10/10 seeds. A context-gated point
implementation matched correct routing exactly. Source hashes and every
run-level outcome are archived in `source_data/trained_subtree_address/`.

The unchanged support test was followed by the complete frozen matrix: 2,700
fits across seeds 4200--4219, $K\in\{1,2,4,8\}$, five matched
representations, six route families, a degree/depth-matched rewired tree, and
exact/backpropagation references. Correct ancestry beats the best matched non-
anatomical route only at $K=4$ and beats rewiring at $K=2,4$; dendritic,
gated-point, flat and grouped-point implementations coincide when supplied
with identical routed fields. Source Data are archived in
`source_data/trained_subtree_address_full_factorial/`. Active conductance is
tested separately as a steady-state focal-sensitivity ensemble rather than as
part of this learning factorial. Larger task families remain future work.

## Scientific contrast

The primary hierarchy is

`scalar -> one coordinate per neuron -> K subtree coordinates per neuron -> exact compartment transport`.

The existing correct-versus-deranged experiment permutes coordinates among
neuronal trees. This new experiment permutes routes **within each neuron** while
leaving neuron identity intact.

## Task

Use a context-dependent credit-reversal task with two input streams assigned
to sibling subtrees. On each trial, a latent context determines which stream is
relevant and which is a distractor. The neuron-level output and the sign of its
scalar output error can be identical across paired contexts, while the desired
updates to the two subtrees differ in sign or magnitude.

Required safeguards:

- balance contexts and labels within every split;
- prevent a direct context-to-label shortcut;
- hold out context/input combinations and include context switches;
- verify with exact gradients that sibling subtrees require different updates;
- reject task configurations in which one neuron-level coordinate captures
  more than 90% of exact within-neuron gradient energy before training.

## Feedback families

At every tested `K` in `{1, 2, 4, 8}`:

1. robust global scalar;
2. neuron-indexed shared coordinate;
3. correct ancestry routes with `K` subtree coefficients per neuron;
4. within-neuron route derangement, fixed per seed;
5. depth-bin routes;
6. random sparse routes matched for support size and nonzeros;
7. unrestricted random rank-`K` feedback;
8. learned unrestricted rank-`K` capacity upper bound, clearly labeled
   nonlocal if its coefficients use exact gradients;
9. exact compartment transport;
10. matched backpropagation.

## Forward architectures

Compare the same dendritic tree with:

- a point neuron;
- a flat multi-input compartment;
- grouped point subunits with the same number of nonlinear subunits;
- a depth/degree-matched randomly rewired tree.

Match and record separately:

- trainable forward parameters;
- active input contacts and input coverage;
- number of forward nonlinearities;
- decoder and output dimension;
- feedback channel count;
- feedback nonzeros and total route-wire length proxy;
- optimizer, schedules, initialization, batches, epochs, and compute budget.

No condition may receive an additional context input, decoder, or optimizer
state unavailable to its matched controls.

## Primary endpoints

1. held-out accuracy or loss after context switches;
2. forgetting on the previously relevant context;
3. eligibility-weighted exact-gradient capture;
4. cosine with the exact compartment gradient;
5. one-step retained progress at a norm-matched update;
6. performance as a function of feedback bandwidth and route alignment.

The primary estimand is correct ancestry minus within-neuron derangement at
fixed `K`, followed by correct ancestry minus the best matched non-anatomical
control. Seed is the paired inferential unit. Report every condition and every
prespecified `K`, including failures.

## Sample plan and execution gates

1. Unit-test route ownership and within-neuron derangement on a two-neuron,
   two-subtree analytic example.
2. Run one canary seed per condition only to validate artifacts, memory, and
   nondegenerate task gradients.
3. Freeze the task generator, seeds, configurations, exclusion rules, and
   source hash before confirmatory execution.
4. Run at least ten paired confirmatory seeds per condition; increase only by a
   prospectively documented precision rule, never by observed significance.
5. Audit balance, fallback counts, exact parameter budgets, route nonzeros,
   checkpoint completeness, and source equivalence before analysis.

## Interpretation

- A correct-route advantage over within-neuron derangement isolates address
  support at matched coordinate count.
- An advantage over random rank-`K` supports a topology-specific benefit.
- An advantage shared by backpropagation is a forward inductive-bias effect.
- A benefit only when routes align with latent causes is the predicted
  conditional result, not a failure of generality.
- A null result rules out a large effect in the tested regime and strengthens
  the current conditional framing; it must not be hidden or replaced by a new
  task after outcome inspection.

## Required outputs

- one resolved configuration and mechanism ledger per run;
- seed-level outcomes and paired contrasts;
- exact feedback-channel and nonzero counts;
- route-overlap and off-route-leakage matrices;
- gradient audits at initialization and trained checkpoints;
- source hashes, environment lock, clean commit, and strict-fallback log;
- one command that regenerates the complete figure and Source Data.
