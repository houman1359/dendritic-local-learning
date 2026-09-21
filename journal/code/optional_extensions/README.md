# Optional mechanism extensions

These experiments extend the nonlinear population model using its verified,
unchanged execution snapshot in `../population_replay/frozen/`. They are new
cohorts, distinct from the published rescue and checkpoint analyses.
The active studies are parent-state proxies and learned inhibitory routing.
The user deferred recurrent training before any recurrent fit started;
`scope_amendment.json` records that decision and the six retained contrasts.

From `article_analysis`, after restoring Source Data, run within an allocated
CPU job, with six single-thread workers by default:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true
mkdir -p /durable/new_run/development/logs
cp source_data/optional_extensions/scope_amendment.json /durable/new_run/
cp source_data/optional_extensions/range_amendment.json /durable/new_run/
python code/optional_extensions/campaign.py --root /durable/new_run --action freeze
python code/optional_extensions/scoped_campaign.py --root /durable/new_run --workers 6
```

The runner freezes executable hashes and all development/fresh seeds before
training. Each worker verifies the code and the original runtime. Development
uses only training and validation data. Rates are selected independently by
mean validation error across three seeds; confirmation uses twenty fresh,
paired seeds per study. Both selected and common-rate policies are reported.
When they select the same rate they reference one trajectory, not two fits.
Selected checkpoints minimize validation error within the fixed budget.

## Parent-state proxies

The forward model and original interaction target are unchanged. The terminal
signal substitutes noisy or quantized estimates of the parent's tanh slope.
Voltage noise has fixed SD 0.25, 0.5 or 1 in the model's voltage units and is
independent across examples, parents and updates. Coarse estimates use two or
four equal-width bands spanning the physical parent-voltage range [0, 1).
Each band delivers the tanh derivative at its midpoint. Neither proxy receives
an exact backward path or the omitted coupling factor.

The shuffled controls permute gains within each branch and context, preserving
their empirical distributions while breaking example-specific assignment.
A context-mean control preserves the mean but removes example dependence.
These finite-minibatch controls retain the usual small fixed-point/self-sample
contribution; they are not asserted to give exactly zero interaction gradient.

An initial development pilot used thresholds extending beyond the reachable
voltage range. It was stopped before any fresh evaluation. All 84 completed
pilot fits and their code are retained in `pilot_v1`; `range_amendment.json`
records the correction. All development fits were rerun with the corrected
bins before confirmation. The failed bin definition is not evidence that
coarse, genuinely varying proxies fail.

## Learned inhibitory routing

A trainable 4-by-4 softmax map converts a supplied context cue into relative
disinhibition, with fixed total inhibitory activity. The cue is removed from
terminal excitation in every routing arm, so a terminal cue shortcut cannot
substitute for routing. All arms retain the original stream-specific sensory
contacts and nonlinear parents.

The main learned map receives local parent error from the same somatic source
used by the synaptic rule. A separate reference supplies the router with the
exact task gradient while leaving the rest of the network on augmented local
credit. Supplied, uniform and wrong routing provide forward controls. This
tests learning a cue-to-route map, not discovering contexts or implementing a
distributed inhibitory circuit. Neuron-specific somatic errors remain supplied.

## Deferred recurrent-memory prototype (no training results)

Eight leaky recurrent states filter the feature sequences before the original
conductance population. Their decay and input gain are learned; the target
uses decay 0.8 and unit gain. The cue arrives at readout. Independent input
features are uniform on [-2,2], the initial states are zero, and training uses
eight time steps. Four and sixteen steps test length generalization.

Exact online parameter eligibilities for these diagonal, linear recurrences
are verified against backpropagation through time. A one-step control keeps
the same forward memory but omits accumulated eligibility; a no-memory
control removes state persistence. The spatial learning rule still determines
the error received by those eligibilities. Memory parameters are shared by
neurons reading the same feature, so this introduces shared upstream credit
as well as temporal processing. This small recurrent front end is not a test
of arbitrary recurrent networks, nonlinear recurrent dynamics or autonomous
generation of error signals. A one-step rule could still learn this task;
its failure is not assumed in advance.

## Analysis

`scripts/review_completion/optional_extension_analysis.py` verifies every
frozen job, result and checkpoint before exporting seed-level records. The
six retained primary paired contrasts are specified in the development
protocol and scope amendment, with exact two-sided sign-flip tests and one
Holm family across all six.
All other comparisons are descriptive. Confidence intervals resample paired
whole seeds. None of the development pilot, rate policies, time lengths or
multiple contexts is counted as an additional independent cohort.

## Reproducing the completed studies

The two studies completed 153 development fits and 460 fresh fits. They use
separate sets of twenty fresh seeds. The 680 endpoint rows include both rate
policies; coincident rates refer to the same checkpoint. Source Data provide
all validation histories, endpoints, route maps, six primary paired contrasts,
post hoc proxy-fidelity diagnostics and the original protocol amendments under
`source_data/optional_extensions/`. The temporal fields retained in the original
protocol are historical plans; `scope_amendment.json` and the fresh job list
explicitly exclude them.

After restoring Source Data to `article_analysis/source_data` as described in
the software-release README, render the published figure without training:

```sh
cd article_analysis
python scripts/review_completion/optional_extension_paper_figure.py
```

To rerun the two studies in a new writable output directory, use the freeze and
scoped-campaign commands above. The commands copy the historical scope and range amendments as inherited
design records; they do not rerun the superseded pilot. Never run the
unrestricted original manager for this revision. The current bins already implement the documented range correction.
The runner selects rates using its own development outcomes and freezes those
choices before fresh runs. Library changes can change trajectories.

The analysis command checks the complete job set, validation minima, original
result hashes and checkpoint hashes before producing any summary:

```sh
python scripts/review_completion/optional_extension_analysis.py \
  --root /durable/new_run --output /durable/new_analysis
python scripts/review_completion/optional_extension_paper_figure.py \
  --data /durable/new_analysis --output /durable/new_figures/optional_extensions.pdf
```

The retained research archive also includes `Optional_extension_runs.zip`: all
153 development and 460 fresh result/checkpoint pairs, all 84 superseded pilot
fits, the executed source snapshots and a SHA-256 manifest. It is a companion
raw-run archive, separate from the smaller numerical Source Data package.
Two independent full-budget replays (four-bin sensitivity and locally learned
augmented routing) matched every validation value, selected/endpoint state
tensor and reported test value exactly in the recorded CPU environment.
