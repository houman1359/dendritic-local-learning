# Input-dependent parent-sensitivity rescue

This follow-up tests the missing parent derivative in the nonlinear interaction
condition of Figure 6, without editing any previously frozen study. All runs
use the existing production DendriNet snapshot and unchanged task. It is an
internally specified prospective test motivated by previously observed failure,
not an externally preregistered experiment.

Six rules: exact BP; unit broadcast; relative resistance h; h times the parent
tanh derivative; a derivative shuffled across examples within each parent and
context; and a full-chain diagnostic supplying all coupling ratios. The last
arm must match BP analytically; it is not independent evidence for a new rule.
Readout and soma updates remain exact in every arm. The new rules detach
inter-compartment transmission and construct delivered terminal signals from
parent diagnostics rather than traversing an autograd path through the tree.

The main bound is [-9,9] in log conductance. All six rules are tested with Adam
and SGD. A predefined [-12,12] Adam sensitivity retains exact BP, resistance,
augmented resistance and shuffled derivative. Three disjoint development seeds
compare three rates per condition over 2,048 updates (144 fits). Twenty fresh
paired seeds evaluate validation-selected rates over 4,096 updates (320 fits).
Development never evaluates test outcomes. Freeze requires every development
result and validates protocol/checkpoint hashes. Seed numbers are identifiers,
not execution dates. A human review separates development from fresh launch.

The two primary ordinary-test comparisons are resistance minus augmented and
shuffled minus augmented, using Adam and the original bound. Report all paired
outcomes, bootstrap intervals and Holm-corrected sign-flip tests. Diagnostics
isolate the target-interaction component using gradient differences at one
state and common normalization; they do not claim zero total learning under
broadcast. Wider bounds, SGD and distractor severity are secondary analyses.
First-bound step, selected bound fraction, clipped steps and validation history
are saved, along with selected and endpoint states. Diagnostic shuffles preserve
the training RNG and never determine tuning or checkpoint selection.

Runtime snapshots, results and scheduler logs belong on kempner_project_b.
The brief netscratch instruction on 20 September 2026 was withdrawn; copies
made there are redundant, not the canonical records or the active run location.
W&B is disabled. These research files remain local under the paper repository's
existing Overleaf-only tracking policy; they are also copied and hashed into the
durable execution snapshot before launch. No jobs or fresh cohorts are submitted
by protocol preparation or freezing alone.
