# CIFAR-10 shunting feedback ladder

Frozen 22 September 2026 before generation, launch, or inspection of any outcome
from seeds 22000--22019. The five-seed exploratory pilot and the August BP
configuration screen have already been inspected. Neither is pooled with this
fresh cohort. This extension fills the missing shunting row of Figure 1E--F;
it is not conditional on finding an advantage for exact paths.

## Fixed design

Use the clean execution checkout at commit
`e516c7fec3169253ff8c14bc5f4ab1325469e4f5`, also used by the additive cohort.
The August validation-only screen selected `shunting_empirical_local_matched`;
its reference is config 39 in `journal_cifar10_bp_recipe_init_screen_20260828101020`.
The screen's corrected version-2 selection and all resolved settings are
checked against that saved reference, without selecting a new recipe.

The model is PyTorch DendriNet with shunting integration: one population of
20 cells, each with branch factors [3,3,3,3], 25 direct excitatory and inhibitory
input contacts per branch, and an MLP readout with widths 32,16,10. Legacy
input_mode=1 supplies direct signed inputs; it does not instantiate an
inhibitory-cell population or recurrent connections. CIFAR-10 is flattened to
3072 features, without normalization or augmentation, with a 90:10 training /
validation split and the official test set.

Use adaptive center-preserving initialization (target conductance 5, child
conductance 1) and empirical param_tanh calibration from three training batches
of 256 examples. Calibration must converge without reversion and match within
seed across all four conditions. Train with Adam, clipping 5, batch size 256,
400-epoch maximum, patience 50, and restoration of the validation-best state.
Parameter groups use learning rates 0.001 for input contacts and decoder and
0.0001 for coupling and reactivation parameters. Both Adam weight decay and
sparse active-weight maintenance are zero. All four conditions share these
settings. The additive cohort used its own validation-selected recipe
(200 epochs, patience 40, unsplit rate 0.001, maintenance 0.01); differences
between architecture means are not a controlled test of the forward operator.

The four paired conditions are strict scalar feedback (`scalar`), one
neuron-specific coordinate (`per_soma_shared`), exact paths (`path_transport`),
and matched BP (`standard`). The local arms use the same five-factor rule with
reactivation and decoder parameters trained by backpropagation; extra
morphology and HSIC modulators are disabled. Twenty seeds, 22000--22019, give
80 runs. Seeds are paired within this shunting cohort, not across architectures.
No runs may be dropped according to their accuracy.

## Endpoints and validation

The endpoint is official-test accuracy from each validation-selected checkpoint.
Report condition means and paired differences with 95% Student-t intervals.
The two directional superiority tests are neuron-specific minus scalar and
exact paths minus neuron-specific; use Holm correction as one family and exact
paired sign-flip tests as sensitivity checks. The exact-path superiority claim
also requires the positive control to pass. Exact paths versus BP uses TOST at
alpha 0.05 with a prespecified +/-1 percentage-point margin. Lack of a
significant difference is not equivalence. Report all effects, including null
or negative results. A substantial path-resolution benefit additionally requires
an average gain of at least one percentage point, as in the additive contract.

Before computing inferential results, require all 80 runs, unique expected
conditions and seeds, finite results and training trajectories, identical
within-seed model/data/common optimizer settings, calibration convergence,
validation-best checkpoint consistency, and frozen source/configuration hashes.
A run reaching the epoch cap with its best epoch in the final five or a negative
last-ten validation-loss slope is flagged as right-censored. An adequate image
assay requires matched BP mean accuracy at least 45%. Audit, adequacy and
convergence failures block confirmatory claims, not disclosure of results.

## Execution and provenance

The YAML `configs/cifar10_shunting_feedback_ladder_confirmatory.yaml` is the
launch source. Generate 80 configurations from the clean frozen checkout;
record the input/resolved YAMLs, manifest, launcher, source-file and generated
configuration hashes before submission. Freeze the analyzer before inspecting
outcomes. Generated artifacts stay under the durable
`cifar_shunting_revision_20260922` directory on kempner_project_b. Existing data
are read-only. Use H100-priority jobs, one GPU and eight CPUs per run, at most
eight concurrent jobs; disable external experiment tracking. Do not change or
cancel other sessions' jobs. Inspect infrastructure status without reading
partial accuracies. Analyze only the complete audited cohort.
