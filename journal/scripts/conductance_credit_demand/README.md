# Conductance-tree credit-demand experiments

This directory contains two separate, fully retained studies. The first has 16 trainable conductances and monotonic terminal responses. It produces a small precision advantage for resolved credit while fixed profiles remain accurate. The second has 24 trainable conductances: each terminal has two excitatory and two inhibitory feature channels, permitting aligned or opposing tuning across subtrees. It produces a large task-dependent learning difference and an explicit three-pattern oracle rescue.

Both use positive E/I conductances and the directed steady-state branch balance of DendriNet with identity reactivation. The expression E/(1+E+I) is a **voltage**, where E and I are effective open conductances. The opening comment in the frozen opponent model calls this expression a current; this is a documentation error in that frozen comment, not the implemented quantity. The paper and methods use the correct voltage definition.

The root `model.py`, `run.py` and `test_model.py` implement the first study and remain byte-identical to its pre-outcome freeze. `opponent_model.py`, `run_opponent.py` and `test_opponent.py` implement the separate second-stage development and fresh cohort. Changing an implementation or frozen protocol causes the runner to fail rather than silently overwrite provenance. The worker scripts are optional Slurm launchers; their working directory is resolved relative to the script.

Completed evidence is under `source_data/conductance_credit_demand`, with the second study in its `opponent` subdirectory. Both `RESULTS.md` files summarize every selected-rate outcome. `opponent/METHODS.md` and the root `conductance_demand_methods.tex` define the generators, all variables, credit adapters, sample sizes, parameter distributions, checkpoints and statistical comparisons. `opponent/DESIGN.md` records why the second study was introduced after the first. No development regime or fresh seed is omitted.

To validate the mathematics and rebuild tables from the restored Source Data, run from the journal directory:

```bash
python -m pytest -q scripts/conductance_credit_demand/test_model.py scripts/conductance_credit_demand/test_opponent.py
python scripts/conductance_credit_demand/report.py
python scripts/conductance_credit_demand/report.py --root source_data/conductance_credit_demand/opponent
python scripts/conductance_credit_demand/context_gradient_audit.py
python scripts/conductance_credit_demand/report_bounds.py
python scripts/conductance_credit_demand/build_opponent_figure.py
python scripts/conductance_credit_demand/build_supplementary.py
```

The stored freezes record Python 3.10.13, NumPy 2.2.6 and pandas 2.3.3. Independent PyTorch checks use 2.9.1. Fits are deterministic NumPy calculations; PyTorch is required only by the independent gradient tests. All fits ran on CPU, with one BLAS thread. The original and second model tests include an independent PyTorch computation, finite differences of every parameter and checks of local eligibility/path factorization. The original-bound replay checks every final parameter exactly.

To refit, use an isolated copy of the journal source and move that copy's completed `runs` and `extension` directories aside. Keep the protocol and selection records. The runners reject an existing completed audit record. Each seed can then be run with `run.py run --phase development|fresh --seed N` or `run_opponent.py run ...`; `extend.py` and `extend_opponent.py` continue the corresponding fresh outcomes. The optional worker files list the frozen seed ranges. Development selection is performed by `run.py select` or `run_opponent.py select`, before any fresh outcomes. Complete archived data suffice to reproduce the reported statistics without refitting.

The later bound sensitivity is explicitly posthoc. It restarts all 320 second-study fresh conditions from their original initial states with log bounds±20, and replays the 320 original-bound conditions to obtain first-contact times. These are 640 trajectories on the original 20 paired seed blocks, not 640 new independent replicates.

Binary context gives at most two distinct path profiles at fixed weights. The three-pattern control uses exact per-example projection coefficients and must be described as an oracle. All q factors are positive, so the mechanism is cancellation when conflicting example gradients are averaged, not an incorrect per-example gradient sign. Raw rank-one capture alone is not a sufficient learning criterion: the retained moderate-gating regime is an explicit counterexample.
