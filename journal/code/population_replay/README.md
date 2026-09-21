# Population selection and parent-sensitivity rescue

`frozen/` preserves the executed DendriNet working-tree snapshot and study
modules. This runtime includes pre-existing production edits; the nominal
production HEAD alone does not identify it. The original protocols, per-file
hashes and explicit retained-source inventory establish its identity. Unused
experiment drivers and raw checkpoints are omitted. Declared release-only
path substitutions, if any, are verified through the software archive's hash
manifest. The launcher refuses any unrecorded change.

Use a disposable extracted software copy: historical imports can create logs
beside the runtime. Install the dependencies described in RELEASE_WORKFLOW.md.
From `article_analysis/`:

```bash
python -B code/population_replay/launch.py --verify-only
python -B code/population_replay/launch.py --rule derivative --seed 2026100200 --output NEW_DIRECTORY
```

The default replays one original-bound Adam condition for 4,096 updates, using
its frozen development-selected rate and validation checkpoint selection.
`--common-rate` uses 0.03; `--separable` also uses 0.03 and removes only the
target interaction while retaining nonlinear parents. All six original rules
are available. `--smoke-steps 8` explicitly produces an excluded short smoke,
not another scientific observation. Every output directory must be new.

The bounded SGD development check can be replayed with:

```bash
python -B code/population_replay/launch.py --rule exact --seed 2026100100 --sgd-rate 0.3 --output NEW_DIRECTORY
```

It retains 32,768 updates and performs no test evaluation. Repeat across
seeds 2026100100–2026100102 and rates 0.003, 0.01, 0.03, 0.1, 0.3, 1 to reproduce
the grid. No setting passed the all-three-seeds NMSE <= 0.001 criterion, so no
fresh SGD cohort was launched. Protocol and endpoint tables are in Source Data.

These launchers keep scientific equations and random-number operations fixed,
but different numerical-library versions need not reproduce trajectories
bitwise. Outputs record the actual Python, Torch and NumPy versions. Original
selection/rescue checkpoints remain in the separately retained study archives;
the compact software package does not redistribute them. The complete original
source hash inventory remains in the frozen protocols, including omitted files.

## Figure 4 noise controls

After the documented Source Data restoration into `article_analysis/`, run
`python -B code/population_replay/noise_replay.py --journal . --seed 2026092100
--output NEW_DIRECTORY` (one line). This verifies the original source identities,
records any declared release substitutions and relocates the protocol's paths.
The frozen numerical runner is unchanged. Repeat for the twenty original seeds
2026092100–2026092119. `--smoke` retains the paired task/noise/rate construction
but stops after one update and explicitly excludes its outputs from evidence.

## Verified full trajectory replay

On 21 September 2026, a predetermined existing seed (2026100200) was replayed
for exact, resistance, derivative and shuffled-derivative delivery, each for
4,096 Adam updates at its original selected rate. The isolated Python 3.10.13
CPU environment used Torch 2.9.1+cpu and NumPy 1.26.4; original CPU execution
used Torch 2.9.1+cu128 and NumPy 2.2.6. All four selected checkpoint steps
agreed, as did all 132 validation-history points and every ordinary/stress
outcome under the frozen tolerances (relative 1e-6, absolute 1e-10). The maximum
absolute numerical discrepancy was 7.76e-14. This checks one existing seed and
task; it is not independent scientific replication or a claim of full-paper
reproduction. The four-rule job completed in 47 seconds on four allocated
HPC CPUs; this is an environment-specific observation, not a desktop estimate.
