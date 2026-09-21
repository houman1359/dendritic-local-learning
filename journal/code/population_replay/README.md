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
