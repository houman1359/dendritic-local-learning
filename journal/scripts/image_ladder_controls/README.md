# MNIST intermediate dictionary and optimization controls

The frozen study trains the original MNIST network with six rules: strict
scalar, one signal per neuron, one projected profile, three projected subtree
profiles, exact path transport, and a decoder-only control. All arms retain the same 128-neuron
single dendritic layer, `[3,3]` topology, 92,160 active input contacts,
2,414,602 parameter slots, image streams, Adam parameter groups, and 180-epoch
budget. Validation loss selects the checkpoint. The three tested common
multipliers are 0.3, 1, and 3; three development seeds choose each rule's rate,
then ten new paired seeds test that rate and the original common rate.

The projected dictionaries act on **activation-space credit**. The one-column
dictionary replaces all twelve nonsomatic errors by their mean. Each
column of the three-column dictionary covers one proximal compartment and its
three distal children. The
coefficient is the average exact transported activation error within those
four compartments. Coefficient calculation therefore has access to the exact
field; this experiment tests the dictionary's delivery capacity and learning
behavior, not a local biological encoder. Activation derivatives remain in
local eligibilities for all parameters, including reactivation gates. The
archived route atlas instead measures voltage-space error; `capture.py` keeps
both coordinate choices and both field-weighting conventions explicit. The
K3-minus-projected-K1 contrast uses the same exact coefficient source and
preserves the same common-mode mean at an identical network state. K3
additionally retains differences between subtrees. This does not equate total
credit-vector norms or force later training trajectories to share a field. The
neuron-shared rule remains a separate, simpler reference.

`run.py` is the unchanged hash-frozen experiment adapter. It uses process-local
hooks and never edits the historical library. `protocol.json` and `freeze.json`
record the design fixed before development outcomes. `selection.json` is
written only after all 90 development fits complete; it reads validation loss
alone. A separate frozen addendum, `projected_k1.py` and
`projected_k1/protocol.json`, adds 18 development fits for the projected
K1 rule, which preserves the same common-mode mean at an identical network
state, before any fresh-seed outcomes. Both selections then precede all fresh
evaluation. The complete design contains 108 development and 190 distinct
fresh fits; selected and common-rate views reuse 50 identical conditions. All
candidate outcomes are retained. Twelve three-epoch canaries are implementation
checks and excluded from scientific summaries.

## Portable replay

1. Restore Source Data so `source_data/image_ladder_controls` contains its
   configs, protocol, conditions, and portable scientific inventory.
2. Extract the library snapshot at commit
   `6c1aaa25abd056c417842e1c46378b65d036f6a7` into a runtime directory with `src/`.
   `runtime_origin.json` lists every required Python source and its original
   hash. This runtime is distinct from the physical-depth runtime `a99c3a7`.
3. Install the released Python environment. The new fits record Python
   3.10.13, PyTorch 2.9.1/CUDA 12.8, torchvision 0.24.1, NumPy 2.2.6,
   SciPy 1.15.3, pandas 2.3.3, OmegaConf 2.3.0, PyYAML 6.0.2, and WandB
   0.23.1 (disabled). Older flagship records identify Python 3.10.13 but do not
   establish all historical package versions.
4. Download the official MNIST files, without transformation, using
   `torchvision.datasets.MNIST(root=DATA_ROOT / "mnist", download=True)` for
   training and test sets. Keep the raw compressed and decompressed files.
   The frozen protocol checks every raw file's SHA-256 before replay.
5. Run from this script directory:

   ```bash
   python portable_run.py \
     --study-root /path/to/source_data/image_ladder_controls \
     --runtime-root /path/to/historical_mnist_runtime \
     --dataset-root /path/to/data \
     --output-root /path/to/new_replay \
     --stage fresh --index 0
   ```

For an installation smoke test use `--stage canary --index 2` (shunting K3)
or `--stage canary --index 9` (additive decoder-only). A new output directory is
required. Only dataset and output locations are rewritten. A separate canonical
scientific configuration hash ensures that learning rates, seeds, model,
examples, stopping rules, and every other scientific setting remain fixed.
A second identity hash binds the architecture, arm, seed, multiplier and
condition index to that configuration. This also protects adapter-dispatched
rules such as K3 and exact transport, whose trainer settings are otherwise
identical.

If packaging sanitized runtime text, pass `--runtime-release-inventory` with
TSV columns `path`, `original_sha256`, and `released_sha256`. Original hashes
must match the frozen runtime, and actual bytes must match the released hash.
If the frozen adapter's I/O paths were sanitized, additionally pass
`--adapter-release-record` with JSON `original_sha256` and `released_sha256`.
No original hash is silently replaced with a hash of different bytes.

GPU reductions and library/hardware differences can change floating-point
trajectories; replay is defined by source, data and scientific settings,
not an unsupported promise of bitwise equality on every device. The local
canary validation records the stronger equality checks actually obtained.

For a packaged release, generate the transformation arguments automatically:

```bash
python prepare_release_links.py \
  --release-root /path/to/extracted/software \
  --study-root /path/to/source_data/image_ladder_controls \
  --output-dir /path/to/replay_links
```

This verifies the root `RELEASED_SOURCE_HASHES.tsv` and its complete declared
transformation chains, the frozen protocol, all restored configurations, and
the allowlisted runtime. It writes `runtime.tsv`, `adapter.json`, and
`capture.json`, and `projected_k1.json`. Supply the first two to `portable_run.py` using
`--runtime-release-inventory` and `--adapter-release-record`; also supply
`--capture-release-record` to `portable_capture.py`. For K1 replay, use a
`projected_k1_fresh`, `projected_k1_development` or `projected_k1_canary` stage and
also pass `--projected-k1-release-record /path/to/replay_links/projected_k1.json`.
The exported runtime path
is `historical_runtimes/image_ladder_6c1aaa2/` under the software root.

The release requires 502 allowlisted Python files (505 source files total),
covering the historical library and training dependency closure. The complete
664-file historical inventory remains an audit record; unrelated project
runners are not required or shipped. Canaries verify every executed historical
module belongs to the allowlist. The optional `MNIST_Capture_Checkpoints.zip`
contains the forty states needed to recompute the exact-checkpoint capture
without repeating training; it is distributed separately from compact code
and Source Data archives.

The environment recorded for scientific GPU fits used NumPy 2.2.6, although
the historical package metadata declares NumPy below version 2. A separate
clean CPU installation with NumPy 1.26.4 successfully executes the excluded
portability canaries. These are distinct checks: the clean installation does
not establish numerical equivalence to the complete GPU cohort.

For checkpoint-only capture replay, extract the optional checkpoint ZIP and run:

```bash
python portable_capture.py \
  --checkpoint-root /path/to/extracted/checkpoints \
  --study-root /path/to/source_data/image_ladder_controls \
  --runtime-root /path/to/historical_mnist_runtime \
  --dataset-root /path/to/data \
  --output-root /path/to/new_capture \
  --device cuda
```

Use the runtime, adapter and capture release records generated above when
replaying sanitized package files. The checkpoint manifest verifies all forty
states and their twenty resolved configurations before computing either credit
coordinate on the original 2,048 probes.
