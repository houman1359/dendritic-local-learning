# Reviewer software release

The final archive must be built after the scientific integration commit. The builder refuses a dirty paper checkout and verifies that both recorded source commits exist and are reachable from named Git branches, tags or remote references. It exports the recorded committed objects and copies article analysis from that export; subsequent working-tree edits cannot enter the archive.

From the journal directory, after committing the integrated paper:

```bash
python scripts/build_software_release.py --force
```

For an isolated paper checkout, supply `--implementation-root` pointing to the real installable implementation repository. Keep the integration commit reachable in the canonical paper repository before packaging it; a detached snapshot existing only elsewhere does not establish canonical provenance. The release records commit IDs, tree IDs, containing refs, file checksums and every release-only portability/packaging patch. It does not upload or submit anything.

The explicit allowlist contains the installable scientific library and its training/sweep drivers, article analysis/configurations/tests, manuscript sources and figure assets. Public package imports require a few generic replacement-related library modules; these remain to preserve the API. Unrelated transformer/text/vision project drivers, production project configurations/tests/docs, presentations, internal revision logs, raw data and trained checkpoints are omitted. Entry points whose drivers are omitted are removed from the release copy of `pyproject.toml`, with original and released hashes recorded. No live production source is modified.

## Draft validation before the integration commit

Use a new output directory for every preview:

```bash
python code/release_noise/validate_draft.py --output analysis/software_release_draft_NAME
```

This creates a clearly marked draft from the committed implementation and allowlisted current paper files. It never writes under `submission/`. Its metadata explicitly says the paper files are uncommitted; the last paper commit is not presented as their identity. The draft validates policy exclusions, complete checksums, ZIP membership and CRCs. It is a staging check, not the final release.

## Clean environment and smoke

Use a virtual environment without system site-packages. A CPU-only environment avoids downloading unnecessary CUDA libraries:

```bash
python -m venv /path/to/reviewer-venv
/path/to/reviewer-venv/bin/python -m pip install --upgrade pip
/path/to/reviewer-venv/bin/python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
REVIEWER_BUILD=$(mktemp -d)
cp -R SOFTWARE/dendritic_modeling/. "$REVIEWER_BUILD/"
/path/to/reviewer-venv/bin/python -m pip install -c SOFTWARE/article_analysis/code/release_noise/constraints.txt "$REVIEWER_BUILD[test]" PyMuPDF
/path/to/reviewer-venv/bin/python -m pip check
/path/to/reviewer-venv/bin/python -I -B SOFTWARE/article_analysis/code/release_noise/cleanroom_smoke.py --release-root SOFTWARE
```

Replace `SOFTWARE` with the extracted release directory. The private build copy prevents setuptools from writing build artifacts into the checksummed archive contents. The smoke imports the installed model/training entry points, exercises the installed core's released noise-loader hook and verifies both noise transformations on tiny tensors. It asserts that the core comes from the fresh virtual environment. It does not download MNIST or stand in for full model training. `code/release_noise/cleanroom_worker.sh` provides the same bounded CPU-queue check in the local Slurm environment; it records `pip freeze`, `pip check`, smoke results and focused test results.

The tested compatibility constraints include NumPy 1.26.4, WandB 0.23.1 and protobuf 6.33.5. WandB 0.23.1 cannot import with the shared environment's protobuf major version 7. The core's remaining dependencies are declared in its released `pyproject.toml`; the clean installation report records their resolved versions. PyMuPDF supplies the figure assembly module `fitz`. Figure recreation also uses Matplotlib, NumPy, pandas and SciPy. PDF manuscript builds require an external TeX installation. MICRONS/CAVE and DANDI/NWB analyses additionally require their service clients, original data/cache access and any upstream credentials; these data and credentials are not distributed. MNIST, Fashion-MNIST and CIFAR-10 require upstream torchvision data or an existing cache.

## Historical noise-task identities

Read `source_data/release_task_identity/README.md` and `task_identity.json`. The same historical key refers to different generators in different cohorts. The release supplies two explicit choices through `code/release_noise/`: `legacy_noisy_lines` is recovered from the executed clean synthetic-data commit; `projected_noise_mnist` is a newly frozen reference for the intended projected-noise protocol. Older projected-MNIST executions and inherited aggregates retain the documented lineage gaps. Set `PYTHONPATH` to the adapter directory to activate the installed core's optional compatibility hook; select an explicit dataset name or `parameters.release_generator` in a historical configuration. An ambiguous `noise_resilience` request fails with an explanatory error.

The frozen generator bodies retain their original equations and random-number operations. The explicit legacy wrapper accepts a seed and preserves the caller's random state; reproducing an archived training trajectory additionally requires that run's model initialization, data-generation RNG timing and complete configuration. The legacy default constructs 784 × 784 images, so full data generation requires substantial memory; the clean smoke uses 8 × 8 images solely to test the same implementation. The default dimensions must not be silently reduced when reproducing the historical cohort.

After the final committed release passes these checks, rebuild the Source Data, manuscript/Overleaf and submission bundles in their dependency order and inspect their inventories separately. The software builder's exclusions do not automatically remove internal logs from other bundle builders.

## Restoring released evidence without changing canonical hashes

The software README uses `code/release_noise/restore_source_data.py` to restore data by `original_source`, including evidence now displayed under `Methods/retained_evidence/`. The helper verifies every released file, uses complete copies rather than display-filtered subsets, and writes `RELEASED_SOURCE_HASHES.tsv` plus an unchanged copy of the source-package manifest. A code file included as supporting Source Data is verified and reported without overwriting the committed software copy.

Canonical code/data manifests retain their original hashes. The audit helper `release_hashes.py` accepts a changed portable copy only when its original hash matches that canonical expectation, its actual bytes match the released hash, and its declared transformation provenance verifies. Software links additionally check every step of `PORTABILITY_PATCHES.tsv`; no undeclared source changes are accepted. Removed private run-directory columns cannot be reconstructed from the released numeric tables. Checkpoint-level reanalysis therefore needs separately supplied or newly generated run records.


## Historical physical-depth runtime

The software archive separately exports the selected source files at implementation commit `a99c3a777f99913e13dfe673a3f3a28bfe3566af`, under `historical_runtimes/physical_depth_a99c3a7/`. `RUNTIME_ORIGINS.tsv` records each original source hash; `RUNTIME_PROVENANCE.json` records the verified commit, tree and containing refs. The archive's released-source manifest links any declared portability edits. The export does not pretend to contain a Git checkout. Replacing this runtime with the current installed core would fail the frozen source audit.

After copying the paper into the documented implementation layout, preserve its source-provenance chain before restoring Source Data:

```bash
python SOFTWARE/article_analysis/code/release_noise/release_hashes.py \
  --remap-paper --release-root SOFTWARE \
  --journal-root SOFTWARE/dendritic_modeling/drafts/dendritic-local-learning/journal
```

This verifies the copied bytes and retains the original software paths in an explicit relocation record. Numerical restoration merges its sidecar without discarding these software identities. The original canonical hashes remain unchanged.

Run the portable depth launcher in a fresh process and use the same constrained environment as the clean smoke:

```bash
/path/to/reviewer-venv/bin/python -B SOFTWARE/article_analysis/code/release_noise/physical_depth_launcher.py \
  --source-root SOFTWARE/dendritic_modeling/drafts/dendritic-local-learning/journal/source_data/physical_depth_budget/canonical \
  --journal-root SOFTWARE/dendritic_modeling/drafts/dendritic-local-learning/journal \
  --runtime-root SOFTWARE/historical_runtimes/physical_depth_a99c3a7 \
  --condition 30 --verify-only
```

Replace `--verify-only` with `--output-root NEW_OUTPUT_DIR` to rerun the chosen frozen condition. Each output directory must be new. The 600-epoch cap, validation-based checkpoint selection, early-stopping patience, initialization seed and learning rate are retained. `--smoke-epochs 1` is an explicitly excluded short run, useful for checking both the exact-gradient and LocalCA training paths without adding scientific observations. The frozen runner is preserved, including its machine-specific checkout check; the portable launcher instead verifies the historical export's bytes and uses only its verified passive observation helpers. Device and library differences can prevent bitwise trajectory identity, and their versions are recorded. This task generator is synthetic and needs no external data.
