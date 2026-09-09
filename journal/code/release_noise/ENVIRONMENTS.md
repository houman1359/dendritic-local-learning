# Scientific execution and supported replay environments

The article combines studies executed at different times. A source commit, an
execution environment and a figure reconstruction environment are separate
identities. The release records each where available; it does not claim that
one installation reproduces every historical floating-point trajectory.

| Work | Scientific implementation and execution record | Supported replay or reconstruction |
|---|---|---|
| MNIST dictionary, rate and decoder controls | Historical core `6c1aaa2`; `source_data/image_ladder_controls/runtime_origin.json` identifies the exported source. Runs recorded NumPy 2.2.6, PyTorch 2.9.1/CUDA 12.8 and torchvision 0.24.1. | Use the historical runtime and the portable launcher in `scripts/image_ladder_controls/README.md`. The isolated NumPy 1.26.4 CPU check establishes source/import portability, not identical GPU trajectories. |
| Physical-depth extension | Historical core `a99c3a7`; the canonical extension protocol and `environment_overlay.json` record the Python 3.10/PyTorch 2.9.1 environment and the WandB/protobuf overlay. | `physical_depth_launcher.py` verifies the historical source. The supported CPU environment retains the frozen task, rate, seed and budget; device differences can change numerical trajectories. |
| Matched interaction and conductance studies | Standalone model code and frozen study protocols under `scripts/credit_rule_bridge/`, `scripts/credit_rule_extension/`, `scripts/conductance_credit_demand/` and `scripts/conductance_local_gate/`. The manuscript and run records identify NumPy 2.2.6 execution where recorded. | Use each study's documented launcher and record its environment in a new output directory. These models do not all import the installable DendriNet package. |
| Historical auxiliary cohorts | The study-specific manifests identify retained source and execution information. Some historical runtime/task-identity information is incomplete, as documented in `source_data/release_task_identity/`. | A new package installation cannot recover missing execution history. Retained summaries, declared source identities and newly generated replays must remain distinguishable. |
| Current publication figures | Retained numerical tables and authenticated vector inputs; `scripts/rebuild_final_publication_figures.py` is the entry point. Rendering provenance records the actual libraries used. | Restore Source Data at the manifest's `original_source` destinations. No model training is needed. Use the requirements and fonts below; verify the regenerated displays against the packaged artifacts. |

Paths beginning `source_data/` refer to the restored journal workspace. The
software archive contains two historical runtime exports in addition to its
current installable core. Do not substitute the current core for a frozen
historical implementation simply because their package names agree.

## Isolated CPU compatibility environment

`validated_environment.txt` records the resolved Python 3.10.13/Linux x86-64
environment actually used for the isolated CPU validation. It is an execution
record, not a reconstruction of the original GPU installations. To recreate
its package set from the software release root in a fresh virtual environment:

```bash
python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r article_analysis/code/release_noise/validated_environment.txt
REVIEWER_PACKAGE_COPY=$(mktemp -d)
cp -R dendritic_modeling/. "$REVIEWER_PACKAGE_COPY/"
python -m pip install --no-deps "$REVIEWER_PACKAGE_COPY"
python -m pip check
```

The smaller `constraints.txt` instead fixes the known compatibility choices
while allowing the core's other declared dependencies to resolve. It pins
NumPy 1.26.4, WandB 0.23.1, protobuf 6.33.5 and PyMuPDF 1.28.2. Installing with
that file alone does not claim a complete environment lock. The public core's
NumPy `<2` declaration and scientific runs using NumPy 2.2.6 therefore describe
different environments; silently relaxing that declaration would not validate
NumPy 2 compatibility.

Figure assembly imports PyMuPDF as `fitz`. Native rendering also needs NumPy,
pandas, SciPy and Matplotlib. The vector compositor requires Nimbus Sans
Regular/Bold from the external `urw-base35` font package at the paths checked
by the renderer. Manuscript compilation additionally requires pdfLaTeX and
BibTeX. Python dependency files do not install these system fonts or TeX.

## What a validation result establishes

- Source-hash checks establish which implementation and data were used.
- Import and short training smokes establish bounded execution in the recorded
  environment; they are excluded from scientific cohorts.
- Figure reconstruction checks establish agreement with retained display data
  and, when tested, the packaged PDF bytes.
- Full scientific replay is a separate experiment. Record its device, package
  versions, seeds and output directory; do not overwrite the retained cohort.

The code-availability statement distinguishes the supplied reviewer archives
from the planned public versioned deposit. Public package CI does not establish
that the paper's complete experiments ran under that CI environment.
