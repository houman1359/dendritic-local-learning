# Explicit noise-task adapter

The historical name `noise_resilience` does not identify one generator. The release separates `legacy_noisy_lines` from `projected_noise_mnist`; see the cohort mapping in `source_data/release_task_identity/` or the release's `article_analysis/source_data_metadata/release_task_identity/`.

`frozen_generators.py` contains unmodified function/class bodies extracted from the two source versions. `ORIGINS.json` records their source and symbol SHA-256 values. The corresponding complete source texts are kept inert in `frozen_sources/` so integrity can be checked without importing the original optional experiments package. The noisy-line source is from the executed clean commit. The projected source is a newly frozen copy of the working implementation, not proof of the bytes executed in earlier cohorts.

Add this directory to `PYTHONPATH` to expose the optional `experiments.data_generation.synthetic_datasets` hook to the installed core. Use an explicit dataset name or specify `parameters.release_generator` for a legacy configuration. The adapter rejects ambiguous names and unsupported legacy parameters. It does not silently convert a noisy-line experiment into projected MNIST.

The projected generator imports the standard MNIST loader from the committed installable core. That loader and its Torch/torchvision dependencies ship in the software package; the original MNIST images remain an upstream download/cache dependency. The clean smoke replaces only this upstream image read with tiny image tensors and then exercises the actual frozen corruption functions.

The legacy wrapper takes an explicit seed and restores the caller's NumPy and Torch RNG states. Its generator and split match the old source for the same initial RNG states, but complete historical training replay also requires the original RNG timing and run configuration. The 784 × 784 default is intentional historical behavior. Tests use small images without changing the generator formula.
