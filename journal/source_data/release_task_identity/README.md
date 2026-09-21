# Historical noise-task identity audit

The key `noise_resilience` has two confirmed implementations. A task name alone does not identify the generator that ran.

| Cohort | Noise fits | Generator identity | Evidence boundary |
|---|---:|---|---|
| clean exact-path/backpropagation rerun | 160 | three-class noisy-line images | The newer projected-MNIST parameters in a launch YAML are not interpreted by this older dispatcher. |
| prospective depth-by-feedback | 320 | projected-noise MNIST intended; consistent with retained model dimensions and outcomes | A live current nested source is not proof of historical executed bytes; do not label the runtime source fully recovered. |
| prospective routing, inhibition-dose and fixed-budget follow-ups | 960 | projected-noise MNIST intended; exact runtime generator remains qualified | Do not infer executed task identity solely from the noise_resilience name or from signed-transfer validity. |
| inherited synthetic feedback and gradient cohorts | unresolved | unresolved historical noise task | Do not retrospectively assign either projected-MNIST or frozen NoisyLine to these plotted aggregates. |
| forward-depth and interference screens inherited from original figures | unresolved | requires panel-specific runtime/config join | No global dataset-key-to-generator mapping is scientifically justified. |

## Frozen sources

- The executed clean nested commit is `4f3612a52756c3691e5bd269c6465c02f240c1e8`; its synthetic source SHA-256 is `6a20a2beed0fa0e7b4019e2c32ee1cbbd66ca147b6e1b4ce86c38094eacc9c25`.
- The current nested working file has SHA-256 `6ef17bc41739409eb599230d608bba2e19eb3cb94763f568ac9fd439645eafdb` and differs from that commit. It supplies a newly frozen projected-MNIST reference; it does not prove which bytes older runs executed.

The three-class generator constructs 784 × 784 images, flattens them to 614,656 inputs, and adds independent Gaussian pixel noise with standard deviation 0.2. The projected-MNIST protocol uses 784 inputs, ten classes, a fixed 784 × 50 column-normalized Gaussian projection, latent-noise scale 1.5 and clipping to [0,1]. They are not interchangeable controls.

Full evidence filenames and SHA-256 values are recorded in `task_identity.json`. `task_identity.csv` provides a compact machine-readable cohort map. Historical raw configurations/checkpoints are not reconstructed from current code.
