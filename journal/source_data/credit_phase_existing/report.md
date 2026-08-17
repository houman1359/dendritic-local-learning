# Credit-phase reanalysis of the frozen subtree factorial

- Frozen trained fits: 2,700
- Reanalysed paired seeds: 20
- Minibatches per seed: 64 at batch size 64
- Utility versus norm-matched one-step progress: Spearman rho = 0.937
- Capture versus norm-matched one-step progress: Spearman rho = 0.877
- Utility versus final held-out accuracy: Spearman rho = 0.916

The smoothness-bound utility combines retained population-gradient signal,
finite-step gain cost, and admitted minibatch-gradient noise. Spectral capture
and morphology regret are computed from the exact task-coefficient covariance.
The analysis is diagnostic at the frozen initialization and does not turn a
same-model local update into an optimizer that uniformly exceeds exact
full-batch backpropagation.
