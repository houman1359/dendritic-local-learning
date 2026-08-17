# Spectral upper-bound and alignment-threshold reanalysis

This deterministic secondary analysis uses the frozen 50-seed spectral phase.
For every rank and seed, dense principal-eigenspace capture obeyed the Ky--Fan
upper bound; the maximum apparent violation was
0.00e+00. At alignment one, the ancestry
dictionary attained the bound at K=4 to numerical precision (mean regret
0.00e+00).

Because covariance is an affine mixture with constant trace, each paired
ancestry-minus-random contrast is affine in alignment rho. At K=4, ancestry
was already favored at rho=0 in 19/50 random
route draws and favored at rho=1 in 50/50. Treating an
already favorable seed as requiring zero alignment, the median minimum
alignment was 0.0173; its mean was
0.0523 (95% paired-seed bootstrap interval
0.0336--0.0730).

The threshold is relative to a sampled random rank-matched route, not a
universal biological constant. The Ky--Fan result is an oracle capacity upper
bound and does not provide a local circuit for learning principal modes.
