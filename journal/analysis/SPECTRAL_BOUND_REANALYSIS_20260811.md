# Spectral/Ky--Fan reanalysis contract

This is a deterministic secondary analysis of the completed 50-seed spectral
credit phase; it does not introduce a new confirmatory cohort.

The analysis verifies two consequences of the review's spectral formulation:

1. every rank-K route projector is compared with the Ky--Fan upper bound, the
   sum of the K largest task-credit covariance eigenvalues divided by total
   spectral energy;
2. because the prescribed covariance depends affinely on alignment rho and
   has constant trace, the ancestry-minus-random capture contrast is affine.
   Endpoint contrasts therefore determine the exact within-seed alignment
   threshold. If ancestry is already favorable at rho=0, its required
   threshold is recorded as zero.

The paired unit remains the original independent simulation seed. Thresholds
are relative to the sampled random rank-matched route and are not interpreted
as universal biological constants.
