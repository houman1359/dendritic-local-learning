# Same-span noisy coefficient-learning result

All 4,800 expected rows from 50 paired confirmatory seeds completed. The
tree-Haar, raw nested and statically scaled nested dictionaries had the same
rank-eight projector to 1.45e-30 target residual;
their positive-spectrum Gram condition numbers were 1, 15 and 388.52.

The preregistered unconditional Haar advantage was falsified in the smallest
sample regime. At effective sample size 4, raw-minus-Haar final loss was
-0.277403 (-0.325975---0.231342)
and scaled-minus-Haar was -0.190551
(-0.250016---0.133489); negative values mean
that slow coordinates filtered enough update noise to improve population loss.

The ordering reversed as observations became reliable. At effective sample
size 256, raw-minus-Haar loss was 0.025554
(0.021712--0.029524) and scaled-minus-Haar was
0.170907
(0.145764--0.196859), positive in
50/50 and 50/50 seeds.

The exact linear finite-time risk separates residual bias from accumulated
variance and predicts median effective-sample crossovers of
38.53 for raw nested and
8.25 for scaled nested coordinates.
Across displayed condition means, the largest absolute observed-minus-expected
loss was 0.0634. Gram-preconditioned field trajectories
agreed across all three parameterizations to 1.16e-15, proving that the effect
comes from finite coefficient dynamics rather than address span.

This is a rate-based bias--variance result. It does not identify a biological
Gram preconditioner or imply that ill-conditioning is generally beneficial.
