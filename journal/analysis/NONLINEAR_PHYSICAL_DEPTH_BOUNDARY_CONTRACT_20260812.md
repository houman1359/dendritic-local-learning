# Single-point accessibility boundary resolution

Frozen: 12 August 2026, after the four-level signal ladder and before observing
signal 0.80 outcomes.

The only tested signal is 0.80.  All production-model, resource, coupling (16),
gain (matched train/test SD 0.25), optimization and data-generation settings
are unchanged.  The three exact-resource depths use two new seeds 10140 and
10141, for six exact-backpropagation fits.

The boundary passes only if every depth has mean test accuracy in [0.60,0.95],
every individual seed exceeds 0.57, and D3 minus D1 is at least +0.02 in both
seeds.  If it fails, no further signal calibration is allowed and the task is
not advanced.  If it passes, signal 0.80 is frozen as the operating point for
a later prospectively specified, sufficient-seed replication.  This boundary
check itself remains exploratory.
