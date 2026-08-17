# Mid-array source-equivalence record

## Event

The physical alignment-dose array (job 38876091) and the first Fashion-MNIST
feedback arrays (jobs 38877680 and 38877693) froze the root tracked-diff SHA256

`966291620f8f9344d1c6df09353d096f8f9c003351ce779d23db91066a438705`.

At 01:57 EDT on 13 August 2026, a concurrent editor changed
`branch_dynamics.py`. The generated source guard rejected every task that
started after that edit with exit code 67. The not-yet-started tasks were
cancelled; completed outputs were not deleted or overwritten. New sweep
manifests froze the updated tracked-diff SHA256

`3d71a800ecca000d7767c6bb302879f418abba9738e7335b57d02a019eb6c937`.

Only missing condition indices were submitted from the new manifests.
The analysis scripts key rows by scientific condition and reject or report
duplicate completed conditions.

## Why the two hashes are scientifically equivalent here

The intervening change adds analysis-only branch-voltage rules selected by the
private attribute `_analysis_rule_override`. Neither the physical alignment
dose configurations nor the Fashion-MNIST feedback configurations install that
attribute. With the attribute absent, `analysis_rule is None` and the ordinary
shunting and additive forward paths execute the same equations. No experiment
configuration, seed, optimizer, data generator, learning rule, route mode or
resource budget changed.

This is a control-flow equivalence claim for these configurations, not a claim
that arbitrary uses of the two source states are identical. The final source
tables retain run directory, configuration hash and result hash for every row.

## Verification

After the edit and before resubmission:

- all 30 focused physical-depth, point-control and resource-contract tests
  passed (`30 passed`);
- the current `branch_dynamics.py` SHA256 was
  `30a728b491cbca35d10cea2e5cc822640d15afcbc42beafbe70c5f5b876ece20`;
- all 64 completed alignment-dose fits under the first hash were finite, used
  the expected resources and contained no fallback or non-finite alerts;
- failed tasks produced no accepted final metrics.

The final audit must show exactly 90 unique alignment-dose rows and 60 unique
Fashion-MNIST rows before either result is reported.
