# Prospective local conductance-gate follow-up

Protocol SHA256: `c441998f6f91e1f8782b2918689806e119af567e479bcd5c94069dca837a2413`. Twenty entirely fresh seed blocks; 1,080 trajectories; all continue to 16,384 updates. Prior panel exploratory results are excluded.

## Primary Adam 0.03 outcomes at the 4,096-update window

| rule                                   |   aligned_strong |   opposed_strong |
|:---------------------------------------|-----------------:|-----------------:|
| exact                                  |     7.531509e-06 |     2.178933e-05 |
| unit_broadcast                         |     0.0005533698 |     0.4873967    |
| calibrated_broadcast                   |     0.0005533773 |     0.4873993    |
| ancestry_three_oracle                  |     9.169546e-06 |     2.250196e-05 |
| hard_distal_unit_proximal              |     1.055373e-05 |     2.327788e-05 |
| swapped_distal_unit_proximal           |     0.005380656  |     0.9492567    |
| hard_distal_and_proximal               |     1.196252e-05 |     0.1232042    |
| ancestry_two_leaf_oracle_unit_proximal |     9.072781e-06 |     2.279912e-05 |
| shunt_proportional_unit_proximal       |     1.052527e-05 |     2.321928e-05 |

## Primary paired tests

| task                  | contrast              |           p |     holm_p |
|:----------------------|:----------------------|------------:|-----------:|
| opposed_strong        | unit_minus_local_gate | 1.90735e-06 | 3.8147e-06 |
| opposed_minus_aligned | unit_minus_local_gate | 1.90735e-06 | 3.8147e-06 |

## Prespecified practical precision reference

{
  "hard_minus_exact_upper95": 4.857309109337419e-06,
  "upper95_below_001": true,
  "hard_maximum": 5.473764983213735e-05,
  "every_seed_below_01": true,
  "scope": "Prespecified descriptive practical-precision reference; not an asymptotic equivalence test."
}

These gates use externally supplied local inhibitory context and a somatic error. They do not discover an endogenous context signal or error. The two-leaf profile rule uses oracle projection coefficients and unit proximal credit; it is not a wholly two-column six-site dictionary. Gating the proximal credit deliberately freezes two inhibitory gains through its interaction with local eligibility. All such outcomes are retained.

Complete selected/fixed endpoints, all rates, seed contrasts, parameters and source hashes accompany this report. These are finite-budget outcomes.
