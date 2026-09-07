# Signed changes under physical cable calibration

This analysis recovers the sign of the gradient-magnitude changes used in the
physical-calibration comparison. It replays the exact archived focal sites at
the two existing membrane-resistance endpoints, 300 and 15,000 ohm cm², at
axial resistivity 150 ohm cm. All other electrical parameters and the
soma-restoring current control retain their original definitions. No
parameters were fitted and no focal sites were resampled.

`signed_category_effects.csv` contains descendant and depth-matched off-route
site medians for each focal intervention. The signed metric is

    log[(abs(gamma_after) + epsilon) / (abs(gamma_before) + epsilon)],

where gamma is the excitatory-conductance gradient and the numerical floor is
1e-15 times the larger of one and the maximum baseline gradient magnitude.
Negative values mean attenuation; they do not imply reversal of the gradient
sign. None of these interventions reverses a gradient sign.

`signed_cell_effects.csv` averages the site medians over focal locations
within each cell. `signed_cohort_summary.csv` then averages cells with equal
weight, retaining the initial eight-cell and disjoint 45-cell cohorts
separately. Descriptive confidence intervals use 20,000 whole-cell bootstrap
draws. Figure 7C plots the focal-shunt rows; current-injection rows are also
retained here for direct comparison.

At membrane resistance 15,000 ohm cm², mean descendant/off-route signed log
changes are -0.014769/-0.007081 in the initial cohort and
-0.010375/-0.006444 in the disjoint cohort. At 300 ohm cm², they are
-0.116546/-0.006448 and -0.098881/-0.005249. Lower membrane resistance thus
increases descendant attenuation much more than off-route attenuation.
The high-resistance condition still changes gradients on both sides of the
route. The small shunt-minus-current localization contrast in Figure 7E is a
statement about spatial selectivity relative to that matched control.

`replayed_focal_localization.csv` preserves the corresponding absolute
endpoints. All 1,344 original focal rows reproduce: the maximum absolute
error across the three gradient-change metrics is 3.44e-12. The larger
5.72e-12 maximum across all checked columns occurs in the soma-restoring
current. `validation.json` records original source hashes, output hashes,
the focal site identities, complete cell-level checks, and the bootstrap
definition.

Run `python scripts/shunt_ancestry_gain/analyze_signed_calibration.py` from
the journal directory to reproduce these tables. The script calls the
original physical-conductance builder and the original exact-gradient,
somatic-compensation, relation and category-summary helpers.
