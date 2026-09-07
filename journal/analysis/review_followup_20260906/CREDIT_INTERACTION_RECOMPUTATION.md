# Independent recomputation of the Figure 4 learning interaction

The reported 0.838 contrast is a learning outcome in `source_data/credit_rule_bridge/summaries/`, not a credit-capture statistic in `credit_resolution_bridge`.

Starting from `all_curves.csv`, retain step 1024, `selected_rate=True`, model `algebraic`, optimizer `adam`, tasks `matching` and `quartet`, and rules `exact` and `calibrated_broadcast`. These filters yield 80 endpoints in 20 paired seeds. For each seed calculate:

    (quartet calibrated NMSE - quartet exact NMSE)
      - (matching calibrated NMSE - matching exact NMSE).

The mean is **0.8378172815696834**. The original 10,000 whole-seed bootstrap draws, recomputed independently with NumPy seed 210999, give the descriptive 95% interval **[0.7190435929903065, 0.9307531820366128]**. All 20 seed differences are positive. The matching-task means are 0.0235760734070933 (calibrated) and 0.0317261153059635 (exact); the quartet means are 0.9813028162601102 and 0.1516355765892970.

These results reproduce CSV line 51 in `paired_contrasts.csv`, whose keys are `selected_rate`, `algebraic`, `quartet_minus_matching`, `adam`, and `calibrated_broadcast minus exact interaction`. The adjacent JSON retains hashes of both source tables, the exact filter, formula and all seed differences. No source data or experiment result was modified.
