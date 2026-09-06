# Exact Boolean morphology audit

This directory contains an exhaustive analytic capacity audit, not learning outcomes. Its implementation in `scripts/boolean_theory/` does not import the Boolean learning code or earlier morphology implementations. The seven task definitions and training input convention were agreed before the learning outcomes were available.

The domain comprises all 16 equally weighted assignments of independent bits `a,b,c,d`. The physical input is `x_i=2 bit_i−1`. Raw targets are 0 or 1; the learning target is `z=(y−E[y])/sqrt(Var(y))`. Consequently, population MSE for `z` equals raw truth-value MSE divided by `Var(y)`. All capacity bounds concern real-valued truth-value regression. They do not assert an error bound after thresholding for classification.

Each candidate is a rooted, unordered, full binary tree with four labeled leaves, three internal units, twelve scalar coefficients and six edges. A unit computes `u=θ0+θ1 l+θ2 r+θ3 lr`. Swapping its children can be absorbed by swapping its coefficients, so the exhaustive enumeration contains 15 trees: three of depth 2 and twelve of depth 3. Depth counts internal computations along the longest input–output path. A subtree is represented by one scalar and each input occurs at one leaf.

## Files and exactness

- `truth_tables.csv`: 112 rows, with raw and normalized outputs. Pattern index uses `a` as its least significant bit.
- `walsh_coefficients.csv`: all 16 coefficients for each task. Raw coefficients are exact rational numbers; normalized coefficients are floating-point values. The normalized constant coefficient is zero.
- `target_summary.csv`: exact raw means and variances, compatible-tree counts, minimum depth, and input-gradient second-moment spectra. The gradient is that of the unique multi-affine extension with respect to physical `x`, not the bit coordinate. This is an **uncentered second moment**, not a centered gradient covariance. Rank is computed with rational arithmetic; eigenvalues are numerical.
- `all_cuts.csv`: 98 task–cut pairs for every nonempty proper input subset. Each matrix drops only the constant row on the subtree side. The retained rational matrix permits independent reconstruction. Rank uses exact rational row reduction. Squared singular values and their rank-one tail use float64 SVD.
- `tree_cuts.csv`: the two nonroot internal cuts for each of the 105 task–tree pairs, 210 rows.
- `tree_capacity.csv`: all 105 pairs, tree expression, depth, maximum cut rank, raw and normalized lower bounds, exact-construction status and parameter certificate. A tree bound is the **maximum** of its cut bounds; overlapping bounds are not summed.
- `depth_summary.csv`: gate counts, exact-compatible counts and minimum depth. The supplied mixed-gate formulas count two AND plus one OR for both `or_of_ands` and `nested`; their exact minimum depths are 2 and 3. Their full input-gradient spectra are not equal.
- `main6a_grouping.csv`: the three balanced groupings for XOR of two AND branches, including source-driven schematic labels and the exact `8/15` normalized crossed-grouping bound.
- `exact_constructions.json`: 49 rational raw-output certificates plus normalized coefficients. Nonroot states use a signed Boolean slice of the full target. Root coefficients alone are centered and scaled. This gauge differs from the canonical 0–1 gate illustration and need not match the internal states learned in an experiment.
- `gate_credit_fields.csv`: 101×101 branch-output grids for each canonical AND, OR and XOR gate, including both partial derivatives. `gate_derivative_corners.csv` contains the 12 Boolean corners. `canonical_gate_derivatives.pdf/.png` visualize the derivative with respect to the left branch.
- `canonical_root_credit_by_pattern.csv`: the realized canonical branch values and raw/normalized root derivatives for the three mixed balanced tasks. XOR-of-AND gives negative left-branch derivative on 4 of 16 patterns; OR-of-AND gives zero on those patterns; AND-of-XOR gives zero on 8 of 16 patterns.
- `METHODS_AND_PROPOSITIONS.tex`: integration-ready protocol, proof, qualifications and schematic content. It is not automatically included by a manuscript build.
- `report.json`, `validation.json`: audit counts, exact checks, numerical methods and source/table hashes. `test_validation.json` and `pytest_output.txt` separately record the focused test run and a replay of every exported normalized certificate.

All 49 compatible cases have an exact rational raw truth-table reconstruction. For the normalized construction, the coefficient bound is checked by squaring it in rational arithmetic before taking the normalization square root. The largest squared coefficient is exactly `16/15`, so the largest absolute coefficient is `4/sqrt(15) ≈ 1.032796`, strictly below the training box limit of 2. This is an existence certificate, not a convergence guarantee from the experimental initialization.

The three associative targets (AND4, OR4 and parity4) are exactly representable on all 15 trees. Each mixed target has exactly one compatible tree in the exhaustive enumeration. Incompatible trees have a strictly positive cut obstruction; compatible trees have an explicitly verified exact construction. Thus the zero-bound classification is an exact capacity classification for these 105 cases, rather than an inference from a zero lower bound alone.

The canonical gate derivative grid illustrates a possible mechanism: for learned branch outputs `u,v`, root XOR has derivative `1−2v`; OR has `1−v`; AND has `v`. These facts do not prove that a fixed broadcast rule cannot learn. Internal affine reparameterizations change local derivative signs and scales, and optimizer-dependent learning outcomes must be measured. A two-input XOR alone has no learned internal subtree in this architecture. The Boolean analysis applies to the algebraic scalar tree; a monotone positive-conductance model does not acquire XOR capability from this calculation.

## Reproduction

Run from the journal directory with NumPy and Matplotlib installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/boolean_theory/audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/boolean_theory/validate.py
```

The second command also requires pytest. It tests exact hand-derived crossed-cut matrices and bounds, exhaustive counts, all compatible exact constructions, a same-gate-count depth obstruction, numerical gate derivatives and lower-bound consistency for independently evaluated random polynomial trees. It then parses and evaluates the exported normalized construction coefficients without importing the audit implementation. No fitting, hyperparameter selection, Monte Carlo uncertainty or held-out training data enter this audit.

## Independent learning-output validation

`scripts/boolean_theory/validate_learning.py` reads the separate Boolean learning experiment's saved weights and child indices without importing its implementation. It reconstructs all 8,400 development/fresh endpoints and their initial states, checks semantic input permutations against this audit's truth tables, compares population/test MSE and classification summaries, checks all recorded checkpoint MSEs against the exact-theory tree bounds, recomputes development-only rate selection, and independently reproduces the two primary paired contrasts and bootstrap intervals. Its outputs are `learning_endpoint_validation.csv` and `learning_validation.json`. This is endpoint replay, not a claim to have rerun complete training trajectories. No learning files are modified.

## Post hoc target-projection diagnostic

`proper_subtree_projection_energy.csv` and `posthoc_projection_validation.json` are explicitly **post hoc**, requested after the fresh Boolean learning results were available. Run `python scripts/boolean_theory/posthoc_projection.py` to reproduce them. The script computes `P_S=||E[z|x_S]||²` from the target's Walsh energy on nonempty masks contained in `S`, then checks all 98 values against direct conditional means with exact rational arithmetic. Pure parity has `P_S=0` for every proper subset. XOR-of-AND has `P_S=1/5` for each two-input subset, including both compatible and crossed pairs, so this quantity is not a grouping-compatibility measure.

At fixed model parameters, a local eligibility `e_S(x_S)` at a nonroot subtree depends only on its own inputs. Conditional expectation gives `E[z e_S]=E[E[z|x_S] e_S]`; thus a zero `P_S` removes the direct target term from its population broadcast update `E[(prediction−z)e_S]`. This does not establish an inability to learn: root coefficients can use labels and then alter the prediction term, and finite training samples and SGD noise do not preserve an exact population cancellation. The diagnostic explains a difference in accessible lower-order target information, while the measured learning comparison determines its consequences under the specified optimizer recipe.
