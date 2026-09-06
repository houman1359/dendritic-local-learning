# Boolean learning results

All 6,720 fresh fits completed without a failure or exclusion. Both primary
comparisons use the predeclared Adam recipes and twenty paired seeds.

| Primary XOR-of-AND comparison | Mean clean-NMSE improvement | Adjusted 97.5% interval | Prespecified 0.01 criterion |
|---|---:|---:|---|
| Mean two crossed balanced pairings minus compatible pairing, exact credit | 0.563174 | [0.549144, 0.582749] | Passed |
| Broadcast minus exact credit, compatible pairing | 0.002507 | [0.001780, 0.003318] | Not met: mean effect below 0.01 |

The compatible exact-credit model has mean clean NMSE 0.000945, compared with
0.564119 averaged over the two crossed pairings. This is a strong effect of
input grouping within the fixed tree class. The compatible broadcast model
also learns the task, with mean NMSE 0.003452; both rules achieve perfect
exhaustive threshold classification in every fresh seed. The credit difference
has a positive adjusted interval but is too small for the predeclared practical
criterion. It must not be described as either a large credit effect or an
absence of any measurable difference.

Same-rate Adam credit differences at 0.003, 0.01 and 0.03 are respectively
0.001891, 0.001701 and 0.002498 NMSE. The compatible-grouping advantage is
0.563174, 0.562725 and 0.566698 at those rates. These are descriptive checks,
not additional primary tests. The learner trains all node coefficients and
can change internal coding: a signed sensitivity in one canonical Boolean
gate construction does not prove unit broadcast cannot learn that function.

The following table uses development-selected Adam rates. One compatible
candidate is shown per family; all four candidates and both optimizers remain
in the numerical tables and supplementary display. AND4, OR4 and parity have
multiple compatible candidates, so the displayed candidate is not a uniquely
preferred morphology.

| Template | Shown compatible tree | Exact NMSE | Broadcast NMSE | Exact / broadcast balanced accuracy |
|---|---|---:|---:|---:|
| and4 | balanced_ab_cd | 0.001266 | 0.003765 | 1.0000 / 1.0000 |
| or4 | balanced_ab_cd | 0.000942 | 0.003402 | 1.0000 / 1.0000 |
| parity4 | balanced_ab_cd | 0.000830 | 0.976114 | 1.0000 / 0.5375 |
| or_of_ands | balanced_ab_cd | 0.001044 | 0.001916 | 1.0000 / 1.0000 |
| xor_of_ands | balanced_ab_cd | 0.000945 | 0.003452 | 1.0000 / 1.0000 |
| and_of_xors | balanced_ab_cd | 0.000895 | 0.002240 | 1.0000 / 1.0000 |
| nested | comb_a_b_cd | 0.001015 | 0.002473 | 1.0000 / 1.0000 |

Parity provides a large descriptive credit effect: exact-credit mean NMSE is
0.000793–0.000995 across the four candidates, with perfect classification;
broadcast gives 0.892780–0.976114 NMSE and balanced accuracy 0.5375–0.58125.
This result does not replace the predeclared XOR-of-AND credit comparison.
For the nested target, the comb learns near the noise-estimation floor whereas
balanced exact-credit candidates retain NMSE 0.134585–0.173875. Associative
AND4 and OR4 learn across all four trees. Mixed tasks distinguish compatible
and crossed groupings, with the magnitude and credit dependence varying by
logical template. These are results for seven fixed functions, not a sampled
population of Boolean functions.

Optimizer/rate and parameter-bound sensitivities remain part of the record.
Across all fresh rates, Adam/broadcast projects parameters at the box boundary
on 110,887 summed fit-steps; Adam/exact on 506, SGD/broadcast on 18,065 and
SGD/exact on zero. These sums are nested optimization diagnostics, not sample
sizes. Gradient clipping occurred on 16 Adam/broadcast fit-steps and zero
other fit-steps. Independent theory certifies compatible functions within the
box; that capacity certificate does not imply every bounded optimization
trajectory will find the same internal representation. No unbounded-training,
longer-horizon or broader-rate claim is made.

Validation reconstructed all 8,400 development/fresh clean and noisy endpoints
from saved weights with maximum independent numerical difference 2.22e-16,
reproduced every input/noise/minibatch stream, and verified all protocol,
source, data and selection hashes. All 336 fits and all saved non-timing
checkpoints of one fresh seed were rerun bit-identically. The excluded full
smoke is retained separately. See `validation_report.json` and the independent
Boolean-theory audit for complementary numerical and capacity checks.
