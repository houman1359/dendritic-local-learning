# Pinky v185 independent-animal execution record

Completed: 20 August 2026.

## Final execution

- Slurm job: `40620405` (`COMPLETED`, exit `0:0`, elapsed 1:49).
- Account/partition: `kempner_bsabatini_lab` / `kempner_requeue`.
- The scheduler required a one-GPU allocation on this partition; the analysis
  itself was CPU-only and did not import or contact W&B.
- Raw, prepared, cache, log and complete analysis outputs remained under
  `/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820`.
- Only compact source tables, the preparation manifest and publication figure
  were copied into the paper repository.

Frozen scientific hashes for the final job:

- preprocessing: `3d2af1226ec9cbe6fa3b5f31e5cf3cef1dde7874f4fedf678953c40f55d66723`
- Pinky analysis: `7bd16e89c544adc84e3a9479b3a15fcd8a733647a244973ade2fa01debfb18e3`
- common MICrONS routing analysis: `cd1fcbb26d7a96d0a13882fdcfcc0493a2fa30296ccb83a7f26b7efd3373714f`
- cohort manifest: `a9d172712aad884ff83dcb60c7b318e12b3fab4172dc6b20022b631c81a94660`
- Embree package record: `ef420997322455b645bb861bee0df68578d7d13ad16c04d34e298bbdd084ebd3`

After the completed inference, the publication plot was regenerated from the
archived source tables with analysis-script hash
`34ad66faa85005ef76eb0262429ae8302deddd5003fbee4fdc731ad188ba476b`;
the only change from the inference hash was shortening panel C's y-axis label
to avoid clipping in the main-figure compositor.

The Zenodo files passed their archived sizes and MD5 checks, including
`layer23_v185.tar.gz` (`c120366230a2d4ce94f213f60353e0f1`),
`pni_synapses_v185.csv` (`f2382e0606ca72b46062d4c56b15351b`) and
`soma_valence_v185.csv` (`a8ce8aa4e5cdf4202caa5ae10411c333`).

## Outcome-blind implementation corrections

No route-capture endpoint was available when these corrections were made.

1. Job `40610039` selected 212-byte AppleDouble `._*.h5` records. It was
   invalidated before inference; exact mesh basenames are now required.
2. Job `40610685` revealed that the soma-nearest vertex could belong to a
   detached four-vertex fragment. Preprocessing now retains the largest
   face-plus-link-edge component before soma rooting.
3. Smoke job `40612954` showed that the triangle ray backend exceeded the
   96-GB memory request; `40614772` was stopped while validating the same
   issue at 192 GB. Embree smoke job `40617078` completed the unchanged
   opposite-surface radius definition in 32 seconds.
4. Job `40617590` completed all 12 skeletons and synapse derivatives but
   failed on a final display-only `relative_to` call for the project-B
   manifest. Hash-validated derivative reuse was added.
5. Job `40619431` was stopped after the 12-cell analysis when summary code
   attempted an unnecessary exhaustive `2^47` sign flip for the historical
   reference cohort. Exact `2^10` descriptive sign flips are retained for
   Pinky; the older reference p-value is not recomputed.
6. Job `40620093` completed the endpoint and figure but a stale post-run hash
   literal prevented the completion marker. The corrected frozen wrapper then
   completed as job `40620405` with identical results.

Invalid or diagnostic runs are retained under project-B logs and
`invalidated_runs`; none contributes to the reported endpoint.

## Result

All 12 selected cells processed without error; ten passed the frozen direct-E/I
QC gate. At four channels, ancestry-route capture was 0.902, compared with
0.926 for dense PCA, 0.374 for random routes, 0.316 for depth bins and 0.245
for shuffled ancestry. The ancestry capture advantage was positive against
all three controls in the same direction as the independent `minnie65` mouse.
The animal is the replication unit; cell-bootstrap intervals remain
descriptive, and the model-matched field does not demonstrate endogenous
credit use.
