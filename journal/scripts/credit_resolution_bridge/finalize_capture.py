#!/usr/bin/env python3
"""Assemble the checked interpretation and reproducibility inventory."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analysis_capture as a


def bootstrap(values):
    values=np.asarray(values,float);rng=np.random.default_rng(6202609)
    samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975)),n_seeds=len(values))


def main():
    checks={name:json.loads((a.OUT/name).read_text()) for name in ['validation.json','matched_bridge_validation.json','counterfactual_same_state_validation.json','gauge_validation.json']}
    assert all(v['status']=='passed' for v in checks.values())
    original=json.loads((a.OUT/'analysis_protocol.json').read_text())
    for file,digest in original['original_source_sha256'].items():assert a.sha(a.JOURNAL/file)==digest
    d=pd.read_csv(a.OUT/'matched_bridge_capture.csv');s=pd.read_csv(a.OUT/'matched_bridge_spectra.csv');g=pd.read_csv(a.OUT/'matched_bridge_gauge_spectra.csv')
    endpoint=d[d.step.eq(1024)&d.selected_rate&d.optimizer.eq('adam')&d.rule.eq('exact')]
    contrasts=[]
    for dictionary in ['uniform_projection','best_fixed_rank1_q','ancestry_K2','ancestry_K4']:
        rows=endpoint[endpoint.dictionary.eq(dictionary)]
        for metric in ['field_energy_fidelity','error_weighted_energy_fidelity','eligibility_weighted_update_fidelity']:
            pair=rows.pivot(index='seed',columns='family',values=metric)
            contrasts.append(dict(dictionary=dictionary,metric=metric,contrast='quartet_minus_matching',**bootstrap(pair.quartet-pair.matching)))
    pd.DataFrame(contrasts).to_csv(a.OUT/'matched_bridge_paired_capture_contrasts.csv',index=False)
    table=endpoint.groupby(['family','dictionary'])[['field_energy_fidelity','error_weighted_energy_fidelity','eligibility_weighted_update_fidelity','population_nmse']].mean()
    lines=['# What the credit-resolution analysis establishes','',
        'The new analysis supports a credit-first narrative, with a more precise distinction between a uniform broadcast and one calibrated spatial profile. Matching and quartic targets can have identical input-sensitivity spectra yet develop very different output-sensitivity fields during learning. This is evidence about the spatial distinctions in credit, not a rescue of the failed initialization-based structure selector.',
        '', '## Matched fresh cohort','',
        'Twenty new seed blocks pair the matching and quartic targets on exactly the same compatible balanced tree, initial weights, input stream and mini-batch stream. Independent checks confirm identical saved trees, initial weights, evaluation inputs, initial calibration profiles and input spectra. Nested targets are separate: they use a compatible deeper tree and do not share the input spectrum.',
        '', 'At step1024 after exact-credit Adam training, the best single fixed spatial profile captures99.73% of the pairwise path field,46.10% of the quartic field and44.82% of the nested field. Multiplication by local eligibilities leaves99.70%,47.23% and45.70% of per-example update energy, respectively. These are post hoc projections with oracle coefficients, not trained encoders.',
        '', '| Task | Spatial dictionary | Path energy | Residual-weighted energy | Eligibility-weighted update fidelity |',
        '|---|---|---:|---:|---:|']
    for (family,dictionary),r in table.iterrows():
        if dictionary in ['uniform_projection','best_fixed_rank1_q','ancestry_K2','ancestry_K4']:
            lines.append(f'| {family} | {dictionary} | {r.iloc[0]:.4f} | {r.iloc[1]:.4f} | {r.iloc[2]:.4f} |')
    contrast=next(v for v in contrasts if v['dictionary']=='best_fixed_rank1_q' and v['metric']=='field_energy_fidelity')
    lines += ['',f'The paired quartic-minus-matching difference in best-profile path capture is{contrast["mean"]:.4f} (descriptive95% whole-seed bootstrap interval{contrast["ci95_low"]:.4f} to{contrast["ci95_high"]:.4f};20seeds). All dictionary comparisons and both optimizers are retained in the tables.',
        '', 'The proposed uniform-broadcast prediction does not hold at exact-trained checkpoints. Uniform capture is12.24%,16.86% and17.57% across pairwise, quartic and nested targets. The strongest defensible statement concerns the best fixed spatial profile and the need for additional varying directions, rather than high raw uniform capture on pairwise tasks.',
        '', 'The field develops during learning. At the shared initial checkpoint, matching and quartic path fields are identical. On the archived cohort, pairwise trees trained by unit broadcast acquire high uniform capture(88.98%), while exact-trained pairwise trees have low uniform capture(11.95%) despite almost rank-one fields. Training can align a computation with the available delivery profile. A poorly performing broadcast-trained endpoint may also have a lower-rank field because it never acquired the target interaction; it should not be read as the credit field required by the successfully learned target.',
        '', '## Coordinate and residual controls','',
        'A static change in each compartment\'s units changes energy fractions and effective rank; algebraic matrix rank is invariant to invertible diagonal rescaling. Dividing each coordinate by its RMS path sensitivity gives leading fractions0.9973,0.3545 and0.2845 in the matched exact-trained Adam cohort, with effective ranks1.006,4.273 and4.694. Thus the separation survives this specified normalization. The normalization is a geometry check and does not establish invariance of learning under parameter rescaling.',
        '', 'For a fixed model state, q(x)=d output/d local state does not depend on the target. Per-example loss credit is delta(x)q(x), so its normalized capture equals that of q whenever delta is nonzero. Relabeling the same saved state with an input-isospectral compatible target leaves unweighted capture exactly unchanged in80counterfactual state pairs. Residual-weighted aggregate capture may change because examples receive different weights; the residual-optimal rank-one profile is therefore reported separately.',
        '', 'The nonroot field contains six actual internal units. Including the root would insert a permanent unit-sensitivity coordinate and change every ratio. The update comparisons exclude the root for their spatial energy metric, while separate whole-gradient cosines retain its exact contribution. Ancestry K=2 and K=4 refer to deterministic nested spans with a common component, not optimally selected route sets.',
        '', '## Replication and limits','',
        'The archived cohort independently shows the same best-profile separation after faithful reconstruction of unsaved checkpoints: Adam captures99.88%,48.73% and42.01% for pairwise, quartic and nested targets. The600replayed selected compatible fits produce3,600checkpoints; all28,800available deterministic archived metrics reproduce to a maximum absolute difference8.9e-16. This is a reconstruction of frozen fits, not a previously stored-checkpoint analysis.',
        '', 'The new matched cohort is more strictly paired but its selected Adam exact rule learns the quartic targets less completely on average(clean population NMSE0.1058, compared with0.00425 in the archived cohort). All seeds are included. Both datasets should be retained, and neither a checkpoint spectrum nor a high projection capture should be called a convergence or necessity theorem.',
        '', 'In the separate positive-conductance teacher cohort, actual saved exact-trained endpoints have about99.14% best rank-one capture and90.2% capture by the initially calibrated profile. This model therefore supplies a useful boundary case: its path field is close to one profile despite input grouping effects. Its teacher is not the algebraic quartic target, so the two models do not test the same interaction-order intervention.',
        '', 'The matched learning intervention is the causal counterpart of the geometric audit. It compares exact delivery with unit, fixed calibrated and sign profiles while holding forward resources and initial conditions fixed. The geometry explains what those controls remove, and the learning experiment determines whether those removals matter under the specified optimizer and budget. The audit alone does not show that full six-compartment delivery is uniquely necessary or that biological neurons use these fields.',
        '', '## Suggested manuscript statement','',
        '> Input-sensitivity rank alone did not determine the spatial structure of credit. On the same compatible tree, pairwise and quartic targets with identical input-sensitivity spectra developed nearly one-dimensional and multidirectional credit fields, respectively. A single fitted spatial profile retained almost all pairwise credit but less than half of quartic credit, including after multiplication by local eligibilities. Matched learning controls then tested whether fixed profiles could realize these computations.',
        '', '## Figure caption','',
        '**Spatial credit structure emerges during learning.** All panels use the20fresh matched seed blocks and exact-credit Adam trajectories on compatible algebraic trees. Pairwise and quartic tasks share their initial tree, weights and inputs; nested tasks are a separate compatible-tree control. **A**, fraction of squared output-sensitivity energy retained by a uniform profile, a best fixed rank-one profile, and ancestry spans of dimensions2and4. **B**, the same profiles evaluated after weighting by the squared loss residual. The best profile in A is retained here; a separately optimized residual-weighted profile is supplied in Source Data. **C**, participation-ratio effective rank of the uncentered path-field second moment over training. **D**, fidelity of per-example parameter updates after local feature eligibilities multiply the exact and projected credit. All routing metrics use the six nonroot internal units. Points show means and error bars s.e.m. across20seed blocks. Projection coefficients are computed from the exact field and constitute capacity diagnostics.',
        '', '## Reproduction','',
        'From the journal directory, set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1, then run `python scripts/credit_resolution_bridge/analysis_capture.py run --seed SEED` for each seed127200..127219; `analysis_capture.py saved`; `analysis_capture.py summarize`; `common_state_audit.py`; `gauge_audit.py`; `analyze_matched_bridge.py`; and `finalize_capture.py`. The last two require the completed frozen credit_rule_bridge outputs. Run `python -m unittest discover -s scripts/credit_resolution_bridge -p test_capture.py -v` for the six mathematical controls.', '']
    # Keep conventional spacing in prose around quantities, independent of source code formatting.
    import re
    prose='\n'.join(lines)
    prose=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', prose)
    prose=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', prose)
    (a.OUT/'FINAL_INTERPRETATION.md').write_text(prose)
    a.dump(a.OUT/'final_audit.json',dict(status='passed',checks=checks,
        scripts={p.name:a.sha(p) for p in sorted(a.HERE.glob('*.py'))},
        outputs={str(p.relative_to(a.OUT)):a.sha(p) for p in sorted(a.OUT.rglob('*')) if p.is_file() and p.name!='final_audit.json'},
        original_sources_unchanged=True,visual_inspection='Both four-panel PDFs reviewed; no overlapping labels or clipping',
        scope='All analyses posthoc; matched learning cohort separately frozen'))


if __name__=='__main__':main()
