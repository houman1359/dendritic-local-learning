"""Read-only evidence ledger separating historical noise_resilience meanings."""
from pathlib import Path
import hashlib,json,subprocess
import pandas as pd

JOURNAL=Path(__file__).resolve().parents[2]
REPO=JOURNAL.parents[2]
OUT=JOURNAL/'source_data/release_task_identity'
NESTED='4f3612a52756c3691e5bd269c6465c02f240c1e8'
RELATIVE='experiments/data_generation/synthetic_datasets.py'

def digest(payload): return hashlib.sha256(payload).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Audit released frozen sources, not whichever optional nested checkout is
    # currently on a user's import path. Historical execution remains qualified.
    frozen=Path(__file__).resolve().parent/'frozen_sources'
    live=(frozen/'projected_reference_synthetic_datasets.txt').read_bytes()
    committed=(frozen/'legacy_synthetic_datasets.txt').read_bytes()
    assert b'load_noise_resilience_mnist' in live and b'load_noise_resilience_mnist' not in committed
    sources={}
    for relative in ['source_data/clean_exact_bp/summary.json','source_data/clean_exact_bp/run_outcomes.csv',
        'source_data/prospective_learning/seed_outcomes.csv','source_data/prospective_followup/seed_outcomes.csv',
        'source_data/review_curve_lineage/curve_lineage.csv','source_data/review_curve_lineage/s8_fixed_budget_run_lineage.csv',
        'scripts/collect_clean_exact_bp_rerun.py','source_data/prospective_input_validity/historical_run_validity.csv']:
        sources[relative]=digest((JOURNAL/relative).read_bytes())
    clean=pd.read_csv(JOURNAL/'source_data/clean_exact_bp/run_outcomes.csv')
    clean=clean[clean.dataset=='noise_resilience']
    dimensions=clean[(clean.core=='additive') & (clean.depth==1)].copy()
    dimensions['inferred_flattened_inputs']=((dimensions.checkpoint_bytes-7458)/4-2314)/(128*2*2)
    assert (dimensions.inferred_flattened_inputs==784**2).all()
    dimensions[['cohort','config_index','depth','seed','checkpoint_bytes','inferred_flattened_inputs']].to_csv(
        OUT/'clean_noise_input_dimension_check.csv',index=False)
    prospective=pd.read_csv(JOURNAL/'source_data/prospective_learning/seed_outcomes.csv')
    prospective=prospective[prospective.task=='noise_resilience']
    follow=pd.read_csv(JOURNAL/'source_data/prospective_followup/seed_outcomes.csv')
    follow=follow[follow.task=='noise_resilience']
    mapping=[dict(cohort='clean exact-path/backpropagation rerun',source_folder='clean_exact_bp',figures='Supplementary Table S18 and exact/BP source tables',
        n_noise_runs=len(clean),identity='three-class noisy-line images',status='resolved_by_frozen_clean_nested_commit',
        source_commit=NESTED,source_sha256=digest(committed),classes=3,input_shape=[784,784],flattened_inputs=784**2,
        generator='NoisyLineDataset; blank, one horizontal line or one vertical line; iid Gaussian pixel noise SD0.2',
        sample_sizes=[8000,1000,1000],split='torch random_split of combined generated dataset; run seed controls RNG',
        evidence=['summary records root74792ca and nested4f3612a5','collector checks both clean checkouts and frozen source identity',
            'committed dispatcher maps noise_resilience to NoisyLineDataset with default image_size784',
            'saved noise checkpoint sizes ~1.259GB and three-class log likelihood support this identity'],
        input_dimension_check='clean_noise_input_dimension_check.csv: ((checkpoint_bytes - 7458)/4 - 2314)/(128*2*2) = 614656 = 784^2',
        caveat='The newer projected-MNIST parameters in a launch YAML are not interpreted by this older dispatcher.'),
        dict(cohort='prospective depth-by-feedback',source_folder='prospective_learning',figures='depth-by-feedback source tables and SI',
            n_noise_runs=len(prospective),identity='projected-noise MNIST intended; consistent with retained model dimensions and outcomes',
            status='intended_generator_and_structural_evidence; exact_runtime_nested_source_not_pinned',classes=10,flattened_inputs=784,
            intended_noise_sd=1.5,intended_latent_dimension=50,intended_projection_seed=0,intended_split_noise_seeds=[1,2,3],
            intended_clamp=[0,1],evidence=['retained frozen YAML specifies projected-noise parameters',
                'same-depth total parameters match MNIST, ruling out784x784 NoisyLine for these runs',
                'noise accuracy exceeds90%, unlike the clean three-class control'],
            caveat='A live current nested source is not proof of historical executed bytes; do not label the runtime source fully recovered.'),
        dict(cohort='prospective routing, inhibition-dose and fixed-budget follow-ups',source_folder='prospective_followup',
            figures='S8 fixed-budget depth; other prospective follow-up tables',n_noise_runs=len(follow),
            family_counts=follow.family.value_counts().to_dict(),identity='projected-noise MNIST intended; exact runtime generator remains qualified',
            status='intended_generator_and_retained_run_audit; raw_configs_not_all_available',
            evidence=['retained fixed-budget YAMLs specify sigma1.5, latent50 and seeds0/1/2/3',
                'S8 run lineage joins320 fixed-budget outcomes to retained validity audit',
                'resolved_config_available_for_recheck is false in retained S8 lineage; validity is not complete dataset lineage'],
            caveat='Do not infer executed task identity solely from the noise_resilience name or from signed-transfer validity.'),
        dict(cohort='inherited synthetic feedback and gradient cohorts',source_folder='inherited_neurips',
            figures='S1D,E; archived S3D and associated noise diagnostics',identity='unresolved historical noise task',
            status='aggregate_only_no_complete_executed_run_join',
            evidence=['curve_lineage.csv marks execution_lineage_status unresolved for inherited S1/S3 records',
                'available intended configs and related diagnostic run IDs do not identify all plotted executed runs'],
            caveat='Do not retrospectively assign either projected-MNIST or frozen NoisyLine to these plotted aggregates.'),
        dict(cohort='forward-depth and interference screens inherited from original figures',source_folder='inherited_neurips',
            figures='any other legacy panel labeled Noise or noise_resilience',identity='requires panel-specific runtime/config join',
            status='unresolved_unless_separately_linked',
            evidence=['the same dataset key has two demonstrably different implementations'],
            caveat='No global dataset-key-to-generator mapping is scientifically justified.')]
    record=dict(status='qualified_mapping; two_distinct_generator_versions_confirmed',
        legacy_nested_commit=NESTED,legacy_source_sha256=digest(committed),
        current_nested_head=NESTED,
        current_working_source_sha256=digest(live),current_source_differs_from_cited_commit=live!=committed,
        evidence_source_sha256=sources,cohorts=mapping,
        release_policy='Ship separate explicit projected_noise_mnist and legacy_noisy_lines entries; never silently reinterpret noise_resilience without a cohort recipe.')
    (OUT/'task_identity.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[{k:r.get(k,'') for k in ['cohort','source_folder','figures','n_noise_runs','identity','status','caveat']} for r in mapping]
    pd.DataFrame(rows).to_csv(OUT/'task_identity.csv',index=False)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
