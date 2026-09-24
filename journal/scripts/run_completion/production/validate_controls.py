"""Verify paired scientific configs and complete run records."""
from pathlib import Path
import copy, hashlib, json, math, yaml

R=Path('/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/additional_figure_controls_20260922')
RESOURCE_KEYS=['total_parameters','trainable_parameters','active_synapses','candidate_synapse_slots','persistent_state_scalars_per_sample']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scientific(a):
    a=copy.deepcopy(a);a.pop('outputs',None);a.pop('_sweep_config_id',None)
    a['experiment'].pop('record_dataset_fingerprints',None)
    return a
def paths(a):return a['model']['core']['population_network']['layers'][0]['population_defaults']['structured_connectivity']['pathways']

def validate_manifest(m):
    assert len(m['jobs'])==90 and sum(len(j['runs']) for j in m['jobs'])==220
    keys=[]
    for i,job in enumerate(m['jobs']):
        assert job['index']==i
        configs=[]
        for rec in job['runs']:
            keys.append(rec['key']);assert sha(rec['config'])==rec['config_sha256']
            a=yaml.safe_load(Path(rec['config']).read_text());configs.append(a)
            assert a['outputs']['exact_run_dir'] is True
            assert a['outputs']['results_dir']==rec['result_dir']
            assert a['experiment']['seed']==rec['seed']
            assert a['training']['main']['common']['epochs']==180
        if job['study']=='fashion':
            assert len(configs)==4
            base=scientific(configs[0]);base['training']['main']['learning_strategy_config'].pop('error_broadcast_mode')
            assert [r['mode'] for r in job['runs']]==['scalar','per_soma','per_soma_shared','path_transport']
            for a,rec in zip(configs,job['runs']):
                b=scientific(a)
                assert b['training']['main']['learning_strategy_config'].pop('error_broadcast_mode')==rec['mode']
                assert b==base
                assert all(a['experiment'][k]==rec['seed'] for k in ['dataset_seed','split_seed','model_seed','topology_seed','loader_seed','evaluation_seed','probe_seed'])
            assert configs[0]['training']['main']['common']['early_stopping'] is False
        else:
            assert len(configs)==2
            aligned,reversed_cfg=map(scientific,configs)
            rec=job['runs'][0];origin=Path(rec['original']['config'])
            assert sha(origin)==rec['original']['config_sha256']
            assert aligned==scientific(yaml.safe_load(origin.read_text()))
            expected=yaml.safe_load(Path(rec['original']['reversal_recipe']).read_text())['base_config']
            for pathway in ['ee','ie']:
                assert paths(reversed_cfg)[pathway]['feature_ranges']==paths(expected)[pathway]['feature_ranges']
                assert paths(reversed_cfg)[pathway]['feature_ranges']!=paths(aligned)[pathway]['feature_ranges']
                paths(reversed_cfg)[pathway]['feature_ranges']=paths(aligned)[pathway]['feature_ranges']
            assert reversed_cfg==aligned
            if rec['hierarchy']==4 and rec['depth']==3:
                assert paths(aligned)['ee']['tier_groups']==[[0],[1],[2,3]]
    assert len(set(keys))==220
    print('Manifest verified: 220 configurations; all scientific pairing checks pass.')

def finite_tree(value):
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,list):return all(finite_tree(v) for v in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return True

def validate_results(m,indices=None,smoke=False):
    indices=range(len(m['jobs'])) if indices is None else indices
    for i in indices:
        datasets=[];seeds=[];resources=[]
        for rec in m['jobs'][i]['runs']:
            out=R/'smoke'/rec['key'] if smoke else Path(rec['result_dir'])
            def read(name):return json.loads((out/name).read_text())
            run=read('execution.json');assert run['exit_code']==0 and run['smoke']==smoke
            assert run['original_config_sha256']==rec['config_sha256']
            assert run['source']==rec['source'] and 'H200' in run['gpu']
            assert finite_tree(read('performance/final.json'))
            train=read('training_summary.json')
            assert finite_tree(train['train_losses']) and finite_tree(train['valid_losses'])
            n=len(train['train_losses']);assert n==len(train['valid_losses']) and n>0
            if smoke:assert n==2
            elif m['jobs'][i]['study']=='fashion':assert n==180
            else:assert n<=180
            config=read('config.json')
            if 'mode' in rec:
                assert config['training']['main']['learning_strategy_config']['error_broadcast_mode']==rec['mode']
            datasets.append(read('dataset_fingerprints.json'));seeds.append(read('resolved_seeds.json'))
            resource=read('model_resources.json');resources.append({k:resource[k] for k in RESOURCE_KEYS})
            if m['jobs'][i]['study']=='physical':
                expected_slots={3:21760,4:16384}[rec['hierarchy']]
                assert resources[-1]==dict(zip(RESOURCE_KEYS,[66178,66178,14336,expected_slots,2944])),resources[-1]
        assert all(d==datasets[0] for d in datasets),('Dataset mismatch',i)
        assert all(d==seeds[0] for d in seeds),('Seed mismatch',i)
        assert all(d==resources[0] for d in resources),('Resource mismatch',i)
    print('Run records verified:', 'smoke' if smoke else 'full',list(indices))

if __name__=='__main__':validate_manifest(json.loads((R/'manifest.json').read_text()))
