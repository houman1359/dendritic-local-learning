"""Prepare matched post-review controls without modifying training implementations."""
from pathlib import Path
import copy,csv,datetime,hashlib,json,subprocess,yaml

R=Path(__file__).resolve().parent
ROOT=Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling')
J=ROOT/'drafts/dendritic-local-learning/journal'
F=Path('/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260827/runtime/dendritic-modeling-cifar-e516c7')
P=Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def merge(a,b):
    out=copy.deepcopy(a)
    for k,v in b.items():out[k]=merge(out[k],v) if k in out and isinstance(out[k],dict) and isinstance(v,dict) else copy.deepcopy(v)
    return out
def inherited(p):
    a=yaml.safe_load(p.read_text());ext=a.pop('extends',None)
    return merge(inherited((p.parent/ext).resolve()),a) if ext else a
def identity(p):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=p)
    assert not diff, f'Frozen source is dirty: {p}'
    return {'path':str(p),'commit':commit,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest()}
def store_config(a,study,key,metadata,source,original):
    a=copy.deepcopy(a);a.pop('_sweep_config_id',None)
    a['outputs']['results_dir']=str(R/'results'/study/key)
    a['outputs']['run_name']=f'{study}_{key}'
    a['outputs']['exact_run_dir']=True
    a['experiment']['record_dataset_fingerprints']=True
    p=R/'configs'/study/f'{key}.yaml';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(yaml.safe_dump(a,sort_keys=False))
    return dict(metadata,key=key,config=str(p),config_sha256=sha(p),result_dir=a['outputs']['results_dir'],source=source,original=original)

def main():
    if (R/'submission.json').exists():raise RuntimeError('Submitted protocol is immutable')
    sources={'fashion':identity(F),'physical':identity(P/'h4-a99c3a7'),'h4d3':identity(P/'h4-d3-repair')}
    jobs=[];records=[]
    for architecture in ['shunting','additive']:
        recipe=J/f'configs/fashion_feedback_ladder/{architecture}.yaml'
        base=inherited(recipe)['base_config'];base['data']['base_dir']=str(R/'data')
        for seed in range(22600,22610):
            group=[]
            for rule,mode in [('strict_scalar','scalar'),('scalar_fallback','per_soma'),('neuron','per_soma_shared'),('exact','path_transport')]:
                a=copy.deepcopy(base)
                # Explicit stream seeds make pairing independent of legacy defaults.
                for k in ['seed','dataset_seed','split_seed','model_seed','topology_seed','loader_seed','evaluation_seed','probe_seed']:a['experiment'][k]=seed
                a['experiment'].update(deterministic=True,strict_deterministic=False,allow_tf32=False,cudnn_benchmark=False,float32_matmul_precision='highest')
                a['training']['main']['learning_strategy_config']['error_broadcast_mode']=mode
                key=f'{architecture}_{seed}_{rule}'
                rec=store_config(a,'fashion',key,dict(architecture=architecture,seed=seed,rule=rule,mode=mode),sources['fashion'],{'recipe':str(recipe),'sha256':sha(recipe)})
                group.append(rec);records.append(rec)
            jobs.append({'study':'fashion','runs':group})
    # Each aligned/reversed pair uses the exact construction source of its
    # archived aligned reference, including the H4 D3 grouping repair.
    for hierarchy,prefix in [(3,'physical_depth_clean_source_replication'),(4,'physical_depth_h4_factorial')]:
        rows=list(csv.DictReader((J/'source_data'/prefix/'seed_outcomes.csv').open()))
        rows=[x for x in rows if x['mechanism']=='raw_additive' and x['architecture']=='serial_tree' and x['regime']=='aligned' and int(x['hierarchy'])==hierarchy]
        assert len(rows)==hierarchy*10,(hierarchy,len(rows))
        for row in sorted(rows,key=lambda x:(int(x['depth']),int(x['seed']))):
            depth=int(row['depth']);seed=int(row['seed'])
            index=int(float(row.get('repair_config_index') or row['config_index']))
            original=Path(row['run_dir'])/'configs'/f'unified_config_{index}.yaml'
            a=yaml.safe_load(original.read_text())
            assert sha(original)==row['config_sha256'],original
            source=sources['h4d3' if hierarchy==4 and depth==3 else 'physical']
            group=[]
            if hierarchy==3:
                control=J/'configs/nonlinear_physical_depth/confirmatory/rewired_tree_shunting_bp.yaml'
            elif depth==3:
                control=J/'configs/physical_depth_h4_d3_repair/h4_d3repair_rewired_tree_serial_shunting_bp.yaml'
            else:
                control=J/'configs/physical_depth_h4_factorial/h4_rewired_tree_serial_shunting_bp.yaml'
            control_base=yaml.safe_load(control.read_text())['base_config']
            rev_paths=control_base['model']['core']['population_network']['layers'][0]['population_defaults']['structured_connectivity']['pathways']
            for placement in ['aligned','reversed']:
                b=copy.deepcopy(a)
                if placement=='reversed':
                    paths=b['model']['core']['population_network']['layers'][0]['population_defaults']['structured_connectivity']['pathways']
                    for pathway in ['ee','ie']:paths[pathway]['feature_ranges']=copy.deepcopy(rev_paths[pathway]['feature_ranges'])
                key=f'h{hierarchy}_d{depth}_{seed}_{placement}'
                rec=store_config(b,'physical',key,dict(hierarchy=hierarchy,depth=depth,seed=seed,placement=placement),source,{'config':str(original),'config_sha256':sha(original),'reference_test_accuracy':float(row['test_accuracy']),'reversal_recipe':str(control),'reversal_recipe_sha256':sha(control)})
                group.append(rec);records.append(rec)
            jobs.append({'study':'physical','runs':group})
    assert len(jobs)==90 and len(records)==220
    for i,job in enumerate(jobs):job['index']=i
    (R/'manifest.json').write_text(json.dumps({'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'jobs':jobs,'n_jobs':90,'n_runs':220,'partition':'kempner_eng','constraint':'h200','max_concurrent':12,'scientific_outcomes_inspected':False},indent=2)+'\n')
    print('Prepared 80 Fashion-MNIST runs and 140 physical-depth runs in 90 paired jobs.')

if __name__=='__main__':main()
