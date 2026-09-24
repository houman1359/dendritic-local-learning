from pathlib import Path
import argparse, copy, hashlib, json, os, random, subprocess, sys, time
import numpy as np

R=Path(__file__).resolve().parent
D=R/'cifar_extension'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def run(index,smoke=False):
    protocol=json.loads((D/'protocol.json').read_text());rec=protocol['records'][index]
    runtime=Path(protocol['runtime'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==protocol['runtime_commit']
    assert not subprocess.check_output(['git','diff','--binary','HEAD'],cwd=runtime)
    freeze=json.loads((D/'freeze.json').read_text())
    for name,digest in freeze.items():assert sha(R/name)==digest,name
    config=Path(rec['config']);assert sha(config)==rec['config_sha256']
    out=Path(rec['result_dir'])
    if smoke:
        import yaml
        out=D/'smoke'/f'config_{rec["config_index"]}'
        cfg=yaml.safe_load(config.read_text());cfg['outputs']['results_dir']=str(out)
        cfg['training']['main']['common']['epochs']=2
        config=D/f'smoke_config_{index}.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(runtime/'src'))
    import torch
    torch.set_num_threads(8)
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.scripts.training import train_experiments
    assert Path(train_experiments.__file__).resolve().is_relative_to(runtime)
    assert torch.cuda.is_available() and 'H200' in torch.cuda.get_device_name()
    execution=dict(started_unix=time.time(),index=index,record=rec,smoke=smoke,
        runtime_commit=protocol['runtime_commit'],config_sha256=sha(config),
        protocol_sha256=sha(D/'protocol.json'),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),host=os.environ.get('SLURMD_NODENAME'))
    dump(out/'execution.json',execution)
    original=Trainer._run_epoch;progress=[]
    def observe(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        model=kwargs.get('model',args[0] if args else None);epoch=int(self.epoch_counter)
        progress.append(dict(epoch=epoch,train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),
            best_loss=float(self.best_loss),best_epoch=int(self.best_epoch),patience_counter=int(self.patience_counter)))
        dump(out/'progress.json',progress)
        stop=self.early_stopping and self.patience_counter>=self.patience
        if epoch%100==0 or stop or smoke:
            opt=self.optimizer;wrappers=[]
            while not hasattr(opt,'state_dict'):
                wrappers.append(type(opt).__name__);opt=opt.optimizer
            loaders={}
            for name in ['train_loader','valid_loader']:
                loader=kwargs.get(name)
                if loader is not None and getattr(loader,'generator',None) is not None:
                    loaders[name]=loader.generator.get_state()
            state=dict(epoch=epoch,current_model=copy.deepcopy(self._unwrap_model(model).state_dict()),
                best_model=copy.deepcopy(self.best_state_dict),optimizer=opt.state_dict(),optimizer_wrappers=wrappers,
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                loader_generators=loaders,loss_lists=result,best_epoch=self.best_epoch,best_loss=self.best_loss,
                patience_counter=self.patience_counter,source_commit=protocol['runtime_commit'])
            torch.save(state,out/f'state_{epoch}.pt')
        return result
    Trainer._run_epoch=observe
    try:
        train_experiments.main(str(config))
        summary=json.loads((out/'training_summary.json').read_text())
        metrics=json.loads((out/'performance/final.json').read_text())
        valid=summary['valid_losses'];assert valid and np.isfinite(valid).all()
        original_summary=json.loads((Path(rec['original_results'])/'training_summary.json').read_text())
        n=min(len(valid),len(original_summary['valid_losses']))
        drift=np.asarray(valid[:n])-np.asarray(original_summary['valid_losses'][:n])
        execution.update(status='complete',epochs=len(valid),best_epoch=summary['best_epoch'],
            patience_stopped=len(valid)-int(summary['best_epoch'])>=49,
            reached_cap=len(valid)==(2 if smoke else 1600),prefix_epochs=n,
            max_abs_validation_replay_drift=float(abs(drift).max()),
            final_metrics=metrics,finished_unix=time.time())
    except BaseException as error:
        execution.update(status='execution_failed',error=repr(error),finished_unix=time.time());raise
    finally:
        Trainer._run_epoch=original;dump(out/'execution.json',execution)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.index,a.smoke)
