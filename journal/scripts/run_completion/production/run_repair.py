"""Continue frozen paper experiments on storage with available file capacity."""
from pathlib import Path
import argparse, copy, hashlib, json, os, random, shutil, subprocess, sys, time
import numpy as np
import yaml

S = Path(__file__).resolve().parent
OLD = Path("/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/run_completion_revision_20260923")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, obj): Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
def rng_state(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
def restore_rng(torch, state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])

def execute(rec, *, study, smoke_name=None, smoke_cap=None, resume_override=None):
    import torch
    source = rec.get("source", {})
    if study == "physical":
        protocol=json.loads((S/"physical/protocol.json").read_text())
        source=dict(path=protocol["runtime"], commit=protocol["runtime_commit"])
    runtime=Path(source["path"])
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=runtime,text=True).strip()==source["commit"]
    assert not subprocess.check_output(["git","diff","--binary","HEAD"],cwd=runtime)
    assert torch.cuda.is_available() and "H200" in torch.cuda.get_device_name()
    torch.set_num_threads(1 if study=="physical" else 8)
    original=Path(rec["config"]); assert sha(original)==rec["config_sha256"]
    cfg=yaml.safe_load(original.read_text())
    out=Path(rec["result_dir"]) if not smoke_name else S/"smoke"/smoke_name
    cap=(rec.get("cap",180) if smoke_cap is None else smoke_cap)
    cfg["outputs"]["results_dir"]=str(out)
    cfg["training"]["main"]["common"]["epochs"]=cap
    out.mkdir(parents=True,exist_ok=False)
    config=out/"executed_config.yaml";config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    resume=resume_override if resume_override is not None else rec.get("resume_checkpoint")
    if resume and resume_override is None: assert sha(resume)==rec["resume_sha256"]
    receipt=dict(status="started",started_unix=time.time(),study=study,record=rec,
        config_sha256=sha(config),ancestor_config_sha256=sha(original),source=source,
        original_config_sha256=sha(original),executed_config_sha256=sha(config),
        smoke=bool(smoke_name),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        host=os.environ.get("SLURMD_NODENAME"),resume_checkpoint=resume,
        resume_sha256=sha(resume) if resume else None,cap=cap)
    dump(out/"execution.json",receipt)
    sys.path.insert(0,str(runtime/"src"))
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.scripts.training import train_experiments
    assert Path(train_experiments.__file__).resolve().is_relative_to(runtime)
    orig_loop=Trainer._run_training_loop; orig_epoch=Trainer._run_epoch
    def loop(self,*args,**kw):
        assert not args
        if resume:
            state=torch.load(resume,map_location="cpu",weights_only=False)
            assert state["source_commit"]==source["commit"]
            model=kw["model"]; device=next(model.parameters()).device
            self._unwrap_model(model).load_state_dict(state["current_model"])
            self.best_state_dict={k:v.to(device) for k,v in state["best_model"].items()}
            self.best_epoch=state["best_epoch"];self.best_loss=state["best_loss"]
            self.epoch_counter=state["epoch"];self.patience_counter=state["patience_counter"]
            opt=self.optimizer
            while not hasattr(opt,"load_state_dict"):opt=opt.optimizer
            opt.load_state_dict(state["optimizer"])
            for name,value in state["loader_generators"].items():kw[name].generator.set_state(value)
            assert set(state["loader_generators"])=={"train_loader","valid_loader"}
            for name,values in zip(["train_losses","valid_losses","train_losses_base","train_losses_reg"],state["loss_lists"]):
                kw[name]=list(values)
            kw["n_epochs"]=cap-state["epoch"];assert kw["n_epochs"]>0
            torch.save(self.best_state_dict,Path(self.save_path)/(self.filename_prefix+"best_model.pt"))
            restore_rng(torch,state["rng"])
            receipt["restored_epoch"]=state["epoch"];dump(out/"execution.json",receipt)
        return orig_loop(self,**kw)
    def epoch(self,*args,**kw):
        result=orig_epoch(self,*args,**kw);n=int(self.epoch_counter);model=kw["model"]
        record=dict(epoch=n,train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),
                    best_epoch=int(self.best_epoch),best_loss=float(self.best_loss),patience_counter=int(self.patience_counter))
        with (out/"progress.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        stopped=self.early_stopping and self.patience_counter>=self.patience
        if n%300==0 or n in (180,600,6000) or n==cap or stopped or smoke_name:
            opt=self.optimizer;wrappers=[]
            while not hasattr(opt,"state_dict"):wrappers.append(type(opt).__name__);opt=opt.optimizer
            generators={name:kw[name].generator.get_state() for name in ("train_loader","valid_loader")}
            state=dict(epoch=n,current_model=copy.deepcopy(self._unwrap_model(model).state_dict()),
                best_model=copy.deepcopy(self.best_state_dict),optimizer=opt.state_dict(),optimizer_wrappers=wrappers,
                rng=rng_state(torch),loader_generators=generators,loss_lists=result,
                best_epoch=self.best_epoch,best_loss=self.best_loss,patience_counter=self.patience_counter,source_commit=source["commit"])
            torch.save(state,out/f"state_{n}.pt")
            if study=="physical" and n in (180,600,6000):
                training=model.training
                try:
                    self._unwrap_model(model).load_state_dict(self.best_state_dict)
                    self.analysis_manager.run_analysis(filename=f"budget{n}_best",training=False)
                finally:
                    self._unwrap_model(model).load_state_dict(state["current_model"]);model.train(training)
                    restore_rng(torch,state["rng"])
                    for name,value in generators.items():kw[name].generator.set_state(value)
        return result
    Trainer._run_training_loop=loop;Trainer._run_epoch=epoch
    try:
        if resume and not smoke_name:
            for budget in (180,600):
                src=Path(rec["prior_results"])/"performance"/f"budget{budget}_best.json"
                if src.exists():
                    (out/"performance").mkdir(exist_ok=True);shutil.copy2(src,out/"performance"/src.name)
        train_experiments.main(str(config))
        summary=json.loads((out/"training_summary.json").read_text())
        n=len(summary["valid_losses"]);assert n and np.isfinite(summary["valid_losses"]).all()
        receipt.update(status="complete",exit_code=0,epochs=n,best_epoch=summary["best_epoch"],
            ordinary_stopping_reached=n-summary["best_epoch"]>=30 and n<cap,
            reached_cap=n==cap,finished_unix=time.time())
    except BaseException as error:
        receipt.update(status="execution_failed",exit_code=1,error=repr(error),finished_unix=time.time());raise
    finally:
        Trainer._run_training_loop=orig_loop;Trainer._run_epoch=orig_epoch
        dump(out/"execution.json",receipt)

def main():
    p=argparse.ArgumentParser();p.add_argument("--study",choices=["physical","controls"],required=True)
    p.add_argument("--index",type=int,required=True);p.add_argument("--smoke-name");p.add_argument("--smoke-cap",type=int)
    p.add_argument("--resume-override")
    a=p.parse_args()
    for name,digest in json.loads((S/"training_freeze.json").read_text()).items():assert sha(S/name)==digest,name
    protocol=json.loads((S/a.study/"protocol.json").read_text())
    records=[protocol["records"][a.index]] if a.study=="physical" else protocol["jobs"][a.index]["runs"]
    for rec in records:execute(rec,study=a.study,smoke_name=a.smoke_name,smoke_cap=a.smoke_cap,resume_override=a.resume_override)
if __name__=="__main__":main()

