"""Common-state context/update alignment at all 400 archived selected states.

Positive M[c,c'] denotes a first-order decrease of context c' loss under a
negative delivered-gradient update from c. This is an endpoint diagnostic,
not a reconstruction of training or an experiment in continual learning.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment import SelectionNet, dataset
from model import RULES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flat(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                      for p in parameters]).detach().clone()


def evaluate(item):
    archive, result_path = item
    torch.set_num_threads(1)
    r = json.loads(result_path.read_text())
    path = archive / 'fresh/checkpoints' / (result_path.stem + '.pt')
    assert sha(path) == r['checkpoint_sha256']
    net = SelectionNet(r['seed'], r['forward'], r['variant']).double()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    # The fitted current anchor is a checkpoint buffer; calibration must not
    # overwrite it. This Boolean is deliberately not a state_dict entry.
    net.current_calibrated = True
    x, inh, y = dataset(r['seed'], 'diagnostic', 2048, r['variant'])
    cue = x[:, -4:].argmax(-1)
    variance = y.var(unbiased=False)
    blocks = {name:list(layer.parameters()) for name, layer in
              zip(['terminal', 'proximal', 'soma'], net.core.branch_layers)}
    blocks['readout'] = list(net.readout.parameters())
    vectors = {}
    counts = {}
    for c in range(4):
        mask = cue == c; counts[c] = int(mask.sum())
        assert counts[c] > 350
        for rule in RULES:
            net.gradients(x[mask], inh[mask], y[mask], variance, rule)
            vectors[c, rule] = {block:flat(params) for block, params in blocks.items()}
    rows = []
    for c in range(4):
        for other in range(4):
            for rule in RULES:
                for block in blocks:
                    v = vectors[c, rule][block]; exact = vectors[other, 'exact'][block]
                    vn, gn = float(v.norm()), float(exact.norm())
                    dot = float(v.dot(exact)); valid = vn > 1e-14 and gn > 1e-14
                    rows.append(dict(seed=r['seed'], variant=r['variant'], forward=r['forward'],
                        trained_rule=r['rule'], delivered_rule=rule, block=block,
                        source_context=c, recipient_context=other, n_source=counts[c],
                        n_recipient=counts[other], cosine=dot/(vn*gn) if valid else np.nan,
                        delivered_norm=vn, exact_norm=gn, dot=dot, defined=valid))
    return rows, {str(result_path):sha(result_path), str(path):sha(path)}


def main(args):
    p = json.loads((args.archive/'fresh_protocol.json').read_text())
    for rel, h in p['source_sha256'].items():
        assert sha(args.archive/rel) == h, rel
    files = sorted((args.archive/'fresh/results').glob('*.json'))
    assert len(files) == 400
    args.root.mkdir(parents=True, exist_ok=False)
    # Timestamp and exact question saved before computing any outcomes.
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-review descriptive endpoint analysis of every original selected checkpoint',
        definition='M[c,c_prime] = dot(delivered_gradient(c),exact_gradient(c_prime))/(norm(delivered_gradient(c))*norm(exact_gradient(c_prime)))',
        interpretation='Positive means infinitesimal decrease under a negative gradient update; not an Adam step, learning trajectory, or continual-learning test',
        blocks=['terminal','proximal','soma','readout'], delivered_rules=list(RULES),
        sampling='2048 deterministic diagnostic examples per seed, stratified by their realized four contexts; same examples at all selected states',
        inference='Average the four diagonal or twelve off-diagonal entries within a seed before seed bootstrap; contexts and neurons are not independent replicates',
        zero_norm='Undefined at norm <=1e-14; counts are retained and excluded from cosine means',
        checkpoint_count=400, archive=str(args.archive), archive_protocol_sha256=sha(args.archive/'fresh_protocol.json'), source_sha256=sha(__file__))
    (args.root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(args.workers) as pool:
        output=list(pool.map(evaluate, [(args.archive,p) for p in files]))
    rows = pd.DataFrame([r for part,_ in output for r in part])
    assert len(rows) == 400*4*4*5*4
    rows.to_csv(args.root/'context_alignment_matrices.csv',index=False)
    rows['relation']=np.where(rows.source_context==rows.recipient_context,'within','cross')
    keys=['variant','forward','trained_rule','delivered_rule','block','relation']
    seeds=rows.groupby(keys+['seed'],as_index=False).agg(cosine=('cosine','mean'),defined=('defined','sum'))
    seeds.to_csv(args.root/'context_alignment_seeds.csv',index=False)
    rng=np.random.default_rng(2026092101); summary=[]
    for key,g in seeds.groupby(keys):
        v=g.cosine.dropna().to_numpy()
        bs=v[rng.integers(len(v),size=(10000,len(v)))].mean(1) if len(v) else np.array([np.nan])
        summary.append(dict(zip(keys,key),mean=float(np.mean(v)) if len(v) else np.nan,
            ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),n=len(v)))
    pd.DataFrame(summary).to_csv(args.root/'context_alignment_summary.csv',index=False)
    record=dict(protocol=protocol,inputs={k:v for _,h in output for k,v in h.items()},
                outputs={p.name:sha(p) for p in args.root.glob('*.csv')})
    (args.root/'context_alignment_provenance.json').write_text(json.dumps(record,indent=2))
    print('Checkpoints:',len(files),'entries:',len(rows),'undefined:',int((~rows.defined).sum()))
    print(pd.DataFrame(summary).query('variant=="separable" and forward=="shunt" and trained_rule=="exact" and block=="terminal"').to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    main(p.parse_args())
