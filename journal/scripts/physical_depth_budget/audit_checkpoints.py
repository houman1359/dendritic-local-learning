#!/usr/bin/env python3
"""Inspect every original state file used by the budget extension."""
import json
from pathlib import Path
import torch
from extend import OUT,sha,dump


def main():
    protocol=json.loads((OUT/'extension_protocol.json').read_text());rows=[]
    for record in protocol['conditions']:
        path=Path(record['original_final_model']);state=torch.load(path,map_location='cpu',weights_only=True)
        keys=list(state)
        tensor_only=all(torch.is_tensor(v) for v in state.values())
        assert tensor_only
        rows.append(dict(index=record['index'],arm=record['arm'],depth=record['depth'],seed=record['seed'],
            path=str(path),sha256=sha(path),state_entries=len(keys),all_values_tensors=True,
            optimizer_state_available=False,random_generator_state_available=False,full_resume_possible=False))
    dump(OUT/'original_checkpoint_inventory.json',dict(status='passed',files=len(rows),records=rows,
        conclusion='All60originalfinalstates contain model tensors only. These are not full optimizer/RNG checkpoints; extension restarts original seeds.',script_sha256=sha(__file__)))


if __name__=='__main__':main()
