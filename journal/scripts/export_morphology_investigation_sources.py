#!/usr/bin/env python3
"""Copy completed investigation sources without modifying frozen originals.

The manifest records source and destination hashes. Existing differing copies
raise an error instead of silently replacing a published scientific input.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis/morphology_investigation_20260905'
MAP={'structure':'morphology_structure','dynamics':'morphology_finite_horizon'}
EXTENSIONS={'.csv','.json','.tsv','.md','.txt'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 for origin,target in MAP.items():
  dest=ROOT/'source_data'/target;dest.mkdir(parents=True,exist_ok=True)
  records=[]
  paths=[p for p in (BASE/origin).rglob('*') if p.is_file() and p.suffix in EXTENSIONS]
  if origin=='structure':
   paths.extend(BASE/'design'/name for name in ['constructive_depth_certificate.csv','design_audit.md','figure_captions.md'])
  for p in sorted(paths):
   relative=p.relative_to(BASE/origin) if p.is_relative_to(BASE/origin) else Path('design')/p.name
   q=dest/relative;q.parent.mkdir(parents=True,exist_ok=True)
   before=sha(p)
   if q.exists() and sha(q)!=before:raise ValueError(f'Existing export differs: {q}')
   if not q.exists():shutil.copyfile(p,q)
   assert sha(q)==before
   records.append(dict(source=str(p.relative_to(ROOT)),destination=str(q.relative_to(ROOT)),sha256=before,bytes=p.stat().st_size))
  protocol={'source_root':str((BASE/origin).relative_to(ROOT)),
   'status':'Completed investigation copied for manuscript integration; frozen original sources retained',
   'independent_unit':'exhaustive finite target family; ALS restarts are algorithmic replicates' if origin=='structure' else '20 independent fresh seed blocks; task conditions and candidate fits nested',
   'files':records}
  (dest/'export_manifest.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
  print(f'{target}: {len(records)} scientific source files verified',flush=True)

if __name__=='__main__':main()
