from pathlib import Path
import csv,json,subprocess,sys
J=Path(__file__).resolve().parents[2];paper=J.parent
sys.path.insert(0,str(J/'scripts'))
import build_software_release as software
paths=set(software.required_article_input_paths(J))
for args in [['diff','--name-only'],['ls-files','--others','--exclude-standard']]:
 paths.update(subprocess.check_output(['git',*args],cwd=paper,text=True).splitlines())
with (J/'source_data/provenance_manifest.tsv').open() as handle:
 for row in csv.DictReader(handle,delimiter='\t'):
  prefix='drafts/dendritic-local-learning/'
  if row['source_path'].startswith(prefix):paths.add(row['source_path'].removeprefix(prefix))
layout=json.loads((J/'configs/credit_first_provenance/panel_sources.json').read_text())
for row in layout['assets']:paths.add('journal/'+row['component'])
for row in layout['source_hash_manifests']:paths.add('journal/'+row['path'])
for inv in ['source_data/image_ladder_controls/final_owner_inventory.tsv','source_data/conductance_credit_demand/study_source_inventory.tsv']:
 with (J/inv).open() as handle:
  for row in csv.DictReader(handle,delimiter='\t'):
   candidate=Path(row['path'])
   if candidate.suffix not in {'.log','.out','.err','.pt','.zip','.pyc'} and '__pycache__' not in candidate.parts:
    paths.add('journal/'+row['path'])
 paths.add('journal/'+inv)
for pattern in ['*.json','*.md','verify_final_archives.py','prepare_commit_inputs.py']:
 for p in (J/'analysis/review_followup_20260906').glob(pattern):paths.add(str(p.relative_to(paper)))
for p in (J/'analysis/review_followup_20260906/mnist_complete_independent_audit').glob('REPORT.*'):paths.add(str(p.relative_to(paper)))
for p in (J/'analysis/repository_alignment_20260906').glob('alignment.json'):paths.add(str(p.relative_to(paper)))
# Generated local archives, raw model runs and private fixtures are never staged.
assert all(p.startswith(('journal/','neurips/scripts/')) for p in paths)
for p in paths:
 assert not any(x in Path(p).parts for x in ['__pycache__','draft_release_portability_v3','legacy_archive_fixture'])
 assert Path(p).suffix not in ['.pt','.zip','.pyc','.log'],p
 assert (paper/p).is_file(),p
out=J/'analysis/review_followup_20260906/commit_inputs.txt'
out.write_text('\n'.join(sorted(paths))+'\n')
print(json.dumps({'candidate_files':len(paths),'bytes':sum((paper/p).stat().st_size for p in paths),'paths_file':str(out)},indent=2))
if '--stage' in sys.argv:subprocess.run(['git','add','-f','--pathspec-from-file',str(out)],cwd=paper,check=True)
