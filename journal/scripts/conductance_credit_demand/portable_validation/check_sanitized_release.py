#!/usr/bin/env python3
"""Excluded partial-release integration fixture; never scientific outcomes."""
import argparse,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];J=HERE.parents[1]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def main(args):
    preview=args.release_root.resolve();fixture=args.fixture_root.resolve();python=args.python.absolute()
    if fixture.exists():raise FileExistsError('Refusing an existing partial fixture directory')
    checksums=table(preview/'SHA256SUMS.tsv')
    for r in checksums:assert digest(preview/r['path'])==r['sha256']
    preview_identity=digest(preview/'SHA256SUMS.tsv')
    launcher='scripts/conductance_credit_demand/portable_run.py'
    assert digest(preview/'journal_package/journal'/launcher)==digest(J/launcher),'Preview launcher needs updating before this check'
    fixture.mkdir(parents=True);copy=fixture/'software_release';shutil.copytree(preview,copy)
    journal=copy/'dendritic_modeling/drafts/dendritic-local-learning/journal'
    shutil.copytree(copy/'journal_package',journal.parent)
    helper=load('_sanitized_conductance_hash_helper',copy/'article_analysis/code/release_noise/release_hashes.py')
    remapped=helper.remap_paper_provenance(copy,journal)
    sys.path.insert(0,str(J/'scripts'))
    import build_nature_source_data as builder
    source_data=fixture/'PARTIAL_Source_Data';source_data.mkdir();rows=[]
    needed=['science_handoff_inventory_20260906.tsv']
    for prefix in ['', 'opponent/']:
        needed += [prefix+f for f in ['development_freeze.json','protocol.json','selection_freeze.json']]
    needed += ['opponent/extension_source_freeze.json']
    def export(original_source,source):
        destination='Methods/'+original_source.removeprefix('source_data/')
        item=builder.SourceFile('Methods','excluded partial-release test',original_source,destination,'portable replay identity','not a scientific cohort','excluded fixture','Partial protocol/config export only; not the complete Source Data release.')
        target=source_data/destination;target.parent.mkdir(parents=True,exist_ok=True)
        builder.copy_source_file(item,source,target);changes=builder.portable_text_copy(target)
        rows.append(dict(figure=item.figure,panels=item.panels,file=destination,role=item.role,independent_unit=item.independent_unit,status=item.status,original_source=original_source,bytes=str(target.stat().st_size),sha256=digest(target),original_sha256=digest(source),transformation='; '.join(changes) or 'byte-identical',notes=item.notes))
    for relative in needed:
        original_source='source_data/conductance_credit_demand/'+relative
        export(original_source,J/original_source)
    # Genuine source-builder prefix sanitization on an explicitly synthetic,
    # non-scientific metadata canary. Conductance protocols already use relative
    # paths, so their actual released bytes need no such transformation.
    canary_original=fixture/'EXCLUDED_original_path_canary.json'
    dump(canary_original,dict(excluded_fixture=True,path=str(J/'source_data/conductance_credit_demand/opponent/protocol.json')))
    canary_relative='source_data/portable_test_only/path_canary.json';export(canary_relative,canary_original)
    builder.write_manifest(source_data,rows);builder.audit_no_machine_local_paths(source_data)
    dump(source_data/'PARTIAL_FIXTURE.json',dict(excluded_from_scientific_results=True,complete_source_data_release=False,scope='Eight canonical conductance replay inputs plus one synthetic metadata path-sanitization canary',source_builder_sha256=digest(J/'scripts/build_nature_source_data.py'),preview_sha256sums=preview_identity))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def command(name,argv,expect=0):
        p=subprocess.run([str(python),*map(str,argv)],env=env,text=True,capture_output=True)
        (fixture/f'{name}.stdout.txt').write_text(p.stdout);(fixture/f'{name}.stderr.txt').write_text(p.stderr)
        if p.returncode!=expect:raise RuntimeError(f'{name}: unexpected return code {p.returncode}: {p.stderr}')
        return p
    command('restore',[copy/'article_analysis/code/release_noise/restore_source_data.py','--source-data-root',source_data,'--journal-root',journal])
    canary=helper.verify_released_file(journal/canary_relative,digest(canary_original))
    assert canary['verified'] and canary['reason']=='verified declared release transformation'
    canonical_results=[]
    for family,seed in [('first',1101),('opponent',2101)]:
        base=[journal/launcher,'--study-root',journal/'source_data/conductance_credit_demand','--journal-root',journal,'--family',family,'--seed',seed]
        p=command(f'{family}_verify',base+['--verify-only']);verdict=json.loads(p.stdout);assert verdict['status']=='passed'
        command(f'{family}_excluded_canary',base+['--output-root',fixture/f'{family}_canary','--excluded-smoke-steps',4])
        audit=json.loads((fixture/f'{family}_canary/portable_audit.json').read_text());assert audit['status']=='passed' and audit['excluded_from_scientific_results']
        canonical_results.append(dict(family=family,n_verified_inputs=len(verdict['verified_inputs']),actual_environment=audit['actual_environment'],steps=audit['steps'],n_excluded_fits=audit['n_fits']))
    # A declared numeric mutation can have a valid byte-provenance chain; the
    # launcher's independent comparison with frozen scientific defaults must
    # still prevent it from entering any fit.
    altered=fixture/'EXCLUDED_numeric_mutation_journal';shutil.copytree(journal,altered)
    relative='source_data/conductance_credit_demand/opponent/protocol.json';target=altered/relative
    original=digest(target);cfg=json.loads(target.read_text());cfg['n_train']+=1;dump(target,cfg)
    manifest=altered/'RELEASED_SOURCE_MANIFEST.tsv';entries=table(manifest)
    for r in entries:
        if r['original_source']==relative:r.update(sha256=digest(target),bytes=str(target.stat().st_size),transformation='excluded fixture numeric mutation; expected scientific-default rejection')
    builder.write_manifest(altered,entries);shutil.copyfile(altered/'manifest.tsv',manifest)
    links=[r for r in table(altered/'RELEASED_SOURCE_HASHES.tsv') if r['kind']=='source_data']
    for r in links:
        r['provenance_sha256']=digest(manifest)
        if r['path']==relative:r.update(release_sha256=digest(target),transformation='excluded fixture numeric mutation; expected scientific-default rejection')
    helper.write_sidecar(altered,links,'source_data')
    declared_numeric=helper.verify_released_file(target,original);assert declared_numeric['verified']
    p=command('declared_numeric_rejection',[altered/launcher,'--study-root',altered/'source_data/conductance_credit_demand','--journal-root',altered,'--family','opponent','--seed',2101,'--verify-only'],expect=1)
    assert 'Released protocol differs from the verified original scientific defaults' in p.stderr
    wrong_origin=helper.verify_released_file(journal/canary_relative,'0'*64);assert not wrong_origin['verified']
    # The real preview is read only throughout the check.
    assert digest(preview/'SHA256SUMS.tsv')==preview_identity
    for r in checksums:assert digest(preview/r['path'])==r['sha256']
    report=dict(status='passed',excluded_from_scientific_results=True,partial_fixture=True,preview=str(preview),preview_sha256sums=preview_identity,n_preview_checksum_entries=len(checksums),preview_unchanged=True,launcher_sha256=digest(journal/launcher),restored_journal=str(journal),partial_source_data=str(source_data),software_remapped_sources=remapped,source_manifest_sha256=digest(source_data/'manifest.tsv'),canonical_checks=canonical_results,path_sanitization_verdict=canary,declared_numeric_byte_chain_verdict=declared_numeric,declared_numeric_scientific_configuration_rejected=True,wrong_original_hash_rejected=True,script_sha256=digest(Path(__file__)))
    dump(fixture/'REPORT.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-root',type=Path,required=True);p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--python',type=Path,required=True);main(p.parse_args())
