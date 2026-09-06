"""Verify released bytes against an unchanged canonical original digest.

No caller should replace a registered original hash with a released hash. This
helper accepts the original expectation and verifies the explicit release link.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def table(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream,delimiter='\t'))


def safe_child(root, relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe provenance path')
    target=(root/rel).resolve()
    if not target.is_relative_to(root):raise ValueError('Provenance escapes release root')
    return target


def verify_released_file(actual_path, expected_origin_sha256, journal_root=None):
    """Return a structured verdict; never accept an unrelated original digest.

    Sidecars are discovered from the actual file's ancestors. Software uses the
    archive root; restored Source Data uses the journal root. `journal_root` is
    accepted for callers' API clarity, but cannot expand the ancestor boundary.
    """
    path=Path(actual_path).resolve()
    if not path.is_file():return {'verified':False,'reason':'missing file'}
    actual=digest(path)
    if actual==expected_origin_sha256:
        return {'verified':True,'reason':'original bytes','released_sha256':actual}
    failures=[]
    for root in path.parents:
        manifest=root/'RELEASED_SOURCE_HASHES.tsv'
        if not manifest.is_file():continue
        relative=path.relative_to(root).as_posix()
        try:
            matches=[row for row in table(manifest) if row['path']==relative]
            if not matches:continue
            if len(matches)!=1:raise ValueError('duplicate released-source identity')
            record=matches[0]
            if record['origin_sha256']!=expected_origin_sha256:
                raise ValueError('sidecar original hash differs from canonical expectation')
            if record['release_sha256']!=actual:
                raise ValueError('actual bytes differ from released hash')
            provenance=safe_child(root,record['provenance_file'])
            if digest(provenance)!=record['provenance_sha256']:
                raise ValueError('transformation provenance digest differs')
            rows=table(provenance)
            if record['kind']=='software':
                current=expected_origin_sha256
                chain_path=record.get('provenance_path') or relative
                if Path(chain_path).is_absolute() or '..' in Path(chain_path).parts:
                    raise ValueError('Unsafe declared software relocation')
                if chain_path!=relative:
                    relocation=safe_child(root,record['relocation_manifest_file'])
                    if digest(relocation)!=record['relocation_manifest_sha256']:
                        raise ValueError('Relocation source manifest digest differs')
                    links=[r for r in table(relocation) if r['path']==chain_path
                           and r['origin_sha256']==expected_origin_sha256
                           and r['release_sha256']==actual]
                    if len(links)!=1:raise ValueError('Relocation lacks an exact source-identity link')
                chain=[r for r in rows if r['path']==chain_path]
                if not chain:raise ValueError('changed software lacks a declared transformation chain')
                for change in chain:
                    if change['origin_sha256']!=current:
                        raise ValueError('broken transformation hash chain')
                    if not change['reason'].strip():raise ValueError('transformation reason is absent')
                    current=change['release_sha256']
                if current!=actual:raise ValueError('transformation chain does not produce released bytes')
            elif record['kind']=='source_data':
                links=[r for r in rows if r['original_source']==relative
                       and r['original_sha256']==expected_origin_sha256 and r['sha256']==actual
                       and r['transformation']==record['transformation']]
                if not links:raise ValueError('original/released data pair is absent from source manifest')
                description=record['transformation'].strip()
                if not description or description=='byte-identical':
                    raise ValueError('changed data lacks a declared transformation')
                if 'display-specific row filter' in description:
                    raise ValueError('display-filtered data cannot replace a complete canonical source')
            else:raise ValueError('unknown release provenance kind')
            return {'verified':True,'reason':'verified declared release transformation',
                    'released_sha256':actual,'origin_sha256':expected_origin_sha256,
                    'manifest':str(manifest),'transformation':record['transformation']}
        except (OSError,ValueError,KeyError) as error:
            failures.append(str(error))
    return {'verified':False,'reason':'; '.join(failures) or 'hash differs and no applicable release provenance exists',
            'released_sha256':actual}


def write_sidecar(root, rows, replace_kind):
    """Keep software and numerical provenance when both share one journal."""
    manifest=root/'RELEASED_SOURCE_HASHES.tsv'
    retained=[r for r in table(manifest) if r['kind']!=replace_kind] if manifest.exists() else []
    combined=sorted(retained+rows,key=lambda row:row['path'])
    if len({r['path'] for r in combined})!=len(combined):
        raise ValueError('Conflicting software/data provenance for one restored path')
    fields=('path','kind','origin_sha256','release_sha256','transformation',
            'provenance_file','provenance_sha256','provenance_path',
            'relocation_manifest_file','relocation_manifest_sha256',
            'origin_repository','origin_commit','origin_path')
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(combined)


def write_source_data_sidecars(journal, source_data, plan):
    """Preserve source-package provenance beside restored complete data files."""
    import shutil
    journal.mkdir(parents=True,exist_ok=True)
    provenance=journal/'RELEASED_SOURCE_MANIFEST.tsv'
    shutil.copyfile(source_data/'manifest.tsv',provenance)
    provenance_digest=digest(provenance)
    rows=[]
    for source,target,row in plan:
        rows.append(dict(path=target.relative_to(journal).as_posix(),kind='source_data',
            origin_sha256=row['original_sha256'],release_sha256=row['sha256'],
            transformation=row['transformation'],provenance_file=provenance.name,
            provenance_sha256=provenance_digest))
    write_sidecar(journal,rows,'source_data')


def remap_paper_provenance(release_root, journal):
    """Verify copied paper bytes and explicitly relocate their software chain."""
    import shutil
    release_root=Path(release_root).resolve();journal=Path(journal).resolve()
    prefix='journal_package/journal/'
    rows=[]
    relocation_digest=digest(release_root/'RELEASED_SOURCE_HASHES.tsv')
    for record in table(release_root/'RELEASED_SOURCE_HASHES.tsv'):
        if not record['path'].startswith(prefix):continue
        relative=record['path'][len(prefix):]
        source=safe_child(release_root,record['path'])
        destination=safe_child(journal,relative)
        verdict=verify_released_file(source,record['origin_sha256'])
        if not verdict['verified'] or digest(source)!=record['release_sha256']:
            raise ValueError(f'Unverified source archive identity: {record["path"]}')
        if not destination.is_file() or digest(destination)!=record['release_sha256']:
            raise ValueError(f'Copied paper file differs: {relative}')
        # Numerical rows are supplied by the Source Data release; do not
        # duplicate them when that sidecar is merged after restoration.
        if relative.startswith('source_data/'):continue
        relocated=dict(record)
        relocated.update(path=relative,provenance_path=record['path'],
                         provenance_file='SOFTWARE_PORTABILITY_PATCHES.tsv',
                         relocation_manifest_file='SOFTWARE_RELEASED_SOURCE_HASHES.tsv',
                         relocation_manifest_sha256=relocation_digest)
        rows.append(relocated)
    if not rows:raise ValueError('No paper source identities found in software archive')
    journal.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(release_root/'PORTABILITY_PATCHES.tsv',journal/'SOFTWARE_PORTABILITY_PATCHES.tsv')
    # The unchanged archive manifest preserves the explicit original mapping.
    shutil.copyfile(release_root/'RELEASED_SOURCE_HASHES.tsv',journal/'SOFTWARE_RELEASED_SOURCE_HASHES.tsv')
    write_sidecar(journal,rows,'software')
    return len(rows)


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remap-paper',action='store_true',required=True)
    parser.add_argument('--release-root',type=Path,required=True)
    parser.add_argument('--journal-root',type=Path,required=True)
    args=parser.parse_args()
    count=remap_paper_provenance(args.release_root,args.journal_root)
    print(json.dumps({'status':'passed','remapped_software_sources':count},indent=2))
