"""Restore original source paths from the current Source Data manifest.

Display folders (including Methods/retained_evidence) do not determine the
analysis path. Only original_source does. Display-filtered rows are never used
as silent replacements for complete source tables.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def child(root, relative):
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError(f'Unsafe manifest path: {relative}')
    result = (root/name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Manifest path escapes root: {relative}')
    return result


def restoration_plan(source_data, journal):
    with (source_data/'manifest.tsv').open(newline='') as stream:
        reader = csv.DictReader(stream,delimiter='\t')
        required = {'file','original_source','sha256','original_sha256','transformation'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Use the current manifest with original_source and transformation columns')
        rows = list(reader)
    grouped = defaultdict(list)
    for row in rows:
        if not row['original_source'].startswith('source_data/'):
            raise ValueError('Only source_data/ destinations are restored by this helper')
        source = child(source_data,row['file'])
        target = child(journal,row['original_source'])
        if not source.is_file() or digest(source) != row['sha256']:
            raise ValueError(f'Released source checksum mismatch: {row["file"]}')
        grouped[target].append((row,source))
    plan, issues = [], []
    for target, candidates in sorted(grouped.items()):
        full = [(r,p) for r,p in candidates if 'display-specific row filter' not in r['transformation']]
        if not full:
            issues.append({'original_source':str(target.relative_to(journal)),
                           'reason':'Only display-filtered copies are released; complete original cannot be restored'})
            continue
        # A full byte-identical source takes precedence over portable copies.
        original = [(r,p) for r,p in full if r['sha256'] == r['original_sha256']]
        eligible = original or full
        if len({r['sha256'] for r,p in eligible}) != 1:
            issues.append({'original_source':str(target.relative_to(journal)),
                           'reason':'Conflicting complete copies; no arbitrary version selected'})
            continue
        row, source = sorted(eligible,key=lambda item:item[0]['file'])[0]
        if target.exists() and digest(target) != row['sha256']:
            issues.append({'original_source':str(target.relative_to(journal)),
                           'reason':'Existing destination differs; use a fresh restored journal directory'})
            continue
        plan.append((source,target,row))
    return plan,issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data-root',type=Path,required=True)
    parser.add_argument('--journal-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    source_data=args.source_data_root.resolve();journal=args.journal_root.resolve()
    plan,issues=restoration_plan(source_data,journal)
    report={'status':'needs_complete_sources' if issues else 'passed',
            'restorable_original_sources':len(plan),'issues':issues,'dry_run':args.dry_run}
    print(json.dumps(report,indent=2))
    if issues:
        raise SystemExit('Restoration stopped before writing; inspect the manifest issues above')
    if not args.dry_run:
        for source,target,row in plan:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                shutil.copyfile(source,target)
            if digest(target) != row['sha256']:
                raise RuntimeError(f'Restored checksum differs: {target}')


if __name__=='__main__':main()
