#!/usr/bin/env python3
"""Read-only identity checks for the completed four-bundle journal release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def verify_zip(archive, stage, prefix):
    expected = {str(Path(prefix) / p.relative_to(stage)): p
                for p in stage.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, archive
        names = z.namelist()
        assert len(names) == len(set(names)), archive
        assert set(names) == set(expected), archive
        for name, path in expected.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest(path), name
    observed = digest(archive)
    assert archive.with_suffix('.zip.sha256').read_text().split()[0] == observed
    return dict(members=len(expected), bytes=archive.stat().st_size, sha256=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--submission-root', type=Path,
                        help='Optional separately extracted, immutable bundle fixture.')
    args = parser.parse_args()
    j = args.journal.resolve()
    sys.path.insert(0, str(j / 'scripts'))
    import build_submission_bundle as bundle
    import build_software_release as software
    from tex_sources import tex_sources

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=j, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=j, text=True).strip()
    sub = args.submission_root.resolve() if args.submission_root else j / 'submission'
    stages = {
        'software': ('Dendritic_credit_assignment_software.zip', 'software_release', 'software_release'),
        'source_data': ('Source_Data.zip', 'nature_source_data', 'Source_Data'),
        'overleaf': ('Overleaf_Project.zip', 'overleaf_project', ''),
        'submission': ('Nature_Communications_Submission.zip', 'nature_communications_bundle', 'nature_communications_bundle'),
    }
    report = dict(status='passed', paper_commit=commit, archives={})
    for key, (archive, stage, prefix) in stages.items():
        report['archives'][key] = verify_zip(sub / archive, sub / stage, prefix)

    sw = sub / 'software_release'
    assert software.scan_release(sw) == []
    report['software_payload_checksums'] = software.validate_checksums(sw)
    meta = json.loads((sw / 'METADATA.json').read_text())
    assert meta['journal_repository_commit'] == commit
    assert meta['journal_repository_clean_at_build'] is True
    assert meta['paper_provenance']['reachable_from_refs']
    assert meta['implementation_provenance']['reachable_from_refs']
    assert not meta.get('draft_only', False)
    report['implementation_commit'] = meta['implementation_repository_commit']

    for name in ['nature_source_data', 'overleaf_project']:
        version = json.loads((sub / name / 'SOURCE_VERSION.json').read_text())
        assert version['paper_commit'] == commit and not version['tracked_source_dirty']
    upload = sub / 'nature_communications_bundle'
    meta = json.loads((upload / 'manifests/bundle_metadata.json').read_text())
    assert meta['source_commit'] == commit and meta['source_worktree_clean']
    assert meta['numbered_main_figure_count'] == len(bundle.MAIN_FIGURES)
    assert meta['supplementary_figure_count'] == len(bundle.SUPPLEMENTARY_FIGURES)
    for row in table(upload / 'manifests/bundle_manifest.tsv'):
        path = upload / row['bundle_path']
        assert path.stat().st_size == int(row['bytes']) and digest(path) == row['sha256']
    assert digest(upload / 'Source_Data.zip') == digest(sub / 'Source_Data.zip')
    assert digest(upload / 'Software.zip') == digest(sub / 'Dendritic_credit_assignment_software.zip')

    sources = table(sub / 'nature_source_data/manifest.tsv')
    for row in sources:
        path = sub / 'nature_source_data' / row['file']
        assert path.stat().st_size == int(row['bytes']) and digest(path) == row['sha256']
        assert digest(j / row['original_source']) == row['original_sha256']
        if row['sha256'] != row['original_sha256']:
            assert row['transformation'] and row['transformation'] != 'byte-identical'
    report['source_data_rows'] = len(sources)

    manuscript = set(tex_sources(j / 'main.tex')) | set(tex_sources(j / 'supplementary/supplementary.tex'))
    manuscript.add(j / 'references.bib')
    manuscript.update(j / 'figures' / name for name in bundle.FIGURES)
    for original in manuscript:
        relative = original.relative_to(j)
        for base in [sw / 'journal_package/journal', sub / 'overleaf_project', upload]:
            assert digest(base / relative) == digest(original), (base, relative)
    for relative in ['main.pdf', 'main_with_supplementary.pdf', 'supplementary/supplementary.pdf']:
        assert digest(upload / relative) == digest(j / relative)
    report['identical_manuscript_and_figure_files_per_bundle'] = len(manuscript)
    report['main_figures'] = len(bundle.MAIN_FIGURES)
    report['supplementary_figures'] = len(bundle.SUPPLEMENTARY_FIGURES)
    report['scope'] = 'Archive membership/CRCs/checksums, committed source identities, numerical original/released links, figure and manuscript equality. Restored execution and clean compilation are recorded separately.'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
