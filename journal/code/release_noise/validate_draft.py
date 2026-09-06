"""Build a clearly marked draft staging archive without touching submission/."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

JOURNAL = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Use a new draft output directory')
    if output == JOURNAL/'submission' or JOURNAL/'submission' in output.parents:
        raise RuntimeError('Draft staging must not touch submission outputs')
    output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('draft_release',JOURNAL/'scripts/build_software_release.py')
    release = importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
    stage = output/'software_release'; stage.mkdir()
    implementation = release.discover_implementation_root(JOURNAL)
    commit = release.run_git('rev-parse','HEAD',repository_root=implementation).strip()
    paper = JOURNAL.parent
    paper_commit = release.run_git('rev-parse','HEAD',repository_root=paper).strip()
    impl = stage/'dendritic_modeling'
    release.extract_git_head(impl,commit,repository_root=implementation,scope='implementation')
    historical_runtime = release.export_physical_runtime(stage, implementation)
    # Preview only: copy explicit current article files so integration can be
    # tested before the parent makes the final scientific commit.
    names = release.run_git('ls-files','--cached','--others','--exclude-standard',repository_root=paper).splitlines()
    names = set(names) | release.required_article_input_paths(JOURNAL)
    selected = sorted({name for name in names if release.repository_file_allowed(Path(name),'paper')})
    for name in selected:
        source = paper/name
        if not source.is_file() or source.is_symlink():
            continue
        target = stage/'journal_package'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    changes = []
    article = stage/'article_analysis'
    counts = release.copy_journal_material(article,source_root=stage/'journal_package/journal')
    release.copy_archived_analysis_scripts(article/'archived_analysis_scripts',paper_root=stage/'journal_package')
    origins = release.capture_release_origins(stage, commit, paper_commit)
    # These article origins are current draft bytes, not committed identities.
    for row in origins.values():
        if row['origin_repository']=='paper':
            row['origin_repository']='uncommitted_draft_paper'; row['origin_commit']=''
    changes.extend(release.prune_release_entrypoints(impl))
    changes.extend(release.prune_release_entrypoints(stage/release.PHYSICAL_RUNTIME_DIRECTORY, release.PHYSICAL_RUNTIME_DIRECTORY))
    for part in ('dendritic_modeling','journal_package','article_analysis',release.PHYSICAL_RUNTIME_DIRECTORY):
        for change in release.sanitize_git_snapshot(stage/part):
            change['path'] = part+'/'+change['path'];changes.append(change)
    release.write_portability_manifest(stage/'PORTABILITY_PATCHES.tsv',changes)
    release.write_released_source_hashes(stage,origins)
    (stage/'README.md').write_text('DRAFT PREVIEW: ARTICLE FILES ARE NOT A COMMITTED RELEASE.\n\n'+release.release_readme(commit,paper_commit))
    metadata = {'physical_depth_historical_runtime':historical_runtime,'draft_only':True,'working_paper_files_included':True,
                'implementation_provenance':release.verify_reachable_commit(implementation,commit),
                'paper_last_commit_not_identity_of_draft':paper_commit,'article_file_counts':counts}
    (stage/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
    findings = release.scan_release(stage)
    (output/'policy_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    if findings:
        raise RuntimeError('Draft policy scan failed; inspect policy_findings.json')
    n = release.write_checksums(stage); assert release.validate_checksums(stage) == n
    archive = output/'DRAFT_software_release.zip'
    release.write_deterministic_zip(stage,archive,1700000000)
    members = release.validate_zip(archive,stage)
    report = dict(status='passed',draft_only=True,archive_bytes=archive.stat().st_size,
                  archive_sha256=release.sha256(archive),zip_members=members,policy_findings=0,
                  implementation_files=len(release.list_files(impl)),paper_files=len(selected))
    (output/'draft_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
