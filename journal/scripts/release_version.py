"""Record the real paper revision and tracked source state for generated bundles."""
from __future__ import annotations
import subprocess
from pathlib import Path

def source_version(journal: Path) -> dict:
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=journal, text=True).strip()
    commit = git('rev-parse', '--verify', 'HEAD')
    git('cat-file', '-e', commit + '^{commit}')
    status = git('status', '--porcelain', '--untracked-files=no')
    return {
        'paper_commit': commit,
        'paper_commit_utc': git('show', '-s', '--format=%cI', commit),
        'tracked_source_dirty': bool(status),
        'provenance_scope': 'Actual source revision; per-file archive manifests identify the exported bytes.',
    }
