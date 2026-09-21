"""The installable implementation and the nested paper are distinct exports."""

from __future__ import annotations

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import pytest


JOURNAL = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_software_repositories", JOURNAL / "scripts/build_software_release.py"
)
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def repository(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "add", ".")
    git(root, "-c", "user.name=Packaging fixture", "-c",
        "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def test_archive_ships_production_package_and_separate_committed_paper(tmp_path: Path) -> None:
    production = tmp_path / "production"
    branch = "src/dendritic_modeling/networks/architectures/excitation_inhibition/dendritic/branch_dynamics.py"
    package_source = "def forward_branch_dynamics():\n    return 'production'\n"
    production_commit = repository(production, {
        "pyproject.toml": '[project]\nname = "dendritic_modeling"\nversion = "0.1.0"\n',
        "LICENSE": "fixture license\n",
        "src/dendritic_modeling/__init__.py": "",
        branch: package_source,
    })
    paper = production / "drafts/dendritic-local-learning"
    paper_commit = repository(paper, {"journal/main.tex": "Committed manuscript\n"})
    assert release.discover_implementation_root(paper / "journal") == production

    # A committed export must neither absorb nor modify unrelated user work.
    (production / branch).write_text("Uncommitted source change\n")
    (production / "untracked_private.txt").write_text("not part of the release\n")
    before = git(production, "status", "--porcelain=v1", "--untracked-files=all")
    stage = tmp_path / "software_release"
    counts = release.export_repository_snapshots(
        stage, production, production_commit, paper_commit, paper_root=paper
    )
    assert counts == {"implementation": 4, "paper": 1}
    assert git(production, "status", "--porcelain=v1", "--untracked-files=all") == before
    assert (production / branch).read_text() == "Uncommitted source change\n"
    assert not (stage / "dendritic_modeling/untracked_private.txt").exists()
    archive = tmp_path / "fixture.zip"
    release.write_checksums(stage)
    release.write_deterministic_zip(stage, archive, 1_700_000_000)
    release.validate_checksums(stage)
    release.validate_zip(archive, stage)
    with zipfile.ZipFile(archive) as saved:
        prefix = "software_release/"
        assert saved.read(prefix + "dendritic_modeling/pyproject.toml").startswith(b"[project]")
        assert saved.read(prefix + "dendritic_modeling/" + branch).decode() == package_source
        assert saved.read(prefix + "journal_package/journal/main.tex") == b"Committed manuscript\n"


def test_isolated_paper_requires_explicit_real_implementation_root(tmp_path: Path) -> None:
    paper = tmp_path / "isolated_paper/journal"
    paper.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="--implementation-root"):
        release.discover_implementation_root(paper)
    production = tmp_path / "implementation"
    repository(production, {
        "pyproject.toml": "[build-system]\n",
        "src/dendritic_modeling/__init__.py": "",
    })
    assert release.discover_implementation_root(paper, production) == production


def test_portability_manifest_records_original_and_release_bytes(tmp_path: Path, monkeypatch) -> None:
    # Released defaults may already be portable; exercise a real substitution
    # independently of whichever machine-specific defaults were distributed.
    original, replacement = "/fixture/source", "${FIXTURE_ROOT}"
    monkeypatch.setattr(release, "PORTABILITY_REPLACEMENTS",
                        ((original, replacement, "fixture relocation"),))
    path = tmp_path / "config.txt"
    path.write_text(original + "/artifact\n")
    old_hash = release.sha256(path)
    changes = release.sanitize_git_snapshot(tmp_path)
    assert path.read_text() == replacement + "/artifact\n"
    assert changes[0]["origin_sha256"] == old_hash
    assert changes[0]["release_sha256"] == release.sha256(path)


def test_allowlist_omits_presentations_internal_logs_and_other_project_drivers():
    for path in ['journal/LLR_n.pptx', 'journal/analysis/WRITING_REVISION_20260905.md',
                 'journal/analysis/PANEL_REVIEW_20260906.md', 'journal/submission/internal_notes.md']:
        assert not release.repository_file_allowed(Path(path), 'paper')
    for path in ['src/dendritic_modeling/scripts/text/train.py',
                 'src/dendritic_modeling/scripts/transformer_replacement/run.py',
                 'configs/transformer_replacement/big_model.yaml', 'docs/compression.md']:
        assert not release.repository_file_allowed(Path(path), 'implementation')
    assert release.repository_file_allowed(Path('journal/scripts/credit_rule_bridge/models.py'), 'paper')
    assert release.repository_file_allowed(Path('journal/code/release_noise/frozen_generators.py'), 'paper')
    assert release.repository_file_allowed(Path('src/dendritic_modeling/networks/architectures/replacement/__init__.py'), 'implementation')


def test_provenance_rejects_unreachable_or_missing_commit(tmp_path):
    root = tmp_path / 'repo'
    first = repository(root, {'a.txt':'one'})
    assert release.verify_reachable_commit(root, first)['object_verified']
    git(root, 'checkout', '--detach', '-q')
    (root/'a.txt').write_text('two')
    git(root,'add','.')
    git(root,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','detached')
    orphan = git(root,'rev-parse','HEAD')
    with pytest.raises(RuntimeError,match='not reachable'):
        release.verify_reachable_commit(root, orphan)
    with pytest.raises(subprocess.CalledProcessError):
        release.verify_reachable_commit(root, '0'*40)


def test_pruning_removes_only_missing_driver_entrypoints(tmp_path):
    (tmp_path/'pyproject.toml').write_text('[project.scripts]\nkeep = "pkg.run:main"\ndrop = "pkg.missing:main"\n[other]\nvalue = 3\n')
    p = tmp_path/'src/pkg/run.py';p.parent.mkdir(parents=True);p.write_text('def main(): pass')
    patches = release.prune_release_entrypoints(tmp_path)
    assert 'keep =' in (tmp_path/'pyproject.toml').read_text()
    assert 'drop =' not in (tmp_path/'pyproject.toml').read_text()
    assert patches[0]['replacement_count'] == 1


def test_ignored_allowlisted_recipe_cannot_silently_disappear(tmp_path):
    root = tmp_path/'paper'
    commit = repository(root, {'journal/main.tex':'paper', '.gitignore':'journal/configs/*.yaml\n'})
    config = root/'journal/configs/new.yaml';config.parent.mkdir(parents=True);config.write_text('seed: 42\n')
    assert 'journal/configs/new.yaml' in release.required_article_input_paths(root/'journal')
    with pytest.raises(RuntimeError,match='Git-ignored recipes'):
        release.assert_article_inputs_committed(root/'journal',root,commit)



def test_registered_generator_and_protocol_cannot_be_silently_omitted(tmp_path):
    import csv
    journal = tmp_path / "journal"
    manifest = journal / "source_data/provenance_manifest.tsv"
    manifest.parent.mkdir(parents=True)
    fields = ["status", "generator_path", "source_path"]
    def write(generator, source=""):
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(dict(status="ready", generator_path=generator, source_path=source))
    prefix = "drafts/dendritic-local-learning/journal/"
    write(prefix + "scripts/not_registered_for_release.py")
    with pytest.raises(RuntimeError, match="omitted software inputs"):
        release.assert_registered_sources_allowlisted(journal)
    write(prefix + "scripts/collect_mnist_feedback_ladder.py",
          prefix + "analysis/CIFAR10_ADDITIVE_FEEDBACK_LADDER_CONFIRMATORY_20260828.md")
    release.assert_registered_sources_allowlisted(journal)
    write(prefix + "scripts/collect_mnist_feedback_ladder.py",
          prefix + "analysis/unregistered_protocol.md")
    with pytest.raises(RuntimeError, match="unregistered_protocol"):
        release.assert_registered_sources_allowlisted(journal)
