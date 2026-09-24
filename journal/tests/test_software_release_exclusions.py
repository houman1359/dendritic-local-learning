"""The frozen CIFAR execution records must never enter the software release.

These frozen launch YAMLs document completed cluster runs with site-specific
paths (the confirmatory one is byte-pinned by its analyzer), so both the
reproducibility private-path rule and the release copier must treat them as
provenance records, not shippable recipes — and the two policies must agree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

JOURNAL = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_release_excludes_frozen_execution_records() -> None:
    release = _load("_release", JOURNAL / "scripts" / "build_software_release.py")
    for filename in sorted(release.FROZEN_EXECUTION_RECORD_FILES):
        assert release.excluded(Path(filename)), filename
        assert release.excluded(Path("reruns") / filename), filename


def test_release_and_reproducibility_policies_agree() -> None:
    release = _load("_release", JOURNAL / "scripts" / "build_software_release.py")
    audit = _load(
        "_repro_audit", JOURNAL / "reproducibility" / "audit_reproducibility.py"
    )
    audited = {Path(p).name for p in audit.FROZEN_EXECUTION_RECORDS}
    assert audited == set(release.FROZEN_EXECUTION_RECORD_FILES)


def test_frozen_records_exist_and_stay_site_specific() -> None:
    release = _load("_release", JOURNAL / "scripts" / "build_software_release.py")
    for filename in sorted(release.FROZEN_EXECUTION_RECORD_FILES):
        path = JOURNAL / "configs" / filename
        assert path.is_file(), filename
        text = path.read_text(encoding="utf-8")
        assert "/n/" in text, f"{filename} no longer looks like an execution record"


def test_protocol_workers_survive_export_and_explicit_copy_only() -> None:
    release = _load("_release", JOURNAL / "scripts" / "build_software_release.py")
    expected_studies = {"conductance_local_gate", "credit_rule_extension", "measured_alignment_power"}
    assert set(release.FROZEN_WORKER_RECORDS) == {
        f"scripts/{study}/worker.sh" for study in expected_studies
    }
    for name in release.FROZEN_WORKER_RECORDS:
        assert (JOURNAL / name).is_file()
        assert release.repository_file_allowed(Path("journal") / name, "paper")
        # Directory copying filters bare .sh files; the explicit list must then
        # copy the required worker into both exported journal layouts.
        assert name.removeprefix("scripts/") in release.JOURNAL_SCRIPTS
        for prefix in ("article_analysis", "journal_package/journal"):
            assert not release.excluded(Path(prefix) / name)
    assert release.excluded(Path("worker.sh"))
    assert release.excluded(Path("journal/scripts/unrelated/worker.sh"))
    assert release.excluded(Path("journal/scripts/conductance_local_gate/other.sh"))


def test_cleanroom_helper_is_required_and_survives_directory_copy(tmp_path: Path) -> None:
    release = _load("_release_cleanroom", JOURNAL / "scripts" / "build_software_release.py")
    name = Path("code/release_noise/cleanroom_worker.sh")
    assert release.repository_file_allowed(Path("journal") / name, "paper")
    assert (Path("journal") / name).as_posix() in release.required_article_input_paths(JOURNAL)
    source = tmp_path / "code"
    helper = source / "release_noise/cleanroom_worker.sh"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((JOURNAL / name).read_bytes())
    (helper.parent / "unrelated.sh").write_text("not a release helper\n")
    destination = tmp_path / "copy"
    assert release.copy_tree_allowlisted(source, destination) == 1
    assert (destination / "release_noise/cleanroom_worker.sh").read_bytes() == helper.read_bytes()
    for prefix in ("article_analysis", "journal_package/journal"):
        assert not release.excluded(Path(prefix) / name)
    assert release.excluded(Path("cleanroom_worker.sh"))
    assert release.excluded(Path("unrelated/cleanroom_worker.sh"))


def test_original_runtime_paths_require_the_frozen_source_hash(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    audit = _load("_frozen_path_audit", JOURNAL / "reproducibility/audit_reproducibility.py")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    frozen = tmp_path / "code/population_replay/frozen"
    source = frozen / "runtime/example.py"
    source.parent.mkdir(parents=True)
    source.write_text('ROOT = "/n/' + 'holylabs/example/historical"\n')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    (frozen / "identity.json").write_text(json.dumps({"files": {"runtime/example.py": expected}}))
    audit.audit_private_paths()
    source.write_text(source.read_text() + '# changed\n')
    with pytest.raises(AssertionError, match="frozen population source changed"):
        audit.audit_private_paths()
    source.write_text('ROOT = "/n/' + 'holylabs/example/historical"\n')
    (frozen / "identity.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(AssertionError, match="private absolute paths remain"):
        audit.audit_private_paths()
