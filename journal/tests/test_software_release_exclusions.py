"""The frozen CIFAR execution records must never enter the software release.

These five launch YAMLs document completed cluster runs with site-specific
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
