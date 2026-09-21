"""Portable verification accepts declared path translation and rejects tampering."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('power_portable_test', HERE / 'portable_replay.py')
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tsv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, delimiter='\t', fieldnames=list(rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def release_fixture(destination):
    """Small release fixture: unchanged science plus authenticated path changes."""
    study = portable.STUDY
    source = JOURNAL / study
    for folder in [study / 'inputs', Path('code/release_noise')]:
        shutil.copytree(JOURNAL / folder, destination / folder)
    for name in ['protocol_freeze.json', 'PROTOCOL.md', 'input_manifest.json', 'input_audit.json']:
        shutil.copyfile(source / name, destination / study / name)
    protocol = json.loads((source / 'protocol_freeze.json').read_text())
    for name in protocol['scientific_code_sha256']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(JOURNAL / name, target)
    helper = portable.module('_power_fixture_hashes', destination / 'code/release_noise/release_hashes.py')
    input_path = destination / study / 'input_manifest.json'
    input_origin = sha(input_path)
    manifest = json.loads(input_path.read_text())
    # This locator is descriptive provenance, never a numerical input path.
    # Translate it without embedding any machine-specific source prefix.
    manifest[0]['source'] = 'fixture-origin://original-observed-pair-table'
    input_path.write_text(json.dumps(manifest, indent=2) + '\n')
    data_provenance = destination / 'RELEASED_SOURCE_MANIFEST.tsv'
    tsv(data_provenance, [dict(original_source=(study / input_path.name).as_posix(),
        original_sha256=input_origin, sha256=sha(input_path),
        transformation='Translate one provenance-only source locator for the portable verification fixture')])
    data_row = dict(path=(study / input_path.name).as_posix(), kind='source_data',
        origin_sha256=input_origin, release_sha256=sha(input_path),
        transformation='Translate one provenance-only source locator for the portable verification fixture',
        provenance_file=data_provenance.name, provenance_sha256=sha(data_provenance))
    worker_relative = Path('scripts/measured_alignment_power/worker.sh')
    worker = destination / worker_relative
    worker_origin = sha(worker)
    worker.write_text(worker.read_text().replace(str(JOURNAL), '${REVIEWER_JOURNAL_ROOT}'))
    software_provenance = destination / 'PORTABILITY_PATCHES.tsv'
    tsv(software_provenance, [dict(path=worker_relative.as_posix(),
        origin_sha256=worker_origin, release_sha256=sha(worker),
        reason='Translate scheduler working directory; numerical worker unchanged')])
    software_row = dict(path=worker_relative.as_posix(), kind='software',
        origin_sha256=worker_origin, release_sha256=sha(worker),
        transformation='Translate scheduler working directory',
        provenance_file=software_provenance.name, provenance_sha256=sha(software_provenance))
    helper.write_sidecar(destination, [data_row], 'source_data')
    helper.write_sidecar(destination, [software_row], 'software')
    return destination


def test_original_sources_and_declared_release_translations(tmp_path):
    _, original = portable.verify(JOURNAL)
    _, released = portable.verify(release_fixture(tmp_path / 'journal'))
    assert original['status'] == released['status'] == 'PASS'
    assert original['declared_release_transformations'] == 0
    assert released['declared_release_transformations'] == 2
    assert original['verified_files'] == released['verified_files']


def test_translated_input_without_link_and_tampered_science_are_rejected(tmp_path):
    root = release_fixture(tmp_path / 'journal')
    sidecar = root / 'RELEASED_SOURCE_HASHES.tsv'
    backup = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(ValueError, match='verification failed'):
        portable.verify(root)
    sidecar.write_bytes(backup)
    model = root / 'scripts/measured_alignment_power/model.py'
    model.write_text(model.read_text() + '\n# Undeclared post-freeze edit.\n')
    with pytest.raises(ValueError, match='model.py'):
        portable.verify(root)


def test_registered_paths_cannot_escape_study(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        portable.child(tmp_path, '../other/data.csv')
