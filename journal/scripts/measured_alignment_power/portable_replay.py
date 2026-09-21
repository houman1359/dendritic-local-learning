#!/usr/bin/env python3
"""Verify frozen sensitivity inputs and replay unchanged science in a new folder.

Absolute provenance paths and the scheduler working directory can be translated
by the release builder.  Original expected hashes remain immutable; the common
release verifier must authenticate any declared original-to-released byte link.
Only the runner's output directory and hash-check entry point are rebound after
this verification.  No numerical function, parameter, seed or input is changed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PROTOCOL_SHA256 = '1fc33b46fdcd7e6c2f84aa9236190f9a17281c4cb5db4cb6b9d6ab720a299cf1'
JOURNAL = Path(__file__).resolve().parents[2]
STUDY = Path('source_data/measured_alignment_power')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe registered input path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Registered input escapes the declared study')
    return path


def verify(journal):
    journal = Path(journal).resolve()
    out = journal / STUDY
    helper = module('_sensitivity_release_hashes', journal / 'code/release_noise/release_hashes.py')
    checks = []

    def check(path, expected):
        verdict = helper.verify_released_file(path, expected, journal_root=journal)
        if not verdict['verified']:
            raise ValueError(f'Frozen source verification failed: {path.relative_to(journal)}: {verdict["reason"]}')
        checks.append(dict(path=path.relative_to(journal).as_posix(),
                           original_sha256=expected,
                           released_sha256=verdict['released_sha256'],
                           verification=verdict['reason']))

    check(out / 'protocol_freeze.json', PROTOCOL_SHA256)
    protocol = json.loads((out / 'protocol_freeze.json').read_text())
    for filename, key in [('PROTOCOL.md', 'protocol_markdown_sha256'),
                          ('input_manifest.json', 'input_manifest_sha256'),
                          ('input_audit.json', 'input_audit_sha256')]:
        check(out / filename, protocol[key])
    for source, expected in protocol['scientific_code_sha256'].items():
        check(child(journal, source), expected)
    inputs = json.loads((out / 'input_manifest.json').read_text())
    for row in inputs:
        check(child(out, row['released']), row['released_sha256'])
    report = dict(status='PASS', protocol_sha256=PROTOCOL_SHA256,
                  verified_files=len(checks),
                  declared_release_transformations=sum(
                      row['verification'] != 'original bytes' for row in checks),
                  checks=checks,
                  scope='Original protocol expectations preserved; any translated bytes require an authenticated release-provenance chain.')
    return protocol, report


def replay(journal, output, chunk):
    journal = Path(journal).resolve()
    canonical = journal / STUDY
    output = Path(output).resolve()
    if output.is_relative_to(canonical) or canonical.is_relative_to(output):
        raise ValueError('Use a separate output folder outside canonical Source Data')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty replay folder')
    protocol, verification = verify(journal)
    if not 0 <= chunk < protocol['n_chunks']:
        raise ValueError('Chunk is outside the frozen protocol')
    scripts = journal / 'scripts/measured_alignment_power'
    model = module('_sensitivity_frozen_model', scripts / 'model.py')
    # The original runner imports its local model by this short name.  Keep
    # other analyses' modules intact while binding this verified model once.
    old_model = sys.modules.get('model')
    sys.modules['model'] = model
    try:
        runner = module('_sensitivity_frozen_runner', scripts / 'run.py')
    finally:
        if old_model is None:
            sys.modules.pop('model', None)
        else:
            sys.modules['model'] = old_model
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(canonical / 'inputs', output / 'inputs')
    shutil.copyfile(canonical / 'protocol_freeze.json', output / 'protocol_freeze.json')
    runner.OUT = output
    runner.checked_protocol = lambda: protocol
    runner.run_chunk(chunk)
    import numpy as np
    reference = canonical / 'runs' / f'chunk_{chunk:02d}.npz'
    actual = output / 'runs' / reference.name
    comparison = None
    if reference.is_file():
        old = np.load(reference)
        new = np.load(actual)
        if set(old.files) != set(new.files):
            raise ValueError('Replay array inventory differs from the original chunk')
        differences = {}
        for key in old.files:
            if old[key].dtype.kind in 'fci':
                differences[key] = float(np.max(np.abs(old[key] - new[key])))
                if not np.allclose(old[key], new[key], rtol=0, atol=1e-12):
                    raise ValueError(f'Replay differs beyond roundoff: {key}')
            elif not np.array_equal(old[key], new[key]):
                raise ValueError(f'Replay identity differs: {key}')
        comparison = dict(status='PASS', maximum_absolute_differences=differences,
                          tolerance=1e-12)
    report = dict(status='PASS', chunk=chunk, verification=verification,
                  canonical_comparison=comparison,
                  numerical_scope='Reproduction of one original 200-dataset chunk; no new independent scientific outcome.',
                  io_adaptation='Verified immutable numerical runner, model and protocol; only private output path and equivalent authenticated hash verification are rebound.')
    (output / 'portable_replay_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal-root', type=Path, default=JOURNAL)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--chunk', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if args.chunk is not None or args.output is not None:
            parser.error('--verify-only does not write simulation outputs')
        _, report = verify(args.journal_root)
    else:
        if args.chunk is None or args.output is None:
            parser.error('Specify --chunk and --output, or --verify-only')
        report = replay(args.journal_root, args.output, args.chunk)
    print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
