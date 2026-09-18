"""Create an immutable, independently rerunnable source/input snapshot."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import subprocess

from run import save_new, sha


def prepare(root, production, journal):
    root.mkdir(parents=True, exist_ok=False)
    study = Path(__file__).resolve().parent
    for source in study.glob('*'):
        if source.suffix in {'.py', '.sh', '.md'}:
            destination = root/'study'/source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
    for source in (production/'src').rglob('*.py'):
        if '__pycache__' not in source.parts:
            destination = root/'runtime'/source.relative_to(production)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    # Existing anatomy builders retained byte-for-byte; no new anatomy fitting.
    for source in (journal/'code/reconstructed_tree').glob('*.py'):
        destination = root/'anatomy_code/code/reconstructed_tree'/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    source = journal/'scripts/analyze_physical_cable_sensitivity.py'
    destination = root/'anatomy_code/scripts'/source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    for name, relative in [('microns_pilot', 'figure3/segment_metrics.csv'),
                           ('microns_replication', 'microns_v661_replication/routing/segment_metrics.csv'),
                           ('pinky', 'pinky_v185_replication/routing/segment_metrics.csv')]:
        source = journal/'source_data'/relative
        destination = root/'inputs'/f'{name}.csv'
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    functional = journal/'source_data/measured_alignment_power/inputs'
    shutil.copytree(functional, root/'inputs/functional')
    jobs = [dict(seed=seed, somata=n, streams=b, task=task, mode=mode)
            for seed, (n, b), task, mode in itertools.product(
                range(2026091900, 2026091903), [(1, 2), (16, 2), (16, 4)],
                ['selection', 'mixture'], ['shunt', 'tonic', 'current'])]
    sources = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
               if p.is_file() and p.suffix in {'.py', '.sh'}}
    inputs = {str(p.relative_to(root)): sha(p) for p in sorted((root/'inputs').rglob('*')) if p.is_file()}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    phase='Exploratory development; not preregistered or confirmatory',
                    source_sha256=sources, input_sha256=inputs,
                    production_head=subprocess.check_output(['git', '-C', str(production), 'rev-parse', 'HEAD'], text=True).strip(),
                    production_state='Working-tree snapshot, including pre-existing user edits; hashes identify actual execution',
                    jobs=jobs, rates=[.01, .03], steps=1024, train_size=2048,
                    endpoint='Test NMSE at training-validation-selected state; test never selects rates or models',
                    planned_fresh_seeds=list(range(2026092000, 2026092020)),
                    next_stage='Review development failures and exact-BP feasibility, freeze rates and protocol, then run 20 fresh paired seeds',
                    comparisons=['forward inhibition x credit-rule interaction', 'spatial gate vs uniform RMS gate',
                                 'wrong-branch assignment', 'equal-norm common-state core-gradient steps',
                                 'selection under stronger held-out distractors'],
                    controls='All factorial conditions retain the same excitatory context cue; all conductances and readouts learn',
                    caveats=['Current injection matches a training-initialization tonic operating point, not every voltage during training',
                             'Cue-derived gate under tonic/current forward models is a computational intervention, not a demonstrated local biological mechanism',
                             'Uniform-RMS matches delivered-signal RMS, not parameter-gradient norm; matched-state diagnostics separately match core update norm',
                             'Common somatic error supplied; all neurons are artificial directed steady-state trees',
                             'Independent source/gradient transfer audit establishes Figure 5 code parity, not population generalization by itself'],
                    anatomy=dict(animals=2, cohorts=['microns_pilot', 'microns_replication', 'pinky'],
                                 membrane_resistance=[15000., 1000.], doses=[.25, 1., 4.],
                                 endpoint='Signed transport-vs-driving-force changes at fixed soma voltage/error and fixed initial electrical parameters',
                                 inference='Cell summaries within each mouse; sites and cells are not independent animals'),
                    functional=dict(endpoint='Within-minus-between-subtree held-out response covariance, weighted by inhibitory contact abundance',
                                    controls='Anatomical sites selected without functional outcomes; split-repeat cross-products; descriptive target-wise adjustment for depth, coverage and excitatory abundance; no independent-pair tests',
                                    limitation='Observational responses, not recorded learning errors, task contexts or inhibitory activity'))
    save_new(root/'protocol.json', protocol)
    for directory in ('results', 'checkpoints', 'logs', 'anatomy_results', 'functional_results'):
        (root/directory).mkdir()
    print(json.dumps(dict(root=str(root), jobs=len(jobs), sources=len(sources), inputs=len(inputs)), indent=2))


def biology_revision(root, parent):
    """Separate revision; never mutate the execution snapshot of running jobs."""
    root.mkdir(parents=True, exist_ok=False)
    (root/'study').mkdir()
    for p in Path(__file__).resolve().parent.glob('*'):
        if p.suffix in {'.py', '.sh', '.md'}:
            shutil.copyfile(p, root/'study'/p.name)
    for name in ['runtime', 'inputs', 'anatomy_code']:
        (root/name).symlink_to(parent/name, target_is_directory=True)
    for name in ['logs', 'anatomy_results', 'functional_results']:
        (root/name).mkdir()
    protocol = json.loads((parent/'protocol.json').read_text())
    protocol.update(parent_protocol_sha256=sha(parent/'protocol.json'),
                    revision='Preserve pre-existing Pinky qc_included exclusions; original bio array cancelled before analysis',
                    created_utc=datetime.now(timezone.utc).isoformat())
    for relative in list(protocol['source_sha256']):
        if relative.startswith('study/'):
            del protocol['source_sha256'][relative]
    protocol['source_sha256'].update({str(p.relative_to(root)): sha(p) for p in (root/'study').glob('*')
                                      if p.suffix in {'.py', '.sh'}})
    save_new(root/'protocol.json', protocol)
    print(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--production', type=Path)
    p.add_argument('--journal', type=Path)
    p.add_argument('--biology-parent', type=Path)
    a = p.parse_args()
    if a.biology_parent:
        biology_revision(a.root, a.biology_parent)
    else:
        prepare(a.root, a.production, a.journal)
