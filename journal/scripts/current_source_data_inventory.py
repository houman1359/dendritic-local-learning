"""Assign numerical evidence to the current, explicitly mapped displays.

Former display-specific subsets keep their original filtering rules and source
identities under Methods. Current figure associations come from the verified
panel manifest, never from an obsolete figure number.
"""
from __future__ import annotations
import csv
import re
from dataclasses import replace
from pathlib import Path

TEXT_EXTENSIONS = {'.csv', '.tsv', '.json', '.jsonl', '.txt', '.md', '.yaml', '.yml'}
DATA_EXTENSIONS = TEXT_EXTENSIONS | {'.npy', '.npz'}
COMPRESSED_TABLE_SUFFIXES = ('.csv.gz', '.tsv.gz')

# Complete evidence from the explicitly authorized September follow-ups.  The
# panel manifest supplies precise displayed associations; the directory fallback
# keeps unplotted seeds, checkpoints and frozen inputs in the same release even
# before a freshly added file has a displayed-panel association.
FOLLOWUP_ASSIGNMENTS = {
    'cifar10_shunting_stopping_extension': ('Figure 1', 'same-seed stopping verification', '80 records, including eight same-seed replays; no new confirmation'),
    'fashion_strict_scalar_control': ('Figure 1', 'strict-scalar control', '80 fresh fits, ten paired seeds per architecture'),
    'additional_figure_controls': ('Supplementary Figure 24', 'complete control evidence', '80 fresh Fashion fits and 140 same-seed physical-placement sensitivities; per-row study labels'),
    'physical_depth_stopping_extension': ('Figure 7', 'same-seed stopping sensitivity', '60 fits, ten paired seeds per arm, uniform H200 hardware'),
    'conductance_local_gate': ('Figure 5', 'supporting local-gate evidence', '20 paired fresh seed blocks; canaries and historical replay separate'),
    'credit_rule_extension': ('Figure 4', 'supporting balanced-extension evidence', '20 previously observed paired seed blocks; 720 continued trajectories'),
    'measured_alignment_power': ('Figure 10', 'supporting conditional-sensitivity evidence', '4000 global simulated datasets; original 13 scans within 7 target cells'),
    'passive_field_diagnostics': ('Figure 8', 'supporting passive-field decomposition', 'reconstructed cell; post-review diagnostic without new learning fits'),
    'physical_depth_followup': ('Figure 7', 'supporting budget-indexed trajectory evidence', '10 paired training seeds; original 60 fits without new training'),
}
FOLLOWUP_EXCLUDED_COMPONENTS = {'__pycache__', '.pytest_cache', 'slurm_logs', 'logs'}
FOLLOWUP_EXCLUDED_NAMES = {
    'report_execution.txt', 'analysis_output.txt', 'figure_build.txt',
    'independent_audit_output.txt', 'original_model_tests.txt',
    'native_audit.txt', 'initial_native_audit.txt',
    'MANUSCRIPT_SNIPPETS.md', 'RELEASE_GUIDANCE.md',
    'FIGURE_CAPTION.md', 'CONTROLS_FIGURE_CAPTION.md',
}
ARCHIVED_INPUT_CODE = {
    'source_data/measured_alignment_power/inputs/original_functional_topology_analysis.py',
    'source_data/measured_alignment_power/inputs/original_target_aggregation.py',
    'source_data/conductance_local_gate/report_initial_render.py',
}
# The portable gate verifier authenticates these exact historical evidence and
# native output bytes. Other scheduler logs and image assets remain excluded.
VERIFIER_REQUIRED_ARTIFACTS = {
    'source_data/conductance_local_gate/canary_45289328.log',
    'source_data/conductance_local_gate/figures/local_gate_primary.pdf',
    'source_data/conductance_local_gate/figures/local_gate_primary.png',
    'source_data/conductance_local_gate/figures/local_gate_all_rates.pdf',
    'source_data/conductance_local_gate/figures/local_gate_all_rates.png',
}


def releasable_followup(path, source):
    """Keep numerical/protocol evidence, not scheduler output or draft prose."""
    return (
        not FOLLOWUP_EXCLUDED_COMPONENTS.intersection(path.parts)
        and path.name not in FOLLOWUP_EXCLUDED_NAMES
        and (path.suffix.lower() in DATA_EXTENSIONS | {'.pt', '.pth'}
             or source in ARCHIVED_INPUT_CODE
             or source in VERIFIER_REQUIRED_ARTIFACTS
             or source.lower().endswith(COMPRESSED_TABLE_SUFFIXES))
    )

def current_files(journal, legacy, cls, filters, counts, inventory=None):
    inventory = inventory or journal / 'source_data/credit_first_provenance/source_inventory.tsv'
    if not inventory.is_file():
        raise FileNotFoundError(f'Finalize the panel provenance before packaging: {inventory}')
    result = []
    for item in legacy:
        old = item.destination
        item = replace(item, figure='Methods', panels='supporting evidence',
            destination='Methods/retained_evidence/' + old,
            notes=item.notes + ' Complete supporting evidence with the original display filter; current panel associations are listed separately in the provenance manifest.')
        if old in filters:
            filters[item.destination] = filters[old]
        if old in counts:
            counts[item.destination] = counts[old]
        result.append(item)
    existing = {(x.figure, x.source) for x in result}
    any_existing = {x.source for x in result}
    complete_existing = {x.source for x in result if x.destination not in filters}
    additions = {}
    with inventory.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    for row in rows:
        source = row['source']
        if not source.startswith('source_data/'):
            continue
        study = Path(source).parts[1] if len(Path(source).parts) > 1 else ''
        if study in FOLLOWUP_ASSIGNMENTS:
            accepted = releasable_followup(Path(source), source)
        else:
            accepted = Path(source).suffix.lower() in DATA_EXTENSIONS or source.lower().endswith(COMPRESSED_TABLE_SUFFIXES)
        if not accepted:
            continue
        if row['record_type'] == 'figure_asset':
            continue
        figure_code = row['figure']
        # Mixed retained SI associations already have separately scoped legacy copies.
        codes = figure_code.split('/')
        panels = row['panel'].split('/') if len(codes) > 1 else [row['panel']]
        for k, code in enumerate(codes):
            m = re.fullmatch(r'fig(S?)(\d+)', code)
            figure = ('Supplementary Figure ' if m[1] else 'Figure ') + m[2] if m else 'Methods'
            if (figure, source) in existing and source in complete_existing:
                continue
            if figure == 'Methods' and source in complete_existing:
                continue
            panel = panels[k] if k < len(panels) else row['panel']
            key = (figure, source)
            if key in additions:
                previous = additions[key]
                additions[key] = replace(previous, panels=', '.join(dict.fromkeys([previous.panels, panel])))
                continue
            directory = figure.replace(' ', '_')
            destination = directory + '/' + str(Path(source).relative_to('source_data'))
            additions[key] = cls(figure, panel, source, destination, row['record_type'],
                row['independent_unit'], 'current; see frozen protocol and lineage records', row['notes'])
    represented = any_existing | {item.source for item in additions.values()}
    for study, (figure, panel, unit) in FOLLOWUP_ASSIGNMENTS.items():
        directory = journal / 'source_data' / study
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            source = path.relative_to(journal).as_posix()
            if source in represented or not releasable_followup(path.relative_to(journal), source):
                continue
            destination = figure.replace(' ', '_') + '/' + str(Path(source).relative_to('source_data'))
            additions[(figure, source)] = cls(
                figure, panel, source, destination,
                'complete supporting experimental source or scientific protocol',
                unit, 'current; see frozen protocol and lineage records',
                'Retained complete follow-up evidence. This supporting association does not imply that every saved observation is plotted or independently replicated.',
            )
            represented.add(source)
    result.extend(additions.values())
    destinations = [x.destination for x in result]
    if len(destinations) != len(set(destinations)):
        raise ValueError('Current Source Data destinations are not unique')
    return tuple(result)

class CurrentInventory:
    """Load after provenance is finalized, including for imported validation tools."""
    def __init__(self, journal, legacy, cls, filters, counts):
        self.args = journal, legacy, cls, filters, counts
    def __iter__(self):
        return iter(current_files(*self.args))
    def __len__(self):
        return len(current_files(*self.args))
