"""Complete follow-up releases retain inputs and outcomes, without runtime logs."""
from dataclasses import dataclass
import csv
import importlib.util
import json
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'current_inventory_followup', JOURNAL / 'scripts/current_source_data_inventory.py')
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


@dataclass(frozen=True)
class Source:
    figure: str
    panels: str
    source: str
    destination: str
    role: str
    independent_unit: str
    status: str
    notes: str


def write_inventory(root, rows=()):
    path = root / 'inventory.tsv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, delimiter='\t', fieldnames=[
            'source', 'record_type', 'figure', 'panel', 'independent_unit', 'notes'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_followups_keep_complete_states_inputs_and_historical_sources(tmp_path):
    numerical = [
        'conductance_local_gate/runs/seed_1_states.npz',
        'conductance_local_gate/implementation_canary/seed_2_config.json',
        'credit_rule_extension/runs/seed_1_checkpoints.pt',
        'credit_rule_extension/protocol_freeze.json',
        'measured_alignment_power/inputs/original_target_aggregation.py',
        'measured_alignment_power/inputs/scan/functional_contact_pairs.csv',
        'measured_alignment_power/inputs/scan/observed_partner_responses.npz',
        'measured_alignment_power/runs/chunk_00.npz',
        'passive_field_diagnostics/surrogate_cell_heterogeneity.csv',
        'physical_depth_followup/paired_seed_trajectories.csv',
    ]
    excluded = [
        'conductance_local_gate/worker.log',
        'conductance_local_gate/main_figure_caption.tex',
        'credit_rule_extension/analysis_output.txt',
        'measured_alignment_power/slurm_logs/task_1.out',
        'measured_alignment_power/__pycache__/model.pyc',
        'measured_alignment_power/MANUSCRIPT_SNIPPETS.md',
        'measured_alignment_power/figures/sensitivity.pdf',
        'physical_depth_followup/figures/native_audit.txt',
    ]
    for source in numerical + excluded:
        path = tmp_path / 'source_data' / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    old = Source('Figure 2', 'c', 'source_data/old.csv', 'Figure_2/Fig2c.csv',
                 'historical', 'seed', 'retained', 'Original display filter.')
    filters = {old.destination: ('historical-filter',)}
    counts = {old.destination: 5}
    rows = inventory.current_files(tmp_path, [old], Source, filters, counts,
                                   write_inventory(tmp_path))
    by_source = {row.source: row for row in rows}
    assert set(by_source) == {'source_data/' + s for s in numerical} | {old.source}
    assert by_source[old.source].figure == 'Supplementary Figure 45'
    assert filters[by_source[old.source].destination] == filters[old.destination]
    assert counts[by_source[old.source].destination] == 5
    for source in numerical:
        study = source.split('/')[0]
        assert by_source['source_data/' + source].figure == inventory.FOLLOWUP_ASSIGNMENTS[study][0]


def test_explicit_companion_panel_assignment_precedes_fallback(tmp_path):
    source = 'source_data/credit_rule_extension/figures/controls_figure_source.csv'
    file = tmp_path / source
    file.parent.mkdir(parents=True)
    file.write_text('panel,value\nA,1\n')
    path = write_inventory(tmp_path, [dict(source=source, record_type='panel_source',
        figure='figS55', panel='a-d', independent_unit='20 paired seeds', notes='Controls.')])
    rows = inventory.current_files(tmp_path, [], Source, {}, {}, path)
    assert len(rows) == 1
    assert rows[0].figure == 'Supplementary Figure 55'
    assert rows[0].panels == 'a-d'


def test_every_frozen_measured_input_is_releasable():
    folder = JOURNAL / 'source_data/measured_alignment_power'
    for row in json.loads((folder / 'input_manifest.json').read_text()):
        path = folder / row['released']
        source = path.relative_to(JOURNAL).as_posix()
        assert path.is_file(), source
        assert inventory.releasable_followup(path.relative_to(JOURNAL), source), source
