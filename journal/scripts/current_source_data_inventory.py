"""Assign the retained numerical evidence to the credit-first figure sequence.

Historical display-specific subsets keep their original filtering rules. Only
S45 is a direct main-to-supplement move; new main panels use the explicit,
verified panel map. Former main-only evidence remains supporting material.
"""
from __future__ import annotations
import csv
import re
from dataclasses import replace
from pathlib import Path

TEXT_EXTENSIONS = {'.csv', '.tsv', '.json', '.jsonl', '.txt', '.md', '.yaml', '.yml'}
DATA_EXTENSIONS = TEXT_EXTENSIONS | {'.npy', '.npz'}

def current_files(journal, legacy, cls, filters, counts, inventory=None):
    inventory = inventory or journal / 'source_data/credit_first_provenance/source_inventory.tsv'
    if not inventory.is_file():
        raise FileNotFoundError(f'Finalize the panel provenance before packaging: {inventory}')
    result = []
    for item in legacy:
        match = re.fullmatch(r'Figure (\d+)', item.figure)
        if not match:
            result.append(item)
            continue
        old = item.destination
        if match[1] == '2':
            destination = old.replace('Figure_2/', 'Supplementary_Figure_45/').replace('Fig2', 'SuppFig45')
            item = replace(item, figure='Supplementary Figure 45', destination=destination)
        else:
            item = replace(item, figure='Methods', panels='supporting evidence',
                destination='Methods/retained_evidence/' + old,
                notes=item.notes + ' Retained numerical evidence; its former main-panel assignment is superseded.')
        if old in filters:
            filters[item.destination] = filters[old]
        if old in counts:
            counts[item.destination] = counts[old]
        result.append(item)
    existing = {(x.figure, x.source) for x in result}
    any_existing = {x.source for x in result}
    additions = {}
    with inventory.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    for row in rows:
        source = row['source']
        if not source.startswith('source_data/') or Path(source).suffix.lower() not in DATA_EXTENSIONS:
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
            if (figure, source) in existing or (figure == 'Methods' and source in any_existing):
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
