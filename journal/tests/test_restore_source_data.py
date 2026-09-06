"""Display moves must not change original paths or silently truncate evidence."""
import csv
import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('restore',Path(__file__).resolve().parents[1]/'code/release_noise/restore_source_data.py')
restore=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(restore)


def manifest(root,records):
    root.mkdir()
    rows=[]
    for display,payload,transformation in records:
        p=root/display;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(payload)
        rows.append(dict(file=display,original_source='source_data/task/outcomes.csv',
                         sha256=restore.digest(p),original_sha256='original_unreleased_digest',transformation=transformation))
    with (root/'manifest.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],delimiter='\t');w.writeheader();w.writerows(rows)


def test_retained_methods_location_restores_original_path_and_prefers_full_copy(tmp_path):
    root=tmp_path/'Source_Data';journal=tmp_path/'journal'
    manifest(root,[('Methods/retained_evidence/Figure_6/full.csv','x\n1\n2\n','portable local-path prefix'),
                   ('Figure_4/subset.csv','x\n1\n','display-specific row filter: selected condition')])
    plan,issues=restore.restoration_plan(root,journal)
    assert not issues and len(plan)==1
    assert plan[0][1] == journal/'source_data/task/outcomes.csv'
    assert plan[0][0].read_text() == 'x\n1\n2\n'


def test_filtered_only_and_conflicting_copies_do_not_overwrite_complete_source(tmp_path):
    root=tmp_path/'Source_Data';journal=tmp_path/'journal'
    manifest(root,[('Figure_4/subset.csv','x\n1\n','display-specific row filter')])
    plan,issues=restore.restoration_plan(root,journal)
    assert not plan and 'Only display-filtered' in issues[0]['reason']
    assert not journal.exists()
    other=tmp_path/'conflicting'
    manifest(other,[('Figure_4/a.csv','x\n1\n','portable local-path prefix'),
                    ('Methods/b.csv','x\n2\n','portable local-path prefix')])
    plan,issues=restore.restoration_plan(other,journal)
    assert not plan and 'Conflicting complete copies' in issues[0]['reason']


def test_manifest_path_cannot_escape_destination(tmp_path):
    with pytest.raises(ValueError,match='Unsafe'):
        restore.child(tmp_path,'../outside.csv')
