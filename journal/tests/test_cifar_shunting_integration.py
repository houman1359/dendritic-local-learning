"""Test-only generated data; none of these fixtures enter manuscript source data."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

J = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(J / 'scripts'), str(J / 'scripts/credit_first_figures')]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

F = load('pending_framework', J/'scripts/credit_first_figures/build_framework.py')
S = load('pending_s04', J/'scripts/build_supplementary_figure_s04_native.py')
A = load('frozen_shunting_analysis', J/'scripts/analyze_cifar10_shunting_feedback_ladder_confirmatory.py')


def package(path):
    folder = path/'cifar10_shunting_feedback_ladder_confirmatory'
    folder.mkdir()
    rows=[]
    for i, seed in enumerate(A.EXPECTED_SEEDS):
        for label, value in [('strict scalar', .32), ('neuron specific', .50),
                             ('exact path', .47), ('backpropagation', .50)]:
            rows.append(dict(seed=seed, feedback=label,
                             test_accuracy=value + .001*(i%4) + (.0002*i if label=='exact path' else 0)))
    frame=pd.DataFrame(rows)
    audit=dict(status='complete_and_validated', integrity_valid=True,
               convergence_valid=True, n_expected=80,n_results_complete=80)
    conditions, contrasts, decision=A.summarize(frame,audit)
    frame.to_csv(folder/'seed_outcomes.csv',index=False)
    conditions.to_csv(folder/'condition_summary.csv',index=False)
    contrasts.to_csv(folder/'paired_contrasts.csv',index=False)
    (folder/'summary.json').write_text(json.dumps(dict(
        audit=audit,decision=decision,contract_frozen_before_confirmatory_outcomes=True)))
    return folder


def test_existing_additive_values_are_unchanged(monkeypatch):
    monkeypatch.setattr(F,'SOURCE',J/'source_data')
    monkeypatch.setattr(F,'ARCHITECTURES',('additive',))
    contrasts,pairs,_=F.read_cifar_ladder('additive')
    assert set(pairs.index)==set(range(10800,10820))
    eq=F.read_bp_equivalence()[0]
    assert eq['equivalent']
    assert eq['mean_pp']==pytest.approx(-.11549955606460571,abs=.005)


def test_valid_negative_shunting_result_remains_visible(tmp_path, monkeypatch):
    folder=package(tmp_path)
    monkeypatch.setattr(F,'SOURCE',tmp_path)
    monkeypatch.setattr(F,'ARCHITECTURES',('shunting',))
    contrasts,pairs,_=F.read_cifar_ladder('shunting')
    assert len(pairs)==20
    eq=F.read_bp_equivalence()[0]
    assert eq['mean_pp']<0 and not eq['equivalent']
    assert eq['architecture']=='shunting'
    outcomes,summary=S.load_confirmatory_analysis(folder,architecture='shunting')
    assert len(outcomes)==80 and not summary['decision']['path_resolution_main_promotion']


def test_incomplete_or_wrong_seed_shunting_package_cannot_be_plotted(tmp_path, monkeypatch):
    folder=package(tmp_path)
    monkeypatch.setattr(F,'SOURCE',tmp_path)
    frame=pd.read_csv(folder/'seed_outcomes.csv')
    frame=frame.iloc[1:]
    frame.to_csv(folder/'seed_outcomes.csv',index=False)
    with pytest.raises(AssertionError):
        F.read_cifar_ladder('shunting')
    with pytest.raises(RuntimeError):
        S.load_confirmatory_analysis(folder,architecture='shunting')


def test_wrong_interval_is_rejected(tmp_path, monkeypatch):
    folder=package(tmp_path)
    monkeypatch.setattr(F,'SOURCE',tmp_path)
    frame=pd.read_csv(folder/'paired_contrasts.csv')
    frame.loc[0,'ci95_high_difference']+=.01
    frame.to_csv(folder/'paired_contrasts.csv',index=False)
    with pytest.raises(AssertionError):
        F.read_cifar_ladder('shunting')


def test_failed_audit_is_rejected(tmp_path, monkeypatch):
    folder=package(tmp_path)
    monkeypatch.setattr(F,'SOURCE',tmp_path)
    p=folder/'summary.json'; summary=json.loads(p.read_text())
    summary['audit']['convergence_valid']=False
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Unvalidated'):
        F.read_cifar_ladder('shunting')


def test_published_shunting_requires_explicit_convergence_disclosure(monkeypatch):
    monkeypatch.setattr(F, 'SOURCE', J/'source_data')
    with pytest.raises(ValueError, match='convergence'):
        F.read_cifar_ladder('shunting')
    contrasts, pairs, summary = F.read_cifar_ladder('shunting', allow_convergence_flags=True)
    assert len(pairs) == 20 and not summary['decision']['audit_passes']
    outcomes, audit = S.load_confirmatory_analysis(
        J/'source_data/cifar10_shunting_feedback_ladder_confirmatory',
        architecture='shunting', allow_convergence_flags=True)
    assert len(outcomes) == 80 and outcomes.right_censored.sum() == 1
    assert audit['audit']['convergence_valid'] is False


def test_display_opt_in_never_waives_integrity(tmp_path, monkeypatch):
    folder = package(tmp_path)
    monkeypatch.setattr(F, 'SOURCE', tmp_path)
    p = folder/'summary.json'; summary = json.loads(p.read_text())
    summary['audit']['integrity_valid'] = False
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Unvalidated'):
        F.read_cifar_ladder('shunting', allow_convergence_flags=True)
    with pytest.raises(RuntimeError, match='integrity'):
        S.load_confirmatory_analysis(folder, architecture='shunting', allow_convergence_flags=True)


def test_inconsistent_flag_is_rejected_even_with_display_opt_in(tmp_path, monkeypatch):
    folder = package(tmp_path)
    monkeypatch.setattr(F, 'SOURCE', tmp_path)
    p = folder/'summary.json'; summary = json.loads(p.read_text())
    summary['audit'].update(status='complete_with_convergence_flags', convergence_valid=False,
                            convergence_flags=['seed-22008: fabricated-condition: flag'])
    summary['decision']['audit_passes'] = False
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Missing seed-level'):
        F.read_cifar_ladder('shunting', allow_convergence_flags=True)
