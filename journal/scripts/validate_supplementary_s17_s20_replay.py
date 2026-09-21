#!/usr/bin/env python3
"""Compare S17–S20 numerical artists with the original source renderers.

This focused rendering test does not rerun learning or rewrite source tables.
All source figure saving and legacy audit writes are suppressed. Plot-coordinate
arrays (lines, scatter offsets, violin paths and error-bar segments) are compared
as multisets, allowing panel placement, labels and typography to change.
"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
import numpy as np

import build_supplementary_figures_s17_s20_native as new
ROOT=new.ROOT


def numerical_artists(ax):
    arrays=[]
    def add(kind, values):
        x=np.asarray(values,dtype=float)
        arrays.append((kind,tuple(x.shape),tuple(x.ravel())))
    for art in ax.lines:
        add('line',art.get_xydata())
    for art in ax.collections:
        if isinstance(art,LineCollection):
            for seg in art.get_segments():add('errorbar',seg)
        elif isinstance(art,PathCollection):
            add('scatter',art.get_offsets())
        elif isinstance(art,PolyCollection):
            for p in art.get_paths():add('density_or_band',p.vertices)
        else:raise TypeError(type(art))
    return Counter(arrays)


def capture(module, function, *args):
    with ExitStack() as stack:
        stack.enter_context(patch.object(Figure,'savefig',lambda *a,**k:None))
        stack.enter_context(patch.object(plt,'close',lambda *a,**k:None))
        for name in ('audit_layout','audit_text_over_data'):
            if hasattr(module,name):stack.enter_context(patch.object(module,name,lambda *a,**k:{}))
        if hasattr(module,'save'):stack.enter_context(patch.object(module,'save',lambda *a,**k:None))
        function(*args)
        return plt.gcf()


def validate():
    import build_alignment_animal_figure as animal
    import analyze_point_dendrite_credit_controls as point
    import analyze_physical_alignment_dose as dose
    import analyze_remaining_physical_experiments as remaining
    import analyze_prospective_learning_results as prospective
    import build_journal_figures as anatomy
    result={'scope':'Frozen-data plotting replay; no experiments or scientific source files modified.',
            'comparison':'exact floating-point array equality (no rounding)','panels':{},'passed':False}
    canvases=new.build(save=False)
    originals={}
    fig,ax=plt.subplots(1,2)
    animal.animal_schematic(ax[0]);animal.neuron_distributions(ax[1],new.read('animal_learning_francioni','neuron_sd_residual_distributions.csv'))
    originals.update({(17,'A'):ax[0],(17,'B'):ax[1]})
    fig=capture(point,point.render_figure,new.read('point_dendrite_credit_controls','condition_summary.csv'),new.read('point_dendrite_credit_controls','paired_contrasts.csv'))
    originals.update({(18,'A'):fig.axes[2],(18,'B'):fig.axes[5]})
    fig=capture(dose,dose.render,new.read('physical_alignment_dose','condition_summary.csv'),new.read('physical_alignment_dose','paired_contrasts.csv'))
    originals.update({(18,'C'):fig.axes[0],(18,'D'):fig.axes[2]})
    fig=capture(remaining,remaining.make_figure,new.read('remaining_physical_experiments','condition_summary.csv'),new.read('remaining_physical_experiments','paired_contrasts.csv'))
    originals.update({(18,'E'):fig.axes[2],(18,'F'):fig.axes[5]})
    fig=capture(prospective,prospective._plot_streamlined_main)
    originals.update({(19,'A'):fig.axes[6],(19,'B'):fig.axes[8]})
    fig=capture(anatomy,anatomy._figure3_detailed)
    originals.update({(20,'A'):fig.axes[2],(20,'B'):fig.axes[7]})
    for (number,letter),old in originals.items():
        a=numerical_artists(old);b=numerical_artists(canvases[number].axes[letter])
        result['panels'][f'S{number}{letter}']={'source_artist_arrays':sum(a.values()),
                                             'native_artist_arrays':sum(b.values()),
                                             'matching':a==b,
                                             'source_only':sum((a-b).values()),
                                             'native_only':sum((b-a).values())}
    result['passed']=all(r['matching'] for r in result['panels'].values())
    inputs=[]
    for folder,names in {
       'animal_learning_francioni':['neuron_sd_residual_distributions.csv'],
       'point_dendrite_credit_controls':['condition_summary.csv','paired_contrasts.csv'],
       'physical_alignment_dose':['condition_summary.csv','paired_contrasts.csv'],
       'remaining_physical_experiments':['condition_summary.csv','paired_contrasts.csv'],
       'trained_subtree_address':['seed_outcomes.csv'],
       'figure3':['segment_metrics.csv','typed_only_compression_curves.csv']}.items():
        for name in names:
            p=new.DATA/folder/name
            inputs.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result['source_inputs']=inputs
    path=ROOT/'analysis/audience_style_revision_20260905/si17_20_replay_validation.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_inputs'},indent=2))
    plt.close('all')
    if not result['passed']:raise AssertionError('Native numerical artists differ; inspect validation JSON')
    return result


if __name__=='__main__':validate()
