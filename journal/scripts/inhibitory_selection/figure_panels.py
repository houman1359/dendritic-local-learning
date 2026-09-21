"""Native Figure 5 population panels; no fitting or outcome selection here."""
from pathlib import Path
import numpy as np
import pandas as pd

from figure_canvas import (COLORS, LW_DATA, LW_ERR, PT_BASE, PT_SMALL, SEED_MS,
                           SEED_ALPHA, style_panel)

J = Path(__file__).resolve().parents[2]
DATA = J / 'source_data/curated_publication'
RULES = ('exact', 'broadcast', 'resistance')
COLORS_BY_RULE = dict(exact=COLORS['bp'], broadcast=COLORS['scalar'],
                     resistance=COLORS['shunting'])


def data():
    ep = pd.read_csv(DATA / 'inhibitory_selection_endpoints.csv')
    summary = pd.read_csv(DATA / 'inhibitory_selection_summary.csv')
    return ep[ep.variant.eq('separable')], summary[summary.variant.eq('separable')]


def population_panels(h, i, log_ticks):
    ep, summary = data()
    for ax in [h, i]:
        style_panel(ax)
        ax.set_ylabel('NMSE')
    severity = np.array([1., 1.5, 2., 2.5, 3.])
    shown = summary[summary.forward.eq('shunt') & summary.rule.isin(RULES)
                    & summary.metric.str.startswith('ood_')].copy()
    for rule in RULES:
        part = shown[shown.rule.eq(rule)].set_index('metric').loc[
            [f'ood_{s}' for s in severity]]
        color = COLORS_BY_RULE[rule]
        h.fill_between(severity, part.ci_low, part.ci_high, color=color, alpha=.16, lw=0)
        h.plot(severity, part['mean'], color=color, lw=LW_DATA,
               label={'exact': 'Exact BP', 'broadcast': 'Unit broadcast',
                      'resistance': 'Relative-resistance gate'}[rule],
               ls=(0, (3.4, 1.7)) if rule == 'resistance' else '-', zorder=4)
    h.legend(loc='upper left', frameon=False, fontsize=PT_SMALL,
             handlelength=2.1, labelspacing=.15, borderaxespad=.25)
    h.set_xlim(.94, 3.06)
    h.set_xticks([1, 2, 3])
    h.set_xlabel('Distractor severity')
    log_ticks(h, [1e-5, 1e-3, 1e-1])
    h.set_ylim(1e-5, .1)
    h.text(.5, 1.045, '16-neuron DendriNet', ha='center', va='bottom',
           transform=h.transAxes, fontsize=PT_BASE, color=COLORS['ink'])
    for x, forward in enumerate(['shunt', 'tonic', 'current']):
        for offset, rule in zip([-.22, 0, .22], RULES):
            values = ep[ep.forward.eq(forward) & ep.rule.eq(rule)].sort_values('seed')['ood_3.0']
            point = summary[summary.forward.eq(forward) & summary.rule.eq(rule)
                            & summary.metric.eq('ood_3.0')].iloc[0]
            color = COLORS_BY_RULE[rule]
            i.scatter(x + offset + np.linspace(-.065, .065, len(values)), values,
                      s=SEED_MS**2, color=color, alpha=SEED_ALPHA, linewidths=0)
            # Open, enlarged means remain distinguishable from the seed cloud.
            i.errorbar(x + offset, point['mean'],
                       yerr=[[point['mean']-point.ci_low], [point.ci_high-point['mean']]],
                       fmt='D', color=color, ms=5.2, lw=LW_ERR, capsize=2.,
                       markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.1, zorder=6)
    i.set_xlim(-.48, 2.48)
    i.set_xticks([0, 1, 2], ['Shunting', 'Tonic', 'Current'])
    i.set_xlabel('Forward inhibition')
    log_ticks(i, [1e-5, 1e-3, 1e-1, 1.])
    i.set_ylim(1e-5, 2.)
    i.text(.5, 1.045, 'Strong distractors (severity 3)', ha='center', va='bottom',
           transform=i.transAxes, fontsize=PT_BASE, color=COLORS['ink'])
    return dict(H=shown.to_dict('records'),
                I=summary[summary.rule.isin(RULES) & summary.metric.eq('ood_3.0')].to_dict('records'))


def display_rows():
    ep, summary = data()
    rows = []
    for _, r in summary[summary.forward.eq('shunt') & summary.rule.isin(RULES)
                         & summary.metric.str.startswith('ood_')].iterrows():
        rows.append(dict(panel='H', record='population severity mean', **r.to_dict()))
    for _, r in ep[ep.rule.isin(RULES)].iterrows():
        rows.append(dict(panel='I', record='population forward contrast seed',
                         seed=int(r.seed), forward=r.forward, rule=r.rule,
                         test_nmse=r['ood_3.0'], severity=3.))
    for _, r in summary[summary.rule.isin(RULES) & summary.metric.eq('ood_3.0')].iterrows():
        rows.append(dict(panel='I', record='population forward mean and interval', **r.to_dict()))
    return rows
