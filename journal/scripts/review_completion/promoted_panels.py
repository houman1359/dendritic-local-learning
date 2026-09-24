"""Plot existing checkpoint and extension evidence without refitting or reselection."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_canvas import COLORS, style_panel, LW_DATA, LW_ERR, LW_HAIR
from journal_style import DIV_CMAP
from review_completion.checkpoint_computation import interaction_component

J = Path(__file__).resolve().parents[2]
CHECKPOINT = J / 'source_data/checkpoint_computation'
EXTENSION = J / 'source_data/optional_extensions'
TUNING_COLORS = {'target': COLORS['mute'], 'unit_broadcast': COLORS['scalar'],
                 'hard_distal_unit_proximal': COLORS['shunting'], 'exact': COLORS['bp']}
TUNING_NAMES = {'target': 'Teacher', 'unit_broadcast': 'Broadcast',
                'hard_distal_unit_proximal': 'Hard distal gate', 'exact': 'Exact'}


def tuning_panel(ax, context, panel, rows, *, legend=False):
    table = pd.read_csv(CHECKPOINT / 'branch_tuning.csv')
    indices = np.random.default_rng(2026092121).integers(20, size=(10000, 20))
    style_panel(ax)
    for rule in ['unit_broadcast', 'hard_distal_unit_proximal', 'exact', 'target']:
        part = table[table.context.eq(context) & table.rule.eq(rule)]
        wide = part.pivot(index='seed', columns='z1', values='branch_contribution').sort_index()
        assert len(wide) == 20
        x = wide.columns.to_numpy(); values = wide.to_numpy(); mean = values.mean(0)
        low, high = np.quantile(values[indices].mean(1), [.025, .975], axis=0)
        dash = (0, (3, 2)) if rule == 'target' else ((0, (1, 2)) if rule == 'exact' else '-')
        ax.plot(x, mean, color=TUNING_COLORS[rule], ls=dash, lw=LW_DATA,
                zorder=5 if rule == 'target' else 3)
        ax.fill_between(x, low, high, color=TUNING_COLORS[rule], alpha=.12, lw=0)
        rows.extend(dict(panel=panel, record='tuning mean', rule=rule, context=context,
                         z1=float(a), mean=float(b), ci_low=float(lo), ci_high=float(hi),
                         n=20, quantity='selected branch contribution')
                    for a, b, lo, hi in zip(x, mean, low, high))
    ax.set_xlim(-2, 2); ax.set_xticks([-2, 0, 2]); ax.set_ylim(0, .55); ax.set_yticks([0, .25, .5])
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Contribution to soma' if context == 0 else '')
    ax.text(.5, 1.005, 'Selected left branch' if context == 0 else 'Selected right branch',
            ha='center', transform=ax.transAxes, fontsize=7)
    if legend:
        # The model-rule colors already appear in C; these patterns identify the
        # teacher and exact reference where the learned curves coincide.
        handles = [Line2D([], [], color=TUNING_COLORS[r], ls=(0, (3, 2)) if r == 'target'
                          else ((0, (1, 2)) if r == 'exact' else '-'), lw=LW_DATA)
                   for r in ('target', 'exact')]
        ax.legend(handles, ['Teacher', 'Exact'], frameon=False, fontsize=7,
                  loc='lower right', handlelength=1.5, handletextpad=.35)


def response_matrices():
    source = pd.read_csv(CHECKPOINT / 'population_surfaces.csv')
    grid = np.sort(source.z1.unique()); weights = np.ones(len(grid))
    weights[[0, -1]] = .5; weights /= weights.sum()
    z1, z2 = np.meshgrid(grid, grid, indexing='ij')
    matrices = {'target': .5 * (np.tanh(z1) + np.tanh(z2)) + .25 * np.tanh(z1) * np.tanh(z2)}
    for rule in ['resistance', 'derivative', 'exact']:
        part = source[source.rule.eq(rule)]
        assert part.seed.nunique() == 20 and part.context.nunique() == 4
        matrices[rule] = part.groupby(['z1', 'z2']).prediction.mean().unstack().loc[grid, grid].to_numpy()
    return grid, weights, matrices


def interaction_maps(host, panel, rows):
    """Three maps in one panel; a common scale and equal feature-axis units."""
    host.set_axis_off()
    grid, weights, matrices = response_matrices()
    for k, (rule, name) in enumerate([('target', 'Target'), ('resistance', 'Resistance h'),
                                     ('derivative', 'Augmented hf′')]):
        ax = host.inset_axes([.015 + .315 * k, .24, .255, .60])
        style_panel(ax)
        matrix = interaction_component(matrices[rule], weights)
        im = ax.pcolormesh(grid, grid, matrix.T, cmap=DIV_CMAP, vmin=-.25, vmax=.25,
                           shading='nearest', rasterized=False)
        ax.set_aspect('equal'); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        if k: ax.set_yticklabels([])
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        ax.set_title(name, fontsize=7, pad=5)
        rows.extend(dict(panel=panel, record='interaction map', rule=rule,
                         z1=float(grid[i]), z2=float(grid[j]), value=float(matrix[i, j]),
                         quantity='interaction', n=20)
                    for i in range(len(grid)) for j in range(len(grid)))
    host.text(.45, .08, 'Feature 1', ha='center', fontsize=8, transform=host.transAxes)
    host.text(-.10, .54, 'Feature 2', va='center', rotation=90, fontsize=8, transform=host.transAxes)
    cbax = host.inset_axes([.965, .36, .018, .35])
    cb = host.figure.colorbar(im, cax=cbax, ticks=[-.25, 0, .25])
    cb.ax.tick_params(labelsize=7, width=LW_HAIR, length=2, pad=2)
    # Review pass 2026-09-23: the heading is stated in the legend.


def extension_mark(ax, ep, summary, study, arm, policy, x, panel, rows, color):
    group = ep[ep.study.eq(study) & ep.arm.eq(arm) & ep.policy.eq(policy)].sort_values('seed')
    ref = summary[summary.study.eq(study) & summary.arm.eq(arm) & summary.policy.eq(policy)
                  & summary.metric.eq('test_nmse')]
    assert len(group) == 20 and len(ref) == 1
    ref = ref.iloc[0]
    assert abs(group.test_nmse.mean() - ref['mean']) < 1e-12
    ax.errorbar(x, ref['mean'], yerr=[[ref['mean']-ref.ci95_low], [ref.ci95_high-ref['mean']]],
                fmt='D', mfc='white', mec=color, ecolor=color, ms=4, lw=LW_ERR,
                capsize=2, zorder=3)
    ax.scatter(x + np.linspace(-.09, .09, 20), group.test_nmse,
               s=5, color=color, alpha=.55, lw=0, zorder=4)
    rows.append(dict(panel=panel, record='extension mean', study=study, arm=arm,
                     policy=policy, mean=ref['mean'], ci_low=ref.ci95_low,
                     ci_high=ref.ci95_high, n=20))
    rows.extend(dict(panel=panel, record='extension seed', study=study, arm=arm,
                     policy=policy, seed=int(r.seed), value=float(r.test_nmse), rate=float(r.rate))
                for r in group.itertuples())


def proxy_panel(ax, panel, rows, log_ticks):
    ep = pd.read_csv(EXTENSION/'endpoints.csv'); summary = pd.read_csv(EXTENSION/'summary.csv')
    arms = ['derivative', 'bins2', 'bins4', 'noise05', 'shuffle_bins4', 'shuffle_noise05']
    labels = ['Parent\nslope', '2 bins', '4 bins', 'Noisy\nslope', 'Shuffled\nbins', 'Shuffled\nnoisy']
    style_panel(ax); log_ticks(ax, [1e-5, 1e-3, 1e-1]); ax.set_ylim(1e-5, .16)
    for x, arm in enumerate(arms):
        color = COLORS['highlight'] if arm.startswith('shuffle') else COLORS['additive']
        extension_mark(ax, ep, summary, 'proxy', arm, 'common', x, panel, rows, color)
    ref = summary[summary.study.eq('proxy') & summary.arm.eq('resistance')
                  & summary.policy.eq('common') & summary.metric.eq('test_nmse')].iloc[0]
    ax.axhline(ref['mean'], color=COLORS['shunting'], ls='--', lw=LW_DATA)
    ax.text(.03, .92, 'Resistance gate h', color=COLORS['shunting'], fontsize=7,
            va='bottom', transform=ax.transAxes)
    rows.append(dict(panel=panel, record='extension reference', study='proxy', arm='resistance',
                     policy='common', mean=ref['mean'], n=20))
    ax.set_xticks(range(len(arms)), labels); ax.set_xlim(-.5, len(arms)-.5)
    ax.set_ylabel('Ordinary-test NMSE'); ax.tick_params(axis='x', labelsize=7)


def routing_panel(ax, panel, rows, log_ticks):
    ep = pd.read_csv(EXTENSION/'endpoints.csv'); summary = pd.read_csv(EXTENSION/'summary.csv')
    arms = ['oracle_augmented', 'learned_local_augmented', 'learned_local_resistance', 'uniform_augmented']
    labels = ['Supplied\nhf′', 'Learned\nhf′', 'Learned\nh', 'Uniform\nhf′']
    colors = [COLORS['additive'], COLORS['additive'], COLORS['shunting'], COLORS['mute']]
    style_panel(ax); log_ticks(ax, [1e-5, 1e-3, 1e-1, 1]); ax.set_ylim(5e-6, 1.5)
    for x, (arm, color) in enumerate(zip(arms, colors)):
        extension_mark(ax, ep, summary, 'routing', arm, 'selected', x, panel, rows, color)
    ax.set_xticks(range(len(arms)), labels); ax.set_xlim(-.5, len(arms)-.5)
    ax.set_ylabel('Ordinary-test NMSE'); ax.tick_params(axis='x', labelsize=7)
