"""Figure 4 panel E: the paired noise sensitivity at both inherited rate policies.

2026-09-21: the former panels E (selected rates) and H (common rate) were the
same estimand twice; one panel now shows both policies side by side, encoded
by marker shape (diamond, selected; circle, common) so that panel F's
filled/open convention (endpoint versus validation-selected state) keeps its
own meaning.  Both markers are endpoints and therefore filled, as in F.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import COLORS, style_panel, LW_HAIR, LW_ERR

J = Path(__file__).resolve().parents[2]
POLICIES = (('selected', 'D', -.19), ('common', 'o', .19))
NOISES = ('fixed_absolute', 'noise_free', 'relative_matched')


def panel(ax, letter, rows):
    d = pd.read_csv(J / 'source_data/curated_publication/noise_controls_curves.csv')
    summary = pd.read_csv(J / 'source_data/curated_publication/noise_controls_contrasts.csv')
    style_panel(ax)
    colour = COLORS['additive']
    for policy, marker, dx in POLICIES:
        common = policy == 'common'
        part = d[(d.common_rate if common else d.selected_rate) & d.step.eq(16384)]
        stats = summary[summary.common_rate.eq(common) & summary.step.eq(16384)
                        & summary.metric.eq('population_nmse')]
        for i, noise in enumerate(NOISES):
            wide = part[part.noise.eq(noise)].pivot(index='seed', columns=['task', 'rule'],
                                                    values='population_nmse')
            v = ((wide['quartet', 'calibrated_broadcast'] - wide['quartet', 'exact'])
                 - (wide['matching', 'calibrated_broadcast'] - wide['matching', 'exact']))
            r = stats[stats.noise.eq(noise)].iloc[0]
            assert len(v) == 20 and abs(float(v.mean()) - float(r['mean'])) < 1e-6, (policy, noise)
            x = i + dx
            ax.errorbar(x, r['mean'], yerr=[[r['mean'] - r.ci_low], [r.ci_high - r['mean']]],
                        fmt=marker, color=colour, mfc=colour, mec=colour, ms=4,
                        elinewidth=LW_ERR, capsize=2, zorder=3)
            ax.scatter(x + np.linspace(-.075, .075, len(v)), v, s=5, lw=0, color=colour,
                       alpha=.3, zorder=4)
            rows.append(dict(panel=letter, record='summary',
                             series='quartic-minus-pairwise calibrated-minus-exact',
                             noise=noise, common_rate=common, rate_policy=policy,
                             mean=r['mean'], ci_low=r.ci_low, ci_high=r.ci_high, n_seeds=20,
                             endpoint='16384 updates; clean complete domain'))
            rows.extend(dict(panel=letter, record='paired seed difference',
                             series='paired seed contrast', noise=noise, common_rate=common,
                             rate_policy=policy, seed=int(seed), value=float(value))
                        for seed, value in v.items())
    ax.axhline(0, color=COLORS['mute'], ls=(0, (2.2, 1.8)), lw=LW_HAIR)
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(-.08, 1.65)
    ax.set_yticks([0, .5, 1., 1.5])
    ax.set_xticks([0, 1, 2], ['Fixed\nabsolute', 'Noise\nfree', 'Relative\nmatched'])
    ax.set_ylabel('Interaction deficit', fontsize=8)
    ax.tick_params(labelsize=7)
