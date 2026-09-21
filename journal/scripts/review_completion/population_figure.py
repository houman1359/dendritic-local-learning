"""Main Figure 6 (fig:conductancepopulation).

Redesign 2026-09-21 (review completion): five panels on a three-row canvas.
A, the task schematic with the terminal learning-signal delivery key;
B, assignment under distractor stress with direct end labels (coincident
curves keep their line patterns and are named in the caption);
C, one shared-axis dot strip: ordinary, stress-3 and common-rate pairs for
broadcast and the resistance gate, then the tonic and current forward
controls at severity three;
D, within- versus cross-context terminal alignment at the archived
exact-trained checkpoints (a controlled diagnostic of recipient assignment,
not a description of the Adam trajectory);
E, the consolidated rescue: selected rates, common rate 0.03 and the
separable-target control for the same five rules, with brackets on the two
predefined primary contrasts only.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_canvas import (NativeCanvas, Margins, COLORS, style_panel,  # noqa: E402
                           LW_HAIR, LW_EDGE, LW_REF, LW_ERR, LW_DATA)

J = Path(__file__).resolve().parents[2]
D = J / 'source_data/curated_publication'
OUTPUT = J / 'figures/main/figure_06.pdf'
INPUTS = ['inhibitory_selection_endpoints.csv', 'inhibitory_selection_summary.csv',
          'inhibitory_selection_rate_sensitivity.csv', 'inhibitory_rescue_endpoints.csv',
          'inhibitory_rescue_common_endpoints.csv', 'inhibitory_rescue_contrasts.csv',
          'nonlinear_separable_endpoints.csv', 'context_alignment_seeds.csv',
          'context_alignment_summary.csv']
PT = 7.0
C = {'exact': COLORS['bp'], 'broadcast': COLORS['scalar'], 'resistance': COLORS['shunting'],
     'swapped': COLORS['point_mlp'], 'uniform_rms': COLORS['mute'],
     'derivative': COLORS['additive'], 'shuffled_derivative': COLORS['highlight']}
NAME = {'exact': 'Exact', 'broadcast': 'Broadcast', 'resistance': 'Resistance gate h',
        'derivative': 'Augmented h f′', 'shuffled_derivative': 'Shuffled f′',
        'swapped': 'Wrong branch', 'uniform_rms': 'Uniform RMS'}
DASH = (0, (2.6, 1.6))
DOT = (0, (0.9, 1.5))
ROWS = []
SCOPE = {
    'main_6A': 'Schematic of the supplied cue, four two-feature streams, one [4,2] DendriNet '
               'tree of the sixteen-neuron population, the linear readout and the terminal '
               'learning-signal delivery key; no measured series.',
    'main_6B': 'Separable target, shunting forward: all five spatial-assignment rules under '
               'distractor stress; original paired seed cohort, development-selected rates.',
    'main_6C': 'Broadcast and resistance at ordinary test, severity three and common rate 0.1 '
               '(same seed blocks), then exact, broadcast and resistance at severity three '
               'under tonic and current forward controls.',
    'main_6D': 'Terminal within- and cross-context alignment of each delivered rule with the '
               'exact gradient at the 20 archived exact-trained separable-shunting checkpoints; '
               'checkpoint-only diagnostic, no training.',
    'main_6E': 'Nonlinear-parent interaction target: original-bound Adam rescue at separately '
               'selected rates (twenty new paired seeds), the same seeds at common Adam rate '
               '0.03, and the same seeds with the target interaction removed.'}
SOURCES = {'main_6A': [], **{k: ['source_data/curated_publication/' + n for n in INPUTS]
                             for k in ('main_6B', 'main_6C', 'main_6D', 'main_6E')}}


def chain(ax, x, y, parts, *, color, size=PT, drop=1.6, ha='left', va='center', zorder=5):
    """Plain-text spans chained on one baseline: ('text', level) with level -1 for a
    subscript, +1 for a superscript and 0 for the line.  Every span is a token size,
    so the figure contract's ban on mathtext sub/superscripts is respected."""
    prev, level = None, 0
    for text, lvl in parts:
        if prev is None:
            prev = ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, zorder=zorder)
        else:
            rise = {-1: -drop, 0: 0.0, 1: 2.6}
            dy = rise[lvl] - rise[level]
            prev = ax.annotate(text, xy=(1.0, 0.0), xycoords=prev, xytext=(0.5, dy),
                               textcoords='offset points', fontsize=size, color=color,
                               ha='left', va='baseline', zorder=zorder, annotation_clip=False)
        level = lvl
    return prev


def boot(v, seed=2026092103):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(len(v), size=(10000, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def mark(ax, x, values, rule, panel, *, marker='D', filled=False, seed_alpha=.45, **meta):
    """Mean marker with 95% seed-bootstrap whiskers; the seeds are drawn ON TOP."""
    values = values.sort_values('seed')
    v = values['value'].to_numpy(dtype=float)
    m, lo, hi = boot(v)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=4.2,
                mfc=C[rule] if filled else 'white', mec=C[rule], color=C[rule],
                elinewidth=LW_ERR, capsize=2, zorder=3)
    ax.scatter(x + np.linspace(-.055, .055, len(v)), v, s=4.5, color=C[rule],
               alpha=seed_alpha, lw=0, zorder=4)
    ROWS.append(dict(panel=panel, record='mean and bootstrap interval', rule=rule,
                     mean=m, ci_low=lo, ci_high=hi, n=len(v), **meta))
    ROWS.extend(dict(panel=panel, record='seed outcome', rule=rule, seed=int(r.seed),
                     value=float(r.value), **meta) for r in values.itertuples())
    return m, lo, hi


def log_axis(ax, log_ticks, ticks=(1e-5, 1e-3, 1e-1, 1.), ylim=(1e-5, 1.5), label='NMSE'):
    style_panel(ax)
    ax.set_ylabel(label, fontsize=8)
    log_ticks(ax, list(ticks))
    ax.set_ylim(*ylim)
    ax.tick_params(axis='x', labelsize=PT, pad=3)


# ── A: task schematic and delivery key ─────────────────────────────────────
def task_panel(ax):
    ink, exc, inh = COLORS['ink'], COLORS['exc'], COLORS['inh']
    sel, edge = C['resistance'], COLORS['edge']
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    soma = (0.235, 0.14)
    px = [0.065, 0.175, 0.285, 0.395]
    py, ty = 0.58, 0.88
    for b, x in enumerate(px):
        col = sel if b == 1 else edge
        for off in (-0.032, 0.032):
            ax.plot([x + off, x], [ty, py], color=col, lw=LW_EDGE, zorder=2)
            ax.scatter([x + off], [ty], s=11, color=exc, lw=0, zorder=4)
        ax.plot([x, soma[0]], [py, soma[1]], color=col, lw=LW_EDGE, zorder=2)
        ax.scatter([x], [py], s=20, facecolor='white', edgecolor=col, lw=LW_EDGE, zorder=4)
        if b != 1:
            ax.scatter([x + 0.03], [py + 0.075], s=11, color=inh, lw=0, zorder=5)
    ax.scatter([soma[0]], [soma[1]], s=30, facecolor=COLORS['soma'], edgecolor=ink,
               lw=LW_EDGE, zorder=4)
    # the supplied cue: one carmine square, dotted to the nearest inhibitory contact
    ax.scatter([0.02], [0.665], s=14, marker='s', facecolor='white', edgecolor=inh,
               lw=LW_EDGE, zorder=5)
    ax.plot([0.032, px[0] + 0.03 - 0.012], [0.665, py + 0.075], color=inh, lw=LW_HAIR,
            ls=DOT, zorder=1)
    ax.text(0.02, 0.585, 'cue', fontsize=PT, color=inh, ha='center', va='top')
    ax.text(px[1] - 0.02, 0.975, 'selected', fontsize=PT, color=sel, ha='center', va='center')
    ax.text(0.375, 0.975, 'inputs exp(±z)', fontsize=PT, color=ink, ha='center', va='center')
    ax.text(0.31, 0.505, 'inhibited', fontsize=PT, color=inh, ha='left', va='center')
    ax.annotate('', xy=(0.43, soma[1]), xytext=(0.265, soma[1]),
                arrowprops=dict(arrowstyle='->', lw=LW_EDGE, color=ink, shrinkA=0, shrinkB=0))
    ax.text(0.445, soma[1], 'ŷ', fontsize=PT, color=ink, ha='left', va='center')
    ax.text(0.345, 0.03, '16 neurons', fontsize=PT, color=COLORS['mute'], ha='center',
            va='bottom')
    # delivery key
    x0 = 0.52
    ax.text(x0, 0.975, 'Terminal learning-signal delivery', fontsize=PT, color=ink,
            ha='left', va='center')
    rows = [('broadcast', 'Broadcast', [('δ', 0), ('u', -1)]),
            ('resistance', 'Resistance gate h', [('δ', 0), ('u', -1), (' h', 0)]),
            ('derivative', 'Augmented h f′', [('δ', 0), ('u', -1), (' h f′', 0), ('p', -1)]),
            ('exact', 'Exact', [('δ', 0), ('u', -1), (' κ h f′', 0), ('p', -1)])]
    for k, (rule, name, formula) in enumerate(rows):
        y = 0.84 - 0.115 * k
        ax.text(x0, y, name, fontsize=PT, color=C[rule], ha='left', va='center')
        chain(ax, 0.82, y, formula, color=ink)
    notes = [[('h: relative parent resistance', 0)], [('f′', 0), ('p', -1), (': parent output slope', 0)],
             [('κ: fixed across examples at given', 0)], [('parameters; changes with learning', 0)]]
    for k, note in enumerate(notes):
        chain(ax, x0, 0.335 - 0.095 * k, note, color=COLORS['mute'])


# ── B: stress curves with direct end labels ────────────────────────────────
def stress_panel(ax, summary, log_ticks):
    sev = [1., 1.5, 2., 2.5, 3.]
    style = {'exact': ('-', LW_DATA, None), 'resistance': (DASH, LW_DATA, None),
             'broadcast': ('-', LW_DATA, None), 'uniform_rms': (DOT, LW_DATA, None),
             'swapped': ('-', LW_DATA, 'o')}
    ends = {}
    for rule in ['broadcast', 'uniform_rms', 'exact', 'resistance', 'swapped']:
        part = summary[summary.variant.eq('separable') & summary.forward.eq('shunt')
                       & summary.rule.eq(rule)].set_index('metric').loc[[f'ood_{s}' for s in sev]]
        ls, lw, marker = style[rule]
        ax.fill_between(sev, part.ci_low, part.ci_high, color=C[rule], alpha=.12, lw=0)
        ax.plot(sev, part['mean'], color=C[rule], lw=lw, ls=ls, marker=marker, ms=2.2,
                zorder=3 if rule in ('resistance', 'uniform_rms') else 2)
        ends[rule] = float(part['mean'].iloc[-1])
        ROWS.extend(dict(panel='B', record='stress mean', rule=rule, severity=s,
                         mean=r['mean'], ci_low=r.ci_low, ci_high=r.ci_high, n=20)
                    for s, (_, r) in zip(sev, part.iterrows()))
    log_axis(ax, log_ticks)
    ax.set_xlim(.9, 3.95)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('Irrelevant-input severity', fontsize=8)

    def label(y, text, color, dy_pt, va):
        ax.annotate(text, xy=(3.0, y), xytext=(4.5, dy_pt), textcoords='offset points',
                    fontsize=PT, color=color, ha='left', va=va, annotation_clip=False,
                    zorder=6)
    label(ends['swapped'], 'Wrong branch', C['swapped'], 1.5, 'bottom')
    label(ends['broadcast'], 'Broadcast', C['broadcast'], 1.0, 'bottom')
    label(ends['broadcast'], 'Uniform RMS', C['uniform_rms'], -1.0, 'top')
    label(ends['exact'], 'Exact', C['exact'], 1.0, 'bottom')
    label(ends['exact'], 'Resistance gate h', C['resistance'], -1.0, 'top')
    return ends


# ── C: ordinary / stress / common-rate pairs and forward controls ──────────
def strip_panel(ax, original, rate, log_ticks):
    groups = [('Ordinary', 'shunt', 'test_nmse', ['broadcast', 'resistance'], 'selected', 0.0),
              ('Stress 3', 'shunt', 'ood_3.0', ['exact', 'broadcast', 'resistance'], 'selected', 1.0),
              ('Stress 3\nrate 0.1', 'shunt', 'ood_3.0', ['broadcast', 'resistance'], 'common 0.1', 2.0),
              ('Tonic\nstress 3', 'tonic', 'ood_3.0', ['exact', 'broadcast', 'resistance'], 'selected', 3.35),
              ('Current\nstress 3', 'current', 'ood_3.0', ['exact', 'broadcast', 'resistance'], 'selected', 4.35)]
    log_axis(ax, log_ticks)
    xt, xl = [], []
    for label, forward, metric, rules, policy, x in groups:
        offs = [-.17, .17] if len(rules) == 2 else [-.27, 0., .27]
        stats = {}
        for off, rule in zip(offs, rules):
            g = original[original.forward.eq(forward) & original.rule.eq(rule)]
            if policy == 'common 0.1' and rule == 'broadcast':
                g = rate[rate.rule.eq(rule) & rate.rate.eq(.1)].rename(columns={'ood_3': 'ood_3.0'})
            stats[rule] = mark(ax, x + off, g.assign(value=g[metric]), rule, 'C',
                               forward=forward, metric=metric, rate_scope=policy,
                               condition=label.replace('\n', ' '))
        if forward == 'shunt':
            ratio = stats['broadcast'][0] / stats['resistance'][0]
            top = max(s[2] for s in stats.values())
            ax.text(x, top * 2.6, f'{ratio:.0f}×' if ratio >= 10 else f'{ratio:.1f}×',
                    fontsize=PT, color=COLORS['ink'], ha='center', va='bottom', zorder=6)
            ROWS.append(dict(panel='C', record='broadcast/gate ratio of means',
                             condition=label.replace('\n', ' '), rate_scope=policy,
                             value=float(ratio)))
        xt.append(x)
        xl.append(label)
    ax.set_xticks(xt, xl)
    ax.set_xlim(-.55, 4.9)
    ax.text(.285, 1.035, "Shunting", transform=ax.transAxes, ha="center", va="center", fontsize=PT, color=COLORS["mute"])
    ax.text(.805, 1.035, "Forward controls", transform=ax.transAxes, ha="center", va="center", fontsize=PT, color=COLORS["mute"])
    ax.axvline(2.72, color=COLORS["grid"], lw=LW_HAIR, ymin=.06, ymax=.92)


# ── D: context alignment at archived checkpoints ───────────────────────────
def alignment_panel(ax, seeds, summary):
    style_panel(ax)
    sel = ((seeds.variant.eq('separable')) & seeds.forward.eq('shunt')
           & seeds.trained_rule.eq('exact') & seeds.block.eq('terminal'))
    seeds = seeds[sel]
    order = ['exact', 'broadcast', 'resistance', 'swapped', 'uniform_rms']
    for k, rule in enumerate(order):
        for relation, filled, dx in (('within', True, -.13), ('cross', False, .13)):
            g = seeds[seeds.delivered_rule.eq(rule) & seeds.relation.eq(relation)]
            values = g.assign(value=g.cosine)[['seed', 'value']]
            m, lo, hi = mark(ax, k + dx, values, rule, 'D', marker='o', filled=filled,
                             relation=relation, block='terminal', trained_rule='exact',
                             forward='shunt', variant='separable')
            ref = summary[summary.variant.eq('separable') & summary.forward.eq('shunt')
                          & summary.trained_rule.eq('exact') & summary.block.eq('terminal')
                          & summary.delivered_rule.eq(rule) & summary.relation.eq(relation)]
            assert len(ref) == 1 and abs(float(ref['mean'].iloc[0]) - m) < 1e-9
    ax.axhline(0, color=COLORS['mute'], lw=LW_HAIR, ls=DOT, zorder=1)
    ax.set_ylim(-.12, 1.08)
    ax.set_yticks([0, .5, 1.])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_ylabel('Alignment (cosine)', fontsize=8)
    ax.set_xticks(range(5), ['Exact', 'Broadcast', 'Resistance\ngate h', 'Wrong\nbranch', 'Uniform\nRMS'])
    ax.set_xlim(-.55, 4.55)
    ax.tick_params(axis='x', labelsize=PT, pad=3)
    handles = [Line2D([], [], marker='o', ms=4.2, mfc=COLORS['ink'], mec=COLORS['ink'], lw=0),
               Line2D([], [], marker='o', ms=4.2, mfc='white', mec=COLORS['ink'], lw=0)]
    ax.legend(handles, ['within context', 'cross context'], loc='upper right', frameon=False,
              fontsize=PT, handletextpad=.4, borderaxespad=.2, labelspacing=.3)


# ── E: consolidated rescue with the two primary contrasts ──────────────────
def rescue_panel(ax, rescue, common, separable, contrasts, log_ticks):
    order = ['exact', 'broadcast', 'resistance', 'derivative', 'shuffled_derivative']
    blocks = [('selected rates', rescue[rescue.optimizer.eq('adam') & rescue.bound.eq(9)], 'D', -.27),
              ('common rate 0.03', common, 'o', 0.),
              ('separable target', separable, 's', .27)]
    xs = {}
    for block, frame, marker, off in blocks:
        for k, rule in enumerate(order):
            g = frame[frame.rule.eq(rule)]
            assert len(g) == 20, (block, rule, len(g))
            mark(ax, k + off, g.assign(value=g.test_nmse), rule, 'E', marker=marker,
                 metric='test_nmse', condition=block)
            xs[(block, rule)] = k + off
    log_axis(ax, log_ticks, ylim=(1e-5, 4.0), label='Ordinary-test NMSE')
    ax.set_xticks(range(5), [NAME[r] for r in order])
    ax.set_xlim(-.6, 4.6)
    # the two predefined primary contrasts, Holm-adjusted
    primary = contrasts[contrasts.primary.eq(True)]
    assert len(primary) == 2
    heights = {'resistance': .16, 'shuffled_derivative': .48}
    for r in primary.itertuples():
        x1, x2 = xs[('selected rates', r.left)], xs[('selected rates', r.right)]
        y = heights[r.left]
        ax.plot([x1, x1, x2, x2], [y / 1.25, y, y, y / 1.25], color=COLORS['ink'], lw=LW_HAIR,
                zorder=5, solid_capstyle='butt')
        p = f'{r.holm_p:.1e}'.split('e')
        chain(ax, (x1 + x2) / 2 - .62, y * 1.4, [(f'P = {p[0]} × 10', 0), (f'−{abs(int(p[1]))}', 1), (' (Holm)', 0)],
              color=COLORS['ink'], va='bottom', zorder=6)
        ROWS.append(dict(panel='E', record='primary contrast', rule=r.left,
                         comparison=f'{r.left} minus {r.right}', mean=float(r.mean),
                         ci_low=float(r.ci_low), ci_high=float(r.ci_high), n=int(r.n),
                         positive=int(r.positive), holm_p=float(r.holm_p),
                         condition='selected rates'))
    handles = [Line2D([], [], marker=m, ms=4.2, mfc='white', mec=COLORS['ink'], lw=0)
               for m in 'Dos']
    ax.legend(handles, ['Selected rates', 'Common rate 0.03', 'No interaction; same tanh parents; rate 0.03'],
              loc='upper left', ncol=3, frameon=False, fontsize=PT, handletextpad=.4,
              columnspacing=1.2, borderaxespad=.2)


def build(log_ticks):
    ROWS.clear()
    ep = pd.read_csv(D / INPUTS[0])
    summary = pd.read_csv(D / INPUTS[1])
    rate = pd.read_csv(D / INPUTS[2])
    rescue = pd.read_csv(D / INPUTS[3])
    common = pd.read_csv(D / INPUTS[4])
    contrasts = pd.read_csv(D / INPUTS[5])
    separable = pd.read_csv(D / INPUTS[6])
    align_seeds = pd.read_csv(D / INPUTS[7])
    align_summary = pd.read_csv(D / INPUTS[8])
    original = ep[ep.variant.eq('separable')]

    c = NativeCanvas(478 / 72, 3, row_weights=[122, 104, 132], hgutter_pt=30, vgutter_pt=30,
                     margins=Margins(left=40, right=6, top=24, bottom=36))
    a = c.panel('A', 0, 0, 6, letter='A', schematic=True, lock=False)
    task_panel(a)
    b = c.panel('B', 0, 6, 6, letter='B', grid='none')
    stress_panel(b, summary, log_ticks)
    s = c.panel('C', 1, 0, 6, letter='C', grid='none')
    strip_panel(s, original, rate, log_ticks)
    d = c.panel('D', 1, 6, 6, letter='D', grid='none')
    alignment_panel(d, align_seeds, align_summary)
    e = c.panel('E', 2, 0, 12, letter='E', grid='none')
    rescue_panel(e, rescue, common, separable, contrasts, log_ticks)

    # One left reserve for every data panel (one x0 per grid column), equal
    # right reserves within row 1 (one width per colspan), and B's right
    # reserve holds its direct end labels.
    locks = c.lock_reserves()
    left = max(locks[p][0] for p in 'BCDE')
    right = max(locks[p][1] for p in 'BCD')
    for p in 'BCD':
        c.declare_reserve(p, left=left, right=right)
    c.declare_reserve('E', left=left, right=max(locks['E'][1], 4.0))
    findings = list(c.save(OUTPUT, name='figure_06', dpi=180))
    from credit_first_figures.focused_provenance import publish
    publish(6, OUTPUT, ROWS, [D / n for n in INPUTS],
            [Path(__file__), J / 'scripts/conductance_local_gate/figure.py',
             J / 'scripts/figure_canvas.py'],
            {k: dict(sources=SOURCES[k], scope=SCOPE[k]) for k in SCOPE},
            emit_main=False, layout_findings=findings,
            notes='Original and rescue cohorts are distinct; the common-rate and '
                  'separable-target blocks reuse rescue seeds; D is a checkpoint-only '
                  'diagnostic of the original cohort. Rendering only.')
    return findings


def display_rows():
    return pd.read_csv(D / 'figure_06_plotted.csv').to_dict('records')


if __name__ == '__main__':
    sys.path.insert(0, str(J / 'scripts/conductance_local_gate'))
    from conductance_local_gate.figure import plain_log_ticks
    for finding in build(plain_log_ticks):
        print(finding)
