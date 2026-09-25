"""Figure 7 stopping-extension curves, with every paired seed retained."""
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from journal_style import COLORS, LW_DATA, LW_REF, LW_HAIR, PT_BASE, MARKER_MS

ARMS = (
    ('exact_autograd_bp_recipe', 3, 'ink', (0, (4.2, 2.0)), 'exact BP', True),
    ('broadcast_autograd_bp_recipe', 3, 'ink', (0, (1.2, 1.6)), 'broadcast (BP)', False),
    ('path_transport', 3, 'bp', '-', 'exact path', True),
    ('per_soma_shared', 3, 'scalar', '-', 'shared soma', True),
    ('broadcast_autograd_localca_recipe', 3, 'scalar', (0, (1.2, 1.6)), 'broadcast (local)', False),
    ('exact_autograd_bp_recipe', 1, 'point_mlp', '-', 'exact BP (D1)', True),
)


def epoch_axis(ax, maximum, *, label_space=False):
    # The linear first 180 epochs preserve the early learning comparison;
    # subsequent logarithmic spacing shows the complete stopping interval.
    ax.set_xscale('symlog', linthresh=180, linscale=0.65)
    if label_space:
        tr = ax.xaxis.get_transform()
        right = float(tr.inverted().transform(tr.transform(maximum) / .64))
    else:
        right = maximum * 1.08
    ax.set_xlim(0, right)
    ticks = [0, 180, 600, maximum]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(['0', '180', '600', f'{maximum:,}']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines['bottom'].set_bounds(0, maximum)
    ax.set_xlabel('Epoch (log after 180)', fontsize=PT_BASE)
    ax.axvline(180, color=COLORS['mute'], lw=LW_REF, ls=(0, (3, 3)), zorder=.5)


def condition_curves(ax, curves, stopping, metric, panel, rows):
    scale = 100. if metric == 'test_accuracy' else 1.
    maximum = int(curves.epoch.max())
    epoch_axis(ax, maximum, label_space=True)
    if metric == 'test_accuracy':
        ax.set_ylim(43, 105)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        ax.set_ylabel('Test accuracy (%)')
        ax.spines['left'].set_bounds(50, 100)
        ax.axhline(50, color=COLORS['mute'], lw=LW_REF, ls=(0, (3, 3)), zorder=.5)
    else:
        ax.set_yscale('log')
        minimum = max(float(curves.loc[curves.metric.eq(metric), 'ci95_low'].min()) * .7, 1e-6)
        ax.set_ylim(minimum, .95)
        ticks = [v for v in [.0001, .001, .01, .1, .7] if v >= minimum]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel('Best validation loss')
        ax.spines['left'].set_bounds(ticks[0], ticks[-1])
    ends = {}
    for arm, depth, color, style, label, band in ARMS:
        p = curves[curves.arm.eq(arm) & curves.depth.eq(depth) & curves.metric.eq(metric)].sort_values('epoch')
        assert (p.n_seeds == 10).all() and len(p)
        x = p.epoch.to_numpy(); y = scale * p['mean'].to_numpy()
        if band:
            ax.fill_between(x, scale*p.ci95_low, scale*p.ci95_high,
                            color=COLORS[color], alpha=.12, lw=0, zorder=1)
        ax.plot(x, y, color=COLORS[color], ls=style,
                lw=LW_REF if depth == 1 else LW_DATA, zorder=2)
        ends[label] = (float(y[-1]), COLORS[color])
        rows.extend(dict(panel=panel, record='curve summary', arm=arm, depth=depth,
            metric=metric, epoch=int(r.epoch), mean=float(r.mean),
            ci95_low=float(r.ci95_low), ci95_high=float(r.ci95_high),
            n_seeds=10, plotted_mean=scale*float(r.mean),
            plotted_ci95_low=scale*float(r.ci95_low), plotted_ci95_high=scale*float(r.ci95_high),
            band_drawn=band) for r in p.itertuples())
    if metric == 'best_validation_loss':
        d1 = stopping[stopping.arm.eq('exact_autograd_bp_recipe') & stopping.depth.eq(1)]
        track = curves[curves.arm.eq('exact_autograd_bp_recipe') & curves.depth.eq(1)
                       & curves.metric.eq(metric)].set_index('epoch')['mean']
        for r in d1.itertuples():
            value = float(track.loc[r.epochs_run])
            ax.plot(r.epochs_run, value, marker='o', ls='none', ms=MARKER_MS*.8,
                    mfc='white', mec=COLORS['point_mlp'], mew=LW_HAIR, zorder=3)
            rows.append(dict(panel=panel, record='stopping marker', arm=r.arm,
                depth=1, seed=int(r.seed), metric='stopping_epoch',
                epoch=int(r.epochs_run), mean=value, n_seeds=10))
    # Labels occupy a reserved column beyond the last observed epoch.
    # The two broadcast variants use their parent coordinate's colour.
    if metric == 'test_accuracy':
        # The D3 endpoints coincide at this scale; the shared style key is F.
        seats = {'exact BP': .92, 'exact BP (D1)': .25}
    else:
        seats = {'exact BP (D1)': .90, 'shared soma': .61,
                 'broadcast (local)': .48, 'exact path': .32,
                 'broadcast (BP)': .17, 'exact BP': .045}
    for label, seat in seats.items():
        value, color = ends[label]
        printed = {'exact BP':'BP D3', 'broadcast (BP)':'broadcast',
                   'broadcast (local)':'broadcast', 'exact BP (D1)':'BP D1'}.get(label, label)
        if metric == 'test_accuracy':
            printed = 'D3: all rules' if label == 'exact BP' else 'D1: BP'
        endpoint = ax.transAxes.inverted().transform(ax.transData.transform((maximum, value)))
        # Keep every connector inside the narrow gutter before the label
        # column, so the coincident accuracy curves cannot cross other names.
        ax.plot([endpoint[0], .67, .678], [endpoint[1], seat, seat],
                transform=ax.transAxes, color=COLORS['mute'], lw=LW_HAIR,
                clip_on=False, zorder=3)
        ax.text(.69, seat, printed, transform=ax.transAxes,
                ha='left', va='center', fontsize=PT_BASE, color=color, zorder=4)
    return ax


def paired_curve(ax, paired, metric, panel, rows):
    p = paired[paired.metric.eq(metric)].sort_values('epoch')
    assert (p.n_seeds == 10).all()
    maximum = int(p.epoch.max())
    epoch_axis(ax, maximum)
    lo = min(0., float(p.ci95_low.min())); hi = max(0., float(p.ci95_high.max()))
    padding = max((hi-lo)*.12, .001)
    ax.set_ylim(lo-padding, hi+padding)
    ax.set_ylabel('Exact − shared soma (pp)' if metric == 'test_accuracy'
                  else 'Test cross-entropy difference\nexact − shared soma (nats)', fontsize=PT_BASE)
    ax.axhline(0, color=COLORS['mute'], lw=LW_REF, ls=(0, (3, 3)), zorder=.5)
    ax.fill_between(p.epoch, p.ci95_low, p.ci95_high, color=COLORS['mute'], alpha=.15, lw=0, zorder=1)
    ax.plot(p.epoch, p['mean'], color=COLORS['bp'], lw=LW_DATA, zorder=2)
    selected = p[p.epoch.isin([180, 600, maximum])]
    ax.plot(selected.epoch, selected['mean'], ls='none', marker='o', ms=MARKER_MS,
            mfc=COLORS['bp'], mec='white', mew=LW_HAIR, zorder=3)
    if metric == 'test_accuracy':
        last = p.iloc[-1]
        label = (f"{last['mean']:+.2f} pp\n"
                 f"95% CI {last.ci95_low:.2f}–{last.ci95_high:.2f}")
        ax.annotate(label, xy=(maximum, last['mean']),
                    xytext=(.98, .36), textcoords='axes fraction',
                    ha='right', va='bottom', fontsize=PT_BASE, color=COLORS['bp'],
                    arrowprops=dict(arrowstyle='-', color=COLORS['mute'], lw=LW_HAIR,
                                    shrinkA=2, shrinkB=4))
    rows.extend(dict(panel=panel, record='paired curve summary', metric=metric,
        epoch=int(r.epoch), mean=float(r.mean), ci95_low=float(r.ci95_low),
        ci95_high=float(r.ci95_high), n_seeds=10,
        positive_seeds=int(r.positive_seeds), negative_seeds=int(r.negative_seeds))
        for r in p.itertuples())
    return ax
