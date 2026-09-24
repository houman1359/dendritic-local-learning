"""Supplementary Figure S22: diagnostic controls and full learned responses.

2026-09-23 clarity pass (analysis/figure_visual_review_20260910/review_20260923/
si_pass/ledger/checkpoint_computation.md): the zero-rule labels 'zero difference'
(A) and 'no cancellation' (B), drawn by the shared panel functions, are removed
here after the call (the shared modules stay untouched); the legend states both
rules.  D's 'Broadcast' and 'Resistance gate h' tick labels, which ran within
1 pt of each other, are set 2 pt apart from their ticks.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conductance_local_gate'))
from figure_canvas import NativeCanvas,Margins,style_panel,LW_HAIR
from journal_style import DIV_CMAP
from review_completion.promoted_panels import response_matrices
from conductance_local_gate import figure as gate
from review_completion import population_figure as pop
J=Path(__file__).resolve().parents[2];D=J/'source_data/checkpoint_computation'

def drop_texts(ax,*strings):
    """Remove in-axes labels by their text; each must exist exactly once."""
    hits=[t for t in ax.texts if t.get_text() in strings]
    assert sorted(t.get_text() for t in hits)==sorted(strings),[t.get_text() for t in ax.texts]
    for t in hits:t.remove()

def spread_ticklabels(fig,ax,shifts_pt):
    """Shift named x tick labels sideways by a few points (label placement only)."""
    labels={t.get_text():t for t in ax.get_xticklabels()}
    assert set(shifts_pt)<=set(labels),list(labels)
    for text,dx in shifts_pt.items():
        t=labels[text];t.set_transform(t.get_transform()+ScaledTranslation(dx/72,0,fig.dpi_scale_trans))

def main():
    c=NativeCanvas(440/72,3,row_weights=[112,118,64],hgutter_pt=28,vgutter_pt=40,
                   margins=Margins(left=48,right=46,top=24,bottom=38))
    rows=[];tables=gate.load_tables()
    contrast=gate.interaction(c.panel('A',0,0,colspan=6),tables)
    gate.cancellation(c.panel('B',0,6,colspan=6),tables)
    drop_texts(c.axes['A'],'zero difference');drop_texts(c.axes['B'],'no cancellation')
    for row in gate.legacy_display_rows(tables,{'E':contrast}):
        if row['panel'] in ('E','G'):
            rows.append(dict(row,panel={'E':'A','G':'B'}[row['panel']]))
    pop.ROWS.clear()
    pop.strip_panel(c.panel('C',1,0,colspan=6),
        pd.read_csv(pop.D/'inhibitory_selection_endpoints.csv').query("variant == 'separable'"),
        pd.read_csv(pop.D/'inhibitory_selection_rate_sensitivity.csv'),gate.plain_log_ticks)
    pop.alignment_panel(c.panel('D',1,6,colspan=6),
        pd.read_csv(pop.D/'context_alignment_seeds.csv'),
        pd.read_csv(pop.D/'context_alignment_summary.csv'))
    spread_ticklabels(c.fig,c.axes['D'],{'Broadcast':-2.0,'Resistance\ngate h':2.0})
    rows.extend(pop.ROWS);pop.ROWS.clear()
    grid,weights,matrices=response_matrices()
    titles=['Target','Resistance h','Augmented hf′','Exact']
    for column,((rule,matrix),letter,title) in enumerate(zip(matrices.items(),'EFGH',titles)):
        ax=c.panel(letter,2,column*3,colspan=3);style_panel(ax)
        im=ax.pcolormesh(grid,grid,matrix.T,cmap=DIV_CMAP,vmin=-1.2,vmax=1.2,shading='nearest')
        ax.set_aspect('equal');ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2' if column==0 else '')
        ax.text(.5,1.08,title,ha='center',transform=ax.transAxes,fontsize=7)
        rows.extend(dict(panel=letter,rule=rule,z1=float(grid[i]),z2=float(grid[j]),value=float(matrix[i,j]),quantity='response') for i in range(len(grid)) for j in range(len(grid)))
    locks=c.lock_reserves()
    for letters in ('AB','CD','EFGH'):
        left=max(locks[l][0] for l in letters)+2;right=max(locks[l][1] for l in letters)+2
        for letter in letters:c.declare_reserve(letter,left=left,right=right)
    c.lock_reserves();box=c.axes['E'].get_position()
    cbax=c.fig.add_axes([.923,box.y0+.1*box.height,.012,.8*box.height])
    cb=c.fig.colorbar(im,cax=cbax,ticks=[-1.2,0,1.2]);cb.ax.tick_params(labelsize=7,width=LW_HAIR,length=2)
    cb.ax.set_ylabel('Response',fontsize=8,labelpad=3)
    output=J/'figures/supplementary/curated/checkpoint_computation.pdf'
    findings=c.save(output,name='checkpoint_computation',dpi=180)
    pd.DataFrame(rows).to_csv(D/'checkpoint_plotted.csv',index=False)
    extra=[pop.D/n for n in ['inhibitory_selection_endpoints.csv','inhibitory_selection_rate_sensitivity.csv','context_alignment_seeds.csv','context_alignment_summary.csv']]
    extra += list((J/'source_data/conductance_local_gate/summaries').glob('*.csv'))
    extra += [J/'source_data/conductance_credit_demand/opponent/summaries/context_gradient_summary.csv']
    report=dict(figure='figS22',source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.csv')},
        diagnostic_sources={str(p.relative_to(J)):hashlib.sha256(p.read_bytes()).hexdigest() for p in extra},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        builder_sha256={str(p.relative_to(J)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(gate.__file__),Path(pop.__file__),J/'scripts/review_completion/promoted_panels.py']},
        cohorts='Original teacher, historical cancellation, original population and rescue cohorts; no new training',findings=[str(x) for x in findings])
    (D/'checkpoint_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
