#!/usr/bin/env python3
"""Main shunt figure: exact ancestry gains and the electrical boundary."""
from pathlib import Path
import sys
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
OUT=JOURNAL/'source_data/shunt_ancestry_gain'
sys.path.insert(0,str(HERE.parent))
from figure_canvas import NativeCanvas,Margins,COLORS,PT_LABEL,PT_ANNOT,PT_LEGEND,LW_DATA,LW_EDGE
from journal_style import style_direct_color_labels,apply_neurips_style
import build_main_figure_08 as old

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ancestry(ax):
    ax.set_xlim(-.02,1.02);ax.set_ylim(-.25,1.02)
    positions={0:(.5,.92),1:(.25,.70),2:(.75,.70),3:(.125,.48),4:(.375,.48),5:(.625,.48),6:(.875,.48)}
    positions.update({7+i:((i+.5)/8,.23) for i in range(8)})
    green={3,7,8};blue={1,4,9,10}
    colors={i:COLORS['shunting'] if i in green else COLORS['additive'] if i in blue else COLORS['mute'] for i in positions}
    for i in range(1,15):
        j=(i-1)//2;x,y=positions[i];xx,yy=positions[j]
        ax.plot([x,xx],[y,yy],color=colors[i],lw=LW_DATA,zorder=1)
    for i,(x,y) in positions.items():
        ax.plot(x,y,'o',ms=5.5,mfc=colors[i],mec='white',mew=LW_EDGE,zorder=3)
    ax.text(.54,.93,'soma',va='center',fontsize=PT_ANNOT)
    x,y=positions[3]
    ax.plot([x-.045,x+.045],[y+.065,y+.065],color=COLORS['inh'],lw=LW_DATA,zorder=4)
    ax.text(x-.055,y+.11,'shunt',ha='right',va='center',fontsize=PT_ANNOT,color=COLORS['inh'])
    for x,label,color in [(.13,'descendants',COLORS['shunting']),(.49,'sister block',COLORS['additive']),(.84,'soma side',COLORS['mute'])]:
        ax.text(x,.04,label,ha='center',va='center',fontsize=PT_ANNOT,color=color)
    ax.text(.5,-.115,'One adjoint gain within each color',ha='center',va='center',fontsize=PT_ANNOT)
    ax.text(.5,-.235,'Driving forces can vary within a block',ha='center',va='center',fontsize=PT_ANNOT)

def build():
    dest=OUT/'figures';dest.mkdir(parents=True,exist_ok=True)
    canvas=NativeCanvas(490/72,3,row_weights=[110,110,120],hgutter_pt=37,vgutter_pt=44,
        margins=Margins(left=49,right=12,top=23,bottom=35))
    a=canvas.panel('A',0,0,6,schematic=True,title='Shunting acts on an ancestry partition')
    b=canvas.panel('B',0,6,6)
    c=canvas.panel('C',1,0,6,grid='y')
    d=canvas.panel('D',1,6,6,grid='y')
    e=canvas.panel('E',2,0,12)
    ancestry(a);old.panel_tree_relation(b);old.panel_passive_dose(c);old.panel_factor_freeze(d);old.panel_electrotonic(e)
    e.set_title('Electrical state determines the localization contrast')
    style_direct_color_labels(canvas.fig)
    output=dest/'shunt_ancestry_gain_native.pdf'
    canvas.save(output,name='shunt_ancestry_gain_native',dpi=200)
    sources=[JOURNAL/'source_data'/folder/name for folder,name in [('figure4','category_effects.csv'),('figure4','focal_localization.csv'),
        ('focal_decomposition','cell_shapley.csv'),('physical_cable_sensitivity','cell_primary_contrasts.csv'),('physical_cable_sensitivity','cell_electrotonic_ratios.csv')]]
    record=dict(output_sha256=sha(output),builders_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in [Path(__file__),HERE.parent/'build_main_figure_08.py']},
        source_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in sources},
        scientific_data='B–E call the unchanged archived panel functions; same source rows and bootstrap seeds. A illustrates the exact partition corollary without fitted gains.',
        changes='Active-channel check moved out of main figure; physical boundary now spans full width; ancestry identity added.')
    (dest/'figure_provenance.json').write_text(json.dumps(record,indent=2)+'\n');plt.close(canvas.fig)
    # Retain the weak-channel/passive comparison as a supplementary asset.
    active=JOURNAL/'source_data/focal_selectivity_active_ensemble'
    apply_neurips_style()
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.75))
    fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.85,wspace=.42)
    old.panel_active_dose(axs[0],pd.read_csv(active/'condition_summary.csv'),old.load_passive_reference())
    old.panel_contrast_forest(axs[1],pd.read_csv(active/'paired_contrasts.csv'),pd.read_csv(active/'cell_condition_metrics.csv'))
    axs[0].set_title('Weak-channel linearization vs passive');axs[1].set_title('Weak-channel localization contrasts')
    for ax,letter in zip(axs,'AB'):
        ax.text(-.19,1.13,letter,transform=ax.transAxes,fontweight='bold',fontsize=10.5,va='bottom')
    style_direct_color_labels(fig)
    fig.savefig(dest/'weak_channel_linearization_check.pdf')
    fig.savefig(dest/'weak_channel_linearization_check.png',dpi=200)
    plt.close(fig)

if __name__=='__main__': build()
