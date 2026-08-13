import pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 260)
CSV='/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/combined_results.csv'
df=pd.read_csv(CSV, low_memory=False)
main=['mnist','context_gating','noise_resilience','info_shunting','hierarchical_processing']
print('========= LOCAL vs BACKPROP (shunting): backprop_mean - local_mean =========')
for ds in main:
    loc=df[(df.dataset==ds)&(df.network_type=='dendritic_shunting')&(df.strategy=='local_ca')].test_accuracy.dropna()
    bp =df[(df.dataset==ds)&(df.network_type=='dendritic_shunting')&(df.strategy=='standard')].test_accuracy.dropna()
    if len(bp)==0:
        print('%-24s local %.4f (n%d)  | NO standard/backprop shunting rows'%(ds,loc.mean(),len(loc))); continue
    gap=bp.mean()-loc.mean()
    print('%-24s backprop %.4f (n%d)  local %.4f (n%d)  backprop-local %+.4f'%(ds,bp.mean(),len(bp),loc.mean(),len(loc),gap))
print()
m=df[df.dataset=='mnist']
bpp=m[(m.sweep_name=='phase1_capacity')&(m.network_type=='dendritic_shunting')].test_accuracy.mean()
tl=m[m.sweep_name=='phase2b_gap_pilot'].test_accuracy.mean()
print('mnist tuned-local(phase2b_gap_pilot) %.4f vs backprop(phase1_capacity) %.4f -> backprop-local %.4f'%(tl,bpp,bpp-tl))
print()
print('========= CROSS-CHECK effect_sizes.csv (claimA shunting-additive per ie/mode) =========')
es=pd.read_csv('/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/effect_sizes.csv')
nok=ndiff=0
for ds in ['info_shunting','noise_resilience']:
    for mode in ['per_soma','scalar','local_mismatch']:
        sub=df[(df.dataset==ds)&(df.strategy=='local_ca')&(df.error_broadcast_mode==mode)]
        for ie in sorted(es[(es.dataset==ds)&(es.error_broadcast_mode==mode)].ie_value.unique()):
            s=sub[(sub.ie_value==ie)&(sub.network_type=='dendritic_shunting')].test_accuracy
            a=sub[(sub.ie_value==ie)&(sub.network_type=='dendritic_additive')].test_accuracy
            mine = (s.mean()-a.mean()) if (len(s) and len(a)) else float('nan')
            ref=es[(es.dataset==ds)&(es.error_broadcast_mode==mode)&(es.ie_value==ie)].value
            refv=ref.values[0] if len(ref) else float('nan')
            ok = (mine==mine) and (refv==refv) and abs(mine-refv)<0.02
            flag='OK' if ok else 'DIFF/na'
            if ok: nok+=1
            else: ndiff+=1
            print('%-16s %-14s ie %-4s mine %+.5f  effsz.csv %+.5f  %s'%(ds,mode,str(ie),mine,refv,flag))
print()
print('cross-check summary: OK=%d  DIFF/na=%d'%(nok,ndiff))
