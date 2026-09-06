"""Independent finite differences, truth tables and credit-coordinate checks."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boolean_morphology.model import FAMILIES,TREES,domain,raw_target,normalization,target,pack_tree,forward,gradient,classification
from boolean_morphology.experiment import conditions,protocol,seed_data


def test_truth_tables_centering_and_threshold():
    x=domain();counts=[1,15,8,7,6,4,5]
    for family,count in zip(FAMILIES,counts):
        raw=raw_target(x,family);z=target(x,family);mean,sd=normalization(family)
        assert raw.sum()==count
        assert abs(z.mean())<1e-15 and abs(np.mean(z*z)-1)<1e-15
        accuracy,balanced=classification(z[None],raw[None],np.array([[mean]]),np.array([[sd]]))
        assert accuracy[0]==1 and balanced[0]==1


def test_exact_gradient_and_broadcast_root():
    rng=np.random.default_rng(390112);x=rng.normal(size=(9,4));y=rng.normal(size=9)
    children=np.array([pack_tree(tree) for tree in TREES.values()])
    weights=rng.normal(0,.3,(4,3,4))
    g,exact,pred,q=gradient(x,y,weights,children,np.zeros(4,bool))
    assert np.array_equal(g,exact) and np.array_equal(q[:,-1],np.ones_like(q[:,-1]))
    error=[];eps=1e-6
    for i in range(4):
        for node in range(3):
            for coefficient in range(4):
                plus=weights.copy();minus=weights.copy();plus[i,node,coefficient]+=eps;minus[i,node,coefficient]-=eps
                lp=np.mean((forward(x,plus,children)[i,6]-y)**2)/2
                lm=np.mean((forward(x,minus,children)[i,6]-y)**2)/2
                error.append(abs((lp-lm)/(2*eps)-g[i,node,coefficient]))
    assert max(error)<2e-9
    broadcast,_,_,_=gradient(x,y,weights,children,np.ones(4,bool))
    np.testing.assert_array_equal(broadcast[:,-1],exact[:,-1])
    assert np.max(abs(broadcast[:,:2]-exact[:,:2]))>1e-3


def test_joint_leaf_permutation_and_pairing():
    rng=np.random.default_rng(448221);x=domain();p=np.array([2,0,3,1]);w=rng.normal(0,.5,(4,3,4))
    base=np.array([pack_tree(tree) for tree in TREES.values()]);mapped=np.array([pack_tree(tree,p) for tree in TREES.values()])
    np.testing.assert_array_equal(forward(x[:,p],w,base),forward(x,w,mapped)[:,[int(p[0]),int(p[1]),int(p[2]),int(p[3]),4,5,6]])
    cfg=protocol();assert not set(cfg['development_seeds'])&set(cfg['fresh_seeds'])
    data=seed_data(cfg['excluded_smoke_seed'],cfg);meta,children=conditions(data['permutation'])
    assert len(meta)==336 and children.shape==(336,3,2)
    assert data['batch_indices'].shape==(7,2048,32)
    for i,family in enumerate(FAMILIES):
        assert len({tuple(row.values()) for row in meta if row['family']==family})==48
        assert data['initial_weights'][i,:,0].sum()==0
        assert not np.array_equal(data['train_x'][i],data['test_x'][i,:256])
