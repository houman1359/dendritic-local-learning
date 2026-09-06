"""Generic tree packing and exact gradient checks before the second freeze."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import end_to_end as experiment
from bridge import candidates,domain


def test_explicit_reference_import_and_arbitrary_root_children():
    assert Path(experiment.credit.__file__).resolve()==experiment.CREDIT.resolve()
    trees=[next(tree for tree in candidates() if tree.name==name) for name in
           ("comb_p0","balanced_p0","mixed_p0")]
    meta,left,right,p=experiment.pack(trees)
    assert len(meta)==6 and p.shape==(6,6,6)
    assert trees[0].children[14][0]<8  # comb root has a leaf child; zones() would reject it
    w=np.random.default_rng(1).normal(0,.2,(6,7,4));x=domain()[:16];y=np.zeros(16)
    delivered,exact,_,q,routed=experiment.credit.gradient(x,y,w,left,right,p,meta,1.)
    for index in (0,2,4):np.testing.assert_array_equal(delivered[index],exact[index])
    np.testing.assert_array_equal(routed[:,6],np.ones((6,16)))
    np.testing.assert_array_equal(routed[[1,3,5],:6],np.ones((3,6,16)))


def test_exact_gradient_matches_central_difference_for_every_generic_tree():
    trees=[next(tree for tree in candidates() if tree.name==name) for name in
           ("comb_p0","balanced_p0","mixed_p0")]
    meta,left,right,p=experiment.pack(trees)
    generator=np.random.default_rng(77);w=generator.normal(0,.2,(6,7,4));x=domain()[::8];y=generator.normal(size=len(x))
    _,exact,_,_,_=experiment.credit.gradient(x,y,w,left,right,p,meta,1.)
    for index in (0,2,4):
        for node,parameter in ((0,3),(3,1),(6,0)):
            h=1e-6;plus=w.copy();minus=w.copy();plus[index,node,parameter]+=h;minus[index,node,parameter]-=h
            a=experiment.credit.forward(x,plus,left,right)[index,14]
            b=experiment.credit.forward(x,minus,left,right)[index,14]
            numerical=(np.mean((a-y)**2)-np.mean((b-y)**2))/(4*h)
            np.testing.assert_allclose(exact[index,node,parameter],numerical,atol=1e-9,rtol=1e-6)
