import unittest
import numpy as np
from experiment import make_task,tree_from_coeff,pack,gradient,forward,domain,zones,shuffled_tree

class CreditTests(unittest.TestCase):
    def test_exact_gradient_and_route_projectors(self):
        for family in ('matching','quartet','nested'):
            coeff=make_task(1,family);tree,_,_=tree_from_coeff(coeff,'compatible')
            p=zones(tree)
            np.testing.assert_allclose(p@p,p,atol=1e-12)
            self.assertEqual(np.linalg.matrix_rank(p),2)
            meta,left,right,ps=pack([tree,shuffled_tree(tree,1)],1)
            meta=meta[:1];left=left[:1];right=right[:1];ps=ps[:1]
            rng=np.random.default_rng(2);w=rng.normal(0,.3,(1,7,4));x=domain()[:32];y=rng.normal(size=32)
            delivered,exact,_,_,_=gradient(x,y,w,left,right,ps,meta,1.)
            np.testing.assert_allclose(delivered,exact,atol=1e-12)
            for j in range(7):
                for k in range(4):
                    wp=w.copy();wm=w.copy();wp[0,j,k]+=1e-6;wm[0,j,k]-=1e-6
                    lp=.5*np.mean((forward(x,wp,left,right)[:,14]-y)**2)
                    lm=.5*np.mean((forward(x,wm,left,right)[:,14]-y)**2)
                    self.assertAlmostEqual((lp-lm)/2e-6,exact[0,j,k],places=8)

    def test_shuffling_preserves_shape_and_parameter_count(self):
        for family in ('matching','quartet','nested'):
            tree,_,_=tree_from_coeff(make_task(12,family),'compatible')
            changed=shuffled_tree(tree,12)
            self.assertEqual(sorted(len(v) for v in tree.descendants.values()),sorted(len(v) for v in changed.descendants.values()))
            self.assertEqual(set(changed.descendants[14]),set(range(8)))

if __name__=='__main__':unittest.main()
