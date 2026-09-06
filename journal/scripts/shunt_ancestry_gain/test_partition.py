"""Numerical checks of the passive-tree ancestry-partition corollary."""
import unittest
import numpy as np

def example(seed,n=25):
    rng=np.random.default_rng(seed)
    parent=np.array([-1]+[int(rng.integers(i)) for i in range(1,n)])
    g=np.diag(np.exp(rng.normal(0,.7,n)))
    for i in range(1,n):
        w=float(np.exp(rng.normal(0,.7)))
        j=parent[i];g[i,i]+=w;g[j,j]+=w;g[i,j]-=w;g[j,i]-=w
    return g,parent

def ancestors(parent,k):
    nodes=[]
    while k>=0:
        nodes.append(k); k=parent[k]
    return nodes[::-1]

def partition(parent,k):
    path=ancestors(parent,k)
    blocks=[]
    for i in range(len(parent)):
        common=set(ancestors(parent,i))&set(path)
        blocks.append(max(common,key=path.index))
    return path,np.array(blocks)

def gains(g,parent,k,kappa):
    r=np.linalg.inv(g);path,blocks=partition(parent,k)
    gain=np.array([1-kappa*r[k,0]*r[a,k]/((1+kappa*r[k,k])*r[a,0]) for a in path])
    b=np.column_stack([blocks==a for a in path]).astype(float)
    return r,path,blocks,gain,b

class PartitionTests(unittest.TestCase):
    def test_all_sites_random_grounded_trees(self):
        largest=0.
        for seed in range(24):
            g,parent=example(392600+seed)
            for k in range(len(parent)):
                for kappa in [.0,.1,1.,10.]:
                    r,path,blocks,gain,b=gains(g,parent,k,kappa)
                    changed=g.copy();changed[k,k]+=kappa
                    after=np.linalg.solve(changed,np.eye(len(parent))[:,0])
                    before=r[:,0]
                    residual=after-before*(b@gain)
                    largest=max(largest,float(abs(residual).max()))
                    np.testing.assert_allclose(after,before*(b@gain),rtol=2e-11,atol=3e-14)
                    self.assertTrue(np.all(gain>0.)); self.assertTrue(np.all(gain<=1.+1e-14))
                    self.assertAlmostEqual(gain[-1],1/(1+kappa*r[k,k]),places=13)
                    self.assertTrue(np.all(b.sum(axis=1)==1))
                    dictionary=before[:,None]*b
                    np.testing.assert_allclose(dictionary@gain,after,rtol=2e-11,atol=3e-14)
        print('Largest absolute ancestry identity residual:',largest)

    def test_soma_source_sign_and_ancestry_span(self):
        g,parent=example(392625)
        r,path,blocks,gain,b=gains(g,parent,18,.7)
        nested=np.column_stack([np.isin(np.arange(len(parent)),[i for i in range(len(parent)) if a in ancestors(parent,i)]) for a in path]).astype(float)
        projection=nested@np.linalg.lstsq(nested,b@gain,rcond=None)[0]
        np.testing.assert_allclose(projection,b@gain,atol=1e-14)
        for source in [-2.,0.,1.]:
            changed=g.copy();changed[18,18]+=.7
            q=np.linalg.solve(changed,np.eye(len(parent))[:,0]*source)
            np.testing.assert_allclose(q,source*r[:,0]*(b@gain),atol=1e-14)

    def test_arbitrary_overlapping_dictionary_is_not_diagonal(self):
        g,parent=example(392625)
        r,path,blocks,gain,b=gains(g,parent,18,3.)
        before=r[:,0]; after=before*(b@gain)
        # A single global baseline route overlaps every ancestry block.
        coefficient=float(before@after/(before@before))
        self.assertGreater(np.linalg.norm(after-coefficient*before),1e-4)

    def test_driving_force_adds_within_block_variation(self):
        g,parent=example(392626)
        r,path,blocks,gain,b=gains(g,parent,1,2.)
        before=np.linspace(.1,.3,len(parent)); after=before+np.linspace(0,.05,len(parent))
        ratio=(b@gain)*(1-after)/(1-before)
        populated=[ratio[blocks==a] for a in path if np.sum(blocks==a)>1]
        self.assertTrue(any(np.ptp(values)>1e-4 for values in populated))

if __name__=='__main__': unittest.main()
