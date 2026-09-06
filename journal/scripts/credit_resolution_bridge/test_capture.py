"""Mathematical controls for the post hoc spatial-credit diagnostic."""
import unittest
import numpy as np
import analysis_capture as a


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.coeff=a.credit.make_task(127200,'matching')
        self.tree,_,_=a.credit.tree_from_coeff(self.coeff,'test')
        self.left=np.array([[self.tree.children[n][0] for n in range(8,15)]])
        self.right=np.array([[self.tree.children[n][1] for n in range(8,15)]])
        self.x=a.credit.domain()
        self.w=np.random.default_rng(41).normal(0,.5,(1,7,4))

    def test_parameter_gradient_matches_finite_difference(self):
        prediction,q,features=a.algebraic_fields(self.w,self.left,self.right,self.x)
        target=a.credit.fourier_design(self.x)@self.coeff
        gradient=np.mean((prediction[0]-target)[:,None,None]*q[0,:,:,None]*features[0],axis=0)
        for node in range(7):
            for term in range(4):
                wp=self.w.copy();wm=self.w.copy();wp[0,node,term]+=1e-6;wm[0,node,term]-=1e-6
                plus=a.credit.forward(self.x,wp,self.left,self.right)[0,14]
                minus=a.credit.forward(self.x,wm,self.left,self.right)[0,14]
                numerical=(np.mean((plus-target)**2)-np.mean((minus-target)**2))/(4e-6)
                self.assertAlmostEqual(gradient[node,term],numerical,places=8)

    def test_scalar_residual_cancels_from_per_example_capture(self):
        _,q,_=a.algebraic_fields(self.w,self.left,self.right,self.x);q=q[0,:,:6]
        p=np.ones((6,6))/6
        residual=np.linspace(-2.13,3.3,len(q))
        credit=q*residual[:,None]
        path_capture=1-np.sum((q-q@p)**2,axis=1)/np.sum(q*q,axis=1)
        credit_capture=1-np.sum((credit-credit@p)**2,axis=1)/np.sum(credit*credit,axis=1)
        np.testing.assert_allclose(path_capture,credit_capture,atol=1e-13)

    def test_fixed_rank_one_can_be_orthogonal_to_uniform(self):
        profile=np.array([1,-1,2,-2,3,-3.])
        field=np.arange(1,12.)[:,None]*profile
        uniform=np.ones((6,6))/6
        self.assertLess(np.linalg.norm(field@uniform),1e-12)
        p=a.projector(a.best_profile(field)[:,None])
        np.testing.assert_allclose(field@p,field,atol=1e-12)
        _,s=a.spectrum(field)
        self.assertAlmostEqual(s['effective_rank'],1,places=12)

    def test_best_rank_one_matches_largest_eigenvalue(self):
        field=np.random.default_rng(88).normal(size=(70,6))
        p=a.projector(a.best_profile(field)[:,None])
        capture=np.sum((field@p)**2)/np.sum(field**2)
        _,spec=a.spectrum(field)
        self.assertAlmostEqual(capture,spec['leading_fraction'],places=12)

    def test_ancestry_budgets_are_nested_complete_and_monotone(self):
        rng=np.random.default_rng(13);field=rng.normal(size=(50,6));previous=-1
        for family in a.credit.FAMILIES:
            tree,_,_=a.credit.tree_from_coeff(a.credit.make_task(127201,family),'test')
            bases=a.ancestry_bases(tree)
            self.assertEqual(list(bases),[1,2,3,4,5,6]);previous=-1
            for k,basis in bases.items():
                p=a.projector(basis)
                np.testing.assert_allclose(p@p,p,atol=1e-12)
                self.assertEqual(np.linalg.matrix_rank(p),k)
                value=np.sum((field@p)**2)/np.sum(field**2)
                self.assertGreaterEqual(value+1e-12,previous);previous=value
            np.testing.assert_allclose(field@p,field,atol=1e-12)

    def test_pairwise_quartic_input_spectra_match_nested_is_separate(self):
        matching=a.credit.input_gradient_covariance(a.credit.make_task(127200,'matching'))
        quartet=a.credit.input_gradient_covariance(a.credit.make_task(127200,'quartet'))
        nested=a.credit.input_gradient_covariance(a.credit.make_task(127200,'nested'))
        np.testing.assert_allclose(matching,np.eye(8)*.25,atol=1e-14)
        np.testing.assert_allclose(quartet,matching,atol=1e-14)
        self.assertGreater(np.max(abs(nested-matching)),.1)


if __name__=='__main__':unittest.main()
