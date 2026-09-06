"""Independent finite differences and controlled credit geometry checks."""
import unittest
import numpy as np
import models

class BridgeTests(unittest.TestCase):
    def test_same_tree_and_isospectral_task_pair(self):
        for seed in (210100,210101,211200):
            cm,tm = models.algebra_task(seed,'matching')
            cq,tq = models.algebra_task(seed,'quartet')
            self.assertEqual(tm.children,tq.children)
            np.testing.assert_allclose(models.structure.input_gradient_covariance(cm),np.eye(8)/4,atol=1e-15)
            np.testing.assert_allclose(models.structure.input_gradient_covariance(cq),np.eye(8)/4,atol=1e-15)
            _,tn = models.algebra_task(seed,'nested')
            self.assertEqual(set(tn.descendants[14]),set(range(8)))

    def test_algebraic_gradient_finite_difference(self):
        rng = np.random.default_rng(812)
        _,tree = models.algebra_task(210100,'matching')
        left=np.array([[tree.children[k][0] for k in range(8,15)]])
        right=np.array([[tree.children[k][1] for k in range(8,15)]])
        w=rng.normal(0,.3,(1,7,4)); x=2.*rng.integers(2,size=(31,8))-1.; y=rng.normal(size=31)
        profile=models.algebra_state(w,x,left,right)['path'][:,:,:6].mean(axis=1)
        g,_=models.algebra_grad(w,x,y,left,right,.7,profile,['exact'])
        for j in range(7):
            for k in range(4):
                wp=w.copy(); wm=w.copy(); wp[0,j,k]+=1e-6; wm[0,j,k]-=1e-6
                lp=.5*np.mean((models.algebra_state(wp,x,left,right)['output']-y)**2)/.7
                lm=.5*np.mean((models.algebra_state(wm,x,left,right)['output']-y)**2)/.7
                self.assertAlmostEqual((lp-lm)/2e-6,g[0,j,k],places=8)

    def test_conductance_exact_gradient_and_positive_sign_identity(self):
        rng=np.random.default_rng(813)
        theta=np.log(models.conductance.NOMINAL_G)[None]+rng.normal(0,.2,(1,16))
        x,y=models.conductance.dataset(210100,0,'training',29); groups=models.conductance.GROUPINGS[[0]]
        profile=models.conductance.calibrate_profiles(theta,x,groups)
        g,_=models.conductance_grad(theta,x,y,groups,.02,profile,['exact'])
        for k in range(16):
            plus=theta.copy(); minus=theta.copy(); plus[0,k]+=1e-6; minus[0,k]-=1e-6
            lp=.5*np.mean((models.conductance.forward(plus,x,groups)['output']-y)**2)/.02
            lm=.5*np.mean((models.conductance.forward(minus,x,groups)['output']-y)**2)/.02
            self.assertAlmostEqual((lp-lm)/2e-6,g[0,k],places=8)
        u,_=models.conductance_grad(theta,x,y,groups,.02,profile,['unit_broadcast'])
        s,_=models.conductance_grad(theta,x,y,groups,.02,profile,['sign_broadcast'])
        np.testing.assert_array_equal(u,s)

    def test_frozen_profile_and_oracle_normal_equations(self):
        rng=np.random.default_rng(814)
        paths=rng.normal(size=(1,33,7)); paths[:,:,-1]=1.
        profile=paths[:,:,:6].mean(axis=1)
        output=models.deliver(paths,profile,['calibrated_broadcast'])
        np.testing.assert_allclose(output[0,:,:6],np.broadcast_to(profile[0],(33,6)))
        np.testing.assert_array_equal(output[:,:,-1],np.ones((1,33)))
        direction=profile[0]; field=paths[0,:,:6]
        projected=(field@direction)[:,None]*direction/(direction@direction)
        np.testing.assert_allclose((field-projected)@direction,0.,atol=1e-15)
        metrics=models.field_metrics(paths,np.ones((1,33)),output,profile)[0]
        expected=np.sum(projected**2)/np.sum(field**2)
        self.assertAlmostEqual(metrics['path_calibrated_oracle_capture'],expected,places=14)
        self.assertAlmostEqual(metrics['path_full_rank_capture'],1.,places=14)
        self.assertLessEqual(metrics['path_calibrated_oracle_capture'],metrics['path_best_rank_one_capture']+1e-14)

    def test_rank_one_is_not_uniform(self):
        direction=np.array([1.,-1.,1.,-1.,1.,-1.])
        paths=np.ones((1,13,7)); paths[0,:,:6]=np.arange(1,14)[:,None]*direction
        profile=direction[None]
        metrics=models.field_metrics(paths,np.ones((1,13)),paths,profile)[0]
        self.assertEqual(metrics['path_uniform_oracle_capture'],0.)
        self.assertAlmostEqual(metrics['path_best_rank_one_capture'],1.)
        self.assertAlmostEqual(metrics['path_calibrated_oracle_capture'],1.)

if __name__=='__main__': unittest.main()
