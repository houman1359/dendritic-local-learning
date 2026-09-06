"""Independent checks of the functional-ANOVA obstruction and its moments."""
import unittest
import numpy as np

import model
import interaction_bound as bound


class InteractionBoundTests(unittest.TestCase):
    def test_missing_mixed_interaction_is_zero_for_arbitrary_student(self):
        rng=np.random.default_rng(1811)
        theta=np.log(model.NOMINAL_G)+rng.normal(0,.8,16)
        x=np.exp(rng.normal(size=(19,10)))
        # Target 01|23 pairs terminal1 with proximal0. Student02|13 places
        # terminal1 under proximal1; any conductances lack this mixed term.
        samples=[]
        for ee,pp in [(1.,1.),(2.,1.),(1.,3.),(2.,3.)]:
            xx=x.copy();xx[:,1]=ee;xx[:,8]=pp;samples.append(xx)
        values=[model.forward(theta[None],xx,model.GROUPINGS[[1]])['output'][0] for xx in samples]
        mixed=values[3]-values[1]-values[2]+values[0]
        np.testing.assert_allclose(mixed,0.,atol=2e-16)
        true=[model.forward(theta[None],xx,model.GROUPINGS[[0]])['output'][0] for xx in samples]
        self.assertGreater(float(np.min(np.abs(true[3]-true[1]-true[2]+true[0]))),1e-5)

    def test_population_moment_factorization_and_orthogonal_residual(self):
        seed=15329
        m=bound.moments(seed,64)
        x,y=model.dataset(seed,0,'spectrum',60000)
        g=m['g']
        state=model.forward(np.log(g)[None],x,model.GROUPINGS[[0]])
        terminal=state['voltage'][0,:,:4]
        attenuation=np.column_stack([1/state['denominator'][0,:,4+p] for p in range(2)])
        # Student02|13 misplaces canonical terminals1 and2.
        residual=np.zeros(len(x))
        for leaf in [1,2]:
            p=leaf//2
            residual+=m['root'][p]*g[10+leaf]*(terminal[:,leaf]-m['terminal_means'][leaf])*(attenuation[:,p]-m['proximal_attenuation_means'][p])
        exact=bound.lower_bound(seed,0,1,64)['additive_subtree_mse_lower_bound']
        observed=float(np.mean(residual**2));se=float(np.std(residual**2)/np.sqrt(len(x)))
        self.assertLess(abs(observed-exact),6*se)
        variance_observed=float(np.mean((y-m['population_mean'])**2))
        variance_se=float(np.std((y-m['population_mean'])**2)/np.sqrt(len(x)))
        self.assertLess(abs(variance_observed-m['population_variance']),6*variance_se)


if __name__=='__main__':unittest.main()
