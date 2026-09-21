"""Exact-gradient, production-forward, physical-domain and task-spectrum tests."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import model

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO/'src'))
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import forward_branch_dynamics


def production_branch(excitation,inhibition,children,coupling):
    """Call actual production balance with minimal prescribed current pathways."""
    class Coupling:
        def __call__(self,value):
            return (value*coupling).sum(dim=-1)
        def sum_conductances(self):
            return coupling.sum(dim=-1)
    owner=SimpleNamespace(use_shunting=True,additive_mode='raw',_store_diagnostics=False,
        input_excitatory=excitation is not None,input_recurrent=False,input_branches=children is not None,
        input_inhibitory=inhibition is not None,input_rec_inhibitory=False,
        branch_excitation=lambda x:excitation,branch_inhibition=lambda x:inhibition,
        branches_to_output=Coupling(),reactivation=torch.nn.Identity(),epsilon=0.,training=False)
    return forward_branch_dynamics(owner,torch.ones(1),inhibitory_input=torch.ones(1),branch_input=children)


def production_forward(theta,x,group):
    g=theta.exp();v=[]
    for leaf in range(4):
        v.append(production_branch(g[leaf]*x[:,leaf],g[4+leaf]*x[:,4+leaf],None,None))
    for p in range(2):
        child=torch.stack([v[int(i)] for i in group[p]],dim=-1)
        v.append(production_branch(None,g[8+p]*x[:,8+p],child,g[10+2*p:12+2*p]))
    v.append(production_branch(None,None,torch.stack(v[4:6],dim=-1),g[14:16]))
    return torch.stack(v,dim=-1)


class ConductanceTests(unittest.TestCase):
    def test_production_forward_and_autograd_match_eligibility_paths(self):
        rng=np.random.default_rng(802)
        theta=np.log(model.NOMINAL_G)+rng.normal(0,.4,(3,16)); x=np.exp(rng.normal(size=(37,10)))
        y=rng.uniform(.05,.3,37);variance=.02
        gradient,state=model.gradients(theta,x,y,model.GROUPINGS,variance)
        for i in range(3):
            tt=torch.tensor(theta[i],dtype=torch.float64,requires_grad=True)
            xx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            voltage=production_forward(tt,xx,model.GROUPINGS[i])
            np.testing.assert_allclose(voltage.detach().numpy(),state['voltage'][i],atol=3e-16)
            loss=((voltage[:,-1]-torch.tensor(y))**2).mean()/(2*variance)
            gg=torch.autograd.grad(loss,tt,retain_graph=True)[0]
            np.testing.assert_allclose(gg.detach().numpy(),gradient[i],rtol=2e-13,atol=1e-14)
            gx=torch.autograd.grad(voltage[:,-1].sum(),xx)[0]
            np.testing.assert_allclose(gx.numpy(),model.input_gradient(theta[[i]],x,model.GROUPINGS[[i]])[0],rtol=2e-13,atol=1e-14)
        self.assertTrue(np.all(state['denominator']>=1.))
        self.assertTrue(np.all((state['voltage']>=0)&(state['voltage']<=1)))

    def test_planted_task_and_input_spectrum_invariance(self):
        eigenvalues=[]
        for grouping in range(3):
            x,y=model.dataset(15300,grouping,'spectrum',1024)
            prediction=model.forward(model.equivalent_teacher_parameters(15300,grouping)[None],x,model.GROUPINGS[[grouping]])['output'][0]
            np.testing.assert_allclose(prediction,y,atol=2e-16)
            jac=model.teacher_gradient_in_task_coordinates(15300,grouping,x)
            eigenvalues.append(np.linalg.eigvalsh(jac.T@jac/len(jac)))
            self.assertTrue(np.all(x>0))
        np.testing.assert_allclose(eigenvalues[1],eigenvalues[0],rtol=1e-12,atol=1e-17)
        np.testing.assert_allclose(eigenvalues[2],eigenvalues[0],rtol=1e-12,atol=1e-17)
        self.assertGreater(min(eigenvalues[0]),0.)

    def test_projection_normal_equations_and_label_free_calibration(self):
        theta=np.log(model.NOMINAL_G)[None].repeat(3,axis=0)
        x,_=model.dataset(15301,0,'calibration',64)
        profiles=model.calibrate_profiles(theta,x,model.GROUPINGS)
        exact=model.forward(theta,x,model.GROUPINGS)['path']
        for rule in [2,3]:
            projected=model.delivery(exact,profiles,model.GROUPINGS,np.full(3,rule))
            error=exact[:,:,:6]-projected[:,:,:6]
            if rule==3:
                np.testing.assert_allclose(np.einsum('mk,mnk->mn',profiles,error),0.,atol=1e-15)
            else:
                for i in range(3):
                    for p in range(2):
                        indices=list(model.GROUPINGS[i,p])+[4+p]
                        np.testing.assert_allclose(error[i][:,indices]@profiles[i,indices],0.,atol=1e-15)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
