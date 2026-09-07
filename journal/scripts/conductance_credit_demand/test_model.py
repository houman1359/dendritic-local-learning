"""Independent finite-difference and PyTorch checks of circuit and credit rules."""
import numpy as np
import model

def test_constructive_floor_and_positivity():
 for gate in [0.,1.,10.]:
  x,y,_=model.data(9,'test',32,1.,gate);s=model.forward(model.teacher(9)[None],x)
  np.testing.assert_array_equal(y,s['output'][0]);assert (x>=0).all();assert ((s['voltage']>=0)&(s['voltage']<=1)).all()

def test_independent_torch_gradient():
 import torch
 torch.set_num_threads(1)
 x,y,_=model.data(9,'train',20,1.,10.);theta=np.log(model.NOMINAL)[None]+.15;variance=np.var(y);profiles=np.ones((1,6));got,_=model.gradients(theta,x,y,variance,profiles,['exact'])
 tt=torch.tensor(theta[0],dtype=torch.float64,requires_grad=True);g=tt.exp();xx=torch.tensor(x);yy=torch.tensor(y)
 leaves=g[:4]*xx[:,:4]/(1+g[:4]*xx[:,:4]+g[4:8]*xx[:,4:8]);parents=[]
 for p in range(2):
  coupling=g[10+2*p:12+2*p];parents.append((leaves[:,2*p:2*p+2]*coupling).sum(1)/(1+coupling.sum()+g[8+p]*xx[:,8+p]))
 parent=torch.stack(parents,1);out=(parent*g[14:16]).sum(1)/(1+g[14:16].sum());loss=((out-yy)**2).mean()/(2*variance);loss.backward();np.testing.assert_allclose(got[0],tt.grad.numpy(),rtol=3e-13,atol=3e-13)

def test_local_eligibility_and_path_factorization():
 x,y,_=model.data(31,'train',20,.5,10.);theta=model.teacher(31)[None];state=model.forward(theta,x);e=model.eligibility(theta,x);full=e*state['path'][:,:,model.PARAM_UNIT]
 for j in range(16):
  a=theta.copy();b=theta.copy();a[:,j]+=1e-5;b[:,j]-=1e-5;fd=(model.forward(a,x)['output']-model.forward(b,x)['output'])/(2e-5);np.testing.assert_allclose(full[:,:,j],fd,rtol=2e-8,atol=2e-11)

def test_context_changes_geometry_and_not_credit_input():
 t=model.teacher(19)[None]
 rows=[]
 for gate in [0.,10.]:
  x,y,_=model.data(19,'diagnostic',300,1.,gate);s=model.forward(t,x);p=s['path'][:,:,:6].mean(1);rows.append(model.metrics(t,x,y,np.var(y),p)[0]);
 assert abs(rows[0]['path_rank_one_capture']-1)<1e-12
 assert rows[1]['path_rank_one_capture']<.9

def test_positive_fixed_profile_is_adam_coordinate_scaling():
 x,y,_=model.data(5,'train',30,1.,10.);theta=np.log(model.NOMINAL)[None];p=model.forward(theta,x)['path'][:,:,:6].mean(1);unit,_=model.gradients(theta,x,y,np.var(y),p,['unit_broadcast']);fixed,_=model.gradients(theta,x,y,np.var(y),p,['calibrated_broadcast']);scale=np.ones(7);scale[:6]=p[0];np.testing.assert_allclose(fixed,unit*scale[model.PARAM_UNIT],atol=2e-13)
