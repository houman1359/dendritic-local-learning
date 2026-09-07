import numpy as np
import opponent_model as model

def test_independent_torch_gradient():
 import torch
 torch.set_num_threads(1);x,y,_=model.data(9,'train',23,1.,10.);theta=np.log(model.NOMINAL)[None]+.15;variance=np.var(y);got,_=model.gradients(theta,x,y,variance,np.ones((1,6)),['exact']);t=torch.tensor(theta[0],dtype=torch.float64,requires_grad=True);g=t.exp();xx=torch.tensor(x);gg=g[:16].reshape(4,4);E=gg[:,0]*xx[:,:4]+gg[:,1]*xx[:,4:8];I=gg[:,2]*xx[:,:4]+gg[:,3]*xx[:,4:8];leaves=E/(1+E+I);parents=[]
 for p in range(2):
  c=g[18+2*p:20+2*p];parents.append((leaves[:,2*p:2*p+2]*c).sum(1)/(1+c.sum()+g[16+p]*xx[:,8+p]))
 pp=torch.stack(parents,1);pred=(pp*g[22:24]).sum(1)/(1+g[22:24].sum());loss=((pred-torch.tensor(y))**2).mean()/(2*variance);loss.backward();np.testing.assert_allclose(got[0],t.grad.numpy(),rtol=5e-13,atol=5e-13)

def test_local_eligibility_factorization():
 x,y,_=model.data(31,'train',20,1.,10.);t=model.teacher(31)[None];s=model.forward(t,x);full=model.eligibility(t,x)*s['path'][:,:,model.PARAM_UNIT]
 for j in range(24):
  a=t.copy();b=t.copy();a[:,j]+=1e-5;b[:,j]-=1e-5;fd=(model.forward(a,x)['output']-model.forward(b,x)['output'])/2e-5;np.testing.assert_allclose(full[:,:,j],fd,rtol=3e-8,atol=3e-11)

def test_same_inputs_and_known_forward_floor():
 x,y,c=model.data(8,'test',100,1.,10.);xx,yy,cc=model.data(8,'test',100,0.,10.);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(x[:,0:2],x[:,2:4]);np.testing.assert_array_equal(y,model.forward(model.teacher(8,True)[None],x)['output'][0]);np.testing.assert_array_equal(yy,model.forward(model.teacher(8,False)[None],x)['output'][0]);assert (x>=0).all()

def test_positive_fixed_profiles_scale_coordinates():
 x,y,_=model.data(9,'train',20,1.,10.);t=np.log(model.NOMINAL)[None];p=model.forward(t,x)['path'][:,:,:6].mean(1);u,_=model.gradients(t,x,y,np.var(y),p,['unit_broadcast']);b,_=model.gradients(t,x,y,np.var(y),p,['calibrated_broadcast']);scale=np.r_[p[0],1.];np.testing.assert_allclose(b,u*scale[model.PARAM_UNIT],rtol=5e-13,atol=5e-13)

def test_diagnostic_shapes_and_eligibility_conflict():
 x,y,_=model.data(19,'diagnostic',300,1.,10.);t=model.teacher(19)[None];p=model.forward(t,x)['path'][:,:,:6].mean(1);m=model.metrics(t,x,y,np.var(y),p)[0];assert m['path_rank_one_capture']<.9;assert m['eligibility_calibrated_oracle_capture']<.8


def test_three_route_projection_matches_independent_least_squares():
 x,y,_=model.data(17,'test',100,1.,10.);t=model.teacher(17)[None];q=model.forward(t,x)['path'][0,:,:6];p=q.mean(0);D=model.ancestry_dictionary(p);actual=model.project_ancestry(q,p);expected=np.linalg.lstsq(D,q.T,rcond=None)[0].T@D.T;np.testing.assert_allclose(actual,expected,atol=3e-15);assert np.linalg.matrix_rank(D)==3;assert np.sum((q-actual)**2)<=np.sum((q-q@p[:,None]/(p@p)*p)**2)+1e-14
