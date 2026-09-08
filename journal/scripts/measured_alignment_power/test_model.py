"""Meaningful checks on dependency preservation, projection and exact inference."""
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import rankdata,wilcoxon
from model import residual_design,statistic,exact_signed_rank,exact_test_lattice,load_dataset,aggregate
J=Path(__file__).resolve().parents[2]
def test_exact_test_all_128_sign_patterns():
 values=(2*np.array(list(itertools.product([0,1],repeat=7)))-1)*np.arange(1,8)
 actual=exact_signed_rank(values)
 expected=np.array([wilcoxon(v,method='exact').pvalue for v in values])
 np.testing.assert_array_equal(actual,expected)
 lattice=exact_test_lattice();assert lattice['minimum_two_sided_p']==.015625;assert lattice['null_rejection_fraction_at_005']==6/128
 assert exact_signed_rank(np.zeros((1,7)))[0]==1
 assert exact_signed_rank(np.array([[0,0,1,2,3,4,5]]))[0]==.0625

def test_valid_joint_kernel_and_observed_statistic():
 d=load_dataset(J/'source_data/measured_alignment_power/inputs');np.testing.assert_allclose(d.factor@d.factor.T,d.kernel,atol=1e-12);np.testing.assert_allclose(np.diag(d.kernel),1)
 import pandas as pd
 expected=pd.read_csv(J/'source_data/measured_alignment_power/inputs/scan_index.csv').set_index('scan')
 found=[]
 for s in d.scans:
  obs=np.load(J/'source_data/measured_alignment_power/inputs'/s.name/'observed_partner_responses.npz');ids=obs['stimulus_ids'];y=np.stack([obs['responses'][ids==v].mean(0) for v in obs['repeated_stimulus_ids']])
  value=statistic(y[None],s)[0];np.testing.assert_allclose(value,expected.loc[s.name,'observed_partial_rank'],atol=1e-12);found.append(value)
 assert len(aggregate(np.array(found)[None],d.scan_target,7)[0])==7
 assert len(d.scans)==13 and len(d.roots)==102

def test_rank_residual_removes_controls_and_is_scale_invariant():
 x=np.array([0,0,1,2,3,4.]);c=np.column_stack([np.arange(6),[0,1,1,2,3,3]])
 q,xx=residual_design(x,c);np.testing.assert_allclose(q.T@xx,0,atol=1e-12);np.testing.assert_allclose(np.linalg.norm(xx),1)
 q2,xx2=residual_design(x, np.log1p(c));np.testing.assert_allclose(xx,xx2,atol=1e-12)

def test_joint_responses_enforce_pair_constraints():
 # Three pair correlations from one response matrix must form a PSD matrix;
 # independently drawn pair coefficients do not in general satisfy this.
 y=np.array([[1,0,1],[0,1,1],[-1,0,-1],[0,-1,-1]],float)
 corr=np.corrcoef(y.T);assert np.linalg.eigvalsh(corr).min()>-1e-12
 impossible=np.array([[1,.9,.9],[.9,1,-.9],[.9,-.9,1]])
 assert np.linalg.eigvalsh(impossible).min()<0
