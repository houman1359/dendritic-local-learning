import torch
from projected_k1 import project_common
from run import project_subtrees

def test_projected_k1_is_uniform_orthogonal_projection_with_same_soma():
    g=torch.Generator().manual_seed(813)
    values=[torch.randn(8,27,generator=g,dtype=torch.float64),torch.randn(8,9,generator=g,dtype=torch.float64),torch.randn(8,3,generator=g,dtype=torch.float64)]
    out=project_common(values)
    fields=torch.cat([values[1].reshape(8,3,3),values[0].reshape(8,3,9)],-1)
    expected=fields.mean(-1)
    torch.testing.assert_close(out[0].reshape(8,3,9),expected.unsqueeze(-1).expand(-1,-1,9),atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(out[1].reshape(8,3,3),expected.unsqueeze(-1).expand(-1,-1,3),atol=1e-15,rtol=1e-15)
    assert out[2] is values[2]
    torch.testing.assert_close(project_common(project_subtrees(values))[0],out[0],atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(project_subtrees(out)[0],out[0],atol=1e-15,rtol=1e-15)
