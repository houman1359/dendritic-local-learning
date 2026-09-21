import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_review_morphology_uncertainty as audit


def segments():
    return pd.DataFrame({"segment_id":[0,1,2,3,4],"parent_segment_id":[-1,0,0,1,2],
        "mean_radius_um":[1.,1.,1.,1.,1.],"edge_length_um":[0.,2.,3.,4.,5.],
        "topological_depth":[0,1,1,2,2],"E_size":[0.,1.,1.,1.,1.],
        "I_size":[.2,.3,.4,.2,.3],"raw_axial_resistance":[1.,2.,3.,4.,5.]})


def test_homogeneous_cables_match_exact_series_resistance():
    s=segments();sites=[1,2,3,4];weight=np.ones(4)
    standard=audit.dictionary(s,sites,weight,"mean_radius")
    series=audit.dictionary(s,sites,weight,"series_resistance")
    np.testing.assert_allclose(standard[0],series[0],atol=1e-13)
    assert standard[2]==series[2]


def test_series_identity_survives_uniform_radius_scaling():
    s=segments();scaled=s.copy();scaled.mean_radius_um*=2;scaled.raw_axial_resistance/=4
    # Nonroot leak normalization can involve soma-area clipping, so compare the
    # exact-vs-mean axial constructions within each homogeneous geometry.
    for frame in [s,scaled]:
        a=audit.dictionary(frame,[1,2,3,4],np.ones(4),"mean_radius")
        b=audit.dictionary(frame,[1,2,3,4],np.ones(4),"series_resistance")
        np.testing.assert_allclose(a[0],b[0],atol=1e-13)


def test_capture_uses_fixed_probe_and_is_invariant_to_nonzero_column_gains():
    rng=np.random.default_rng(27);d=rng.normal(size=(6,3));target=rng.normal(size=(6,8));weights=np.arange(1.,7.)
    a,rank=audit.capture(target,d,weights)
    b,scaled_rank=audit.capture(target,d@np.diag([.1,2.,5.]),weights)
    assert 0<=a<=1 and rank==scaled_rank==3
    np.testing.assert_allclose(a,b,atol=1e-13)
