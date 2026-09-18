import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from publish import exact_signflip, interval


def test_exact_signflip_matches_enumeration():
    values=np.array([1., -.5, .7, .8])
    brute=sum(abs(np.dot(signs, values))>=abs(values.sum())-1e-12
              for signs in itertools.product([-1, 1], repeat=4))/16
    assert exact_signflip(values)==brute
    assert exact_signflip(np.ones(20))==2/(2**20)
    assert exact_signflip(np.zeros(4))==1.


def test_interval_resamples_seeds():
    values=np.array([1., 2., 4.])
    draws=np.random.default_rng(3).integers(0, 3, (2000, 3))
    result=interval(values, draws)
    assert result['mean']==pytest.approx(7/3)
    assert result['n']==3 and result['ci_low']<=result['mean']<=result['ci_high']
