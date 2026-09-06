"""Independent checks of weighted decomposition, span and stated costs."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("commonmode_run", Path(__file__).with_name("run.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_weighted_capture_matches_least_squares_and_residual_identity():
    rng = np.random.default_rng(12)
    sqrt_weight = np.sqrt(rng.uniform(.2, 3., 25))
    response = rng.normal(size=(25, 11)) + 2
    weighted = response * sqrt_weight[:, None]
    dictionary = np.column_stack([np.ones(25), rng.normal(size=(25, 3))])
    answer = analysis.evaluate(dictionary, weighted, sqrt_weight)
    weighted_dictionary = dictionary * sqrt_weight[:, None]
    fitted = weighted_dictionary @ np.linalg.lstsq(weighted_dictionary, weighted, rcond=None)[0]
    independent_capture = 1 - np.sum((weighted - fitted) ** 2) / np.sum(weighted ** 2)
    np.testing.assert_allclose(answer["total_capture"], independent_capture, atol=1e-13)
    q0 = sqrt_weight / np.linalg.norm(sqrt_weight)
    common_capture = np.sum((q0 @ weighted) ** 2) / np.sum(weighted ** 2)
    np.testing.assert_allclose(answer["total_capture"], common_capture + (1-common_capture)*answer["residual_capture"], atol=1e-13)


def test_common_only_field_has_no_spatial_residual_credit():
    weight = np.sqrt(np.arange(1, 13))
    weighted = weight[:, None] * np.array([[1., -3., 2.]])
    d = np.column_stack([np.ones(12), np.arange(12) > 4])
    answer = analysis.evaluate(d, weighted, weight)
    np.testing.assert_allclose(answer["total_capture"], 1, atol=1e-13)
    assert answer["residual_capture"] == 0


def test_depth_common_reparameterization_preserves_span_but_changes_wiring():
    rng = np.random.default_rng(48)
    depth = rng.uniform(size=50)
    response = rng.normal(size=(50, 20))
    weight = np.ones(50)
    original = analysis.cable.depth_dictionary(depth, 8)
    augmented = analysis.make_depth_with_common(depth, 8)
    np.testing.assert_allclose(analysis.cable.capture(original, response, weight),
                               analysis.cable.capture(augmented, response, weight), atol=1e-13)
    assert np.linalg.matrix_rank(augmented) == 8
    assert np.count_nonzero(augmented) > np.count_nonzero(original)


def test_duplicate_routes_do_not_count_as_independent_channels():
    rng = np.random.default_rng(4)
    route = np.arange(30) < 7
    d = np.column_stack([np.ones(30), route, route])
    answer = analysis.evaluate(d, rng.normal(size=(30, 9)), np.ones(30))
    assert answer["dictionary_rank"] == 2
    assert answer["nonzero_coefficients"] == 44
    assert answer["coverage"] == 1


def test_common_constrained_oracle_captures_expected_residual_singular_energy():
    rng = np.random.default_rng(6)
    weight = np.sqrt(rng.uniform(.4, 2, 30))
    weighted = rng.normal(size=(30, 10))
    q0 = weight / np.linalg.norm(weight)
    residual = weighted - q0[:, None]*(q0 @ weighted)[None, :]
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    d = np.column_stack([np.ones(30), u[:, :4] / weight[:, None]])
    answer = analysis.evaluate(d, weighted, weight)
    np.testing.assert_allclose(answer["residual_capture"], sum(singular[:4]**2)/sum(singular**2), atol=1e-13)
