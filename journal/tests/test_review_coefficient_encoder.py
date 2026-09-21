import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_review_coefficient_encoder as experiment


def test_oracle_dictionary_matches_archived_subtree_rule():
    context = np.arange(8)
    actual = experiment.route_fields(np.eye(4)[context // 2])
    expected = experiment.task.grouped_routes(np.arange(8), 4, "correct_ancestry_subtrees",np.random.default_rng(0))
    np.testing.assert_allclose(actual,expected,atol=1e-14)


def test_noiseless_local_encoder_learns_group_without_task_labels():
    context=np.tile(np.arange(8),32)
    cue=np.eye(8)[context]
    cfg={"epochs":30,"batch_size":16,"learning_rate":.2,"weight_decay":.001}
    fitted=experiment.fit_encoder(cue,context,cfg,np.random.default_rng(1))
    prediction=experiment.coefficient_predictions(np.eye(8),fitted,0)
    np.testing.assert_array_equal(prediction.argmax(1),np.arange(8)//2)
    fields=experiment.route_fields(prediction)
    np.testing.assert_allclose(np.linalg.norm(fields,axis=1),1,atol=1e-14)
    np.testing.assert_allclose(fields[:,::2],fields[:,1::2],atol=1e-14)


def test_delay_moves_only_cue_not_task_or_target():
    cues=np.eye(8); w=np.arange(36).reshape(9,4)/100
    current=experiment.coefficient_predictions(cues,w,0)
    delayed=experiment.coefficient_predictions(cues,w,1)
    np.testing.assert_allclose(delayed,np.roll(current,1,axis=0))
