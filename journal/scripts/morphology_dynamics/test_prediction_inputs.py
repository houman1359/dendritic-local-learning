"""Enforce that endpoint predictions do not request training/test observations."""
import unittest
from unittest.mock import patch

import investigate


class PredictionInputTests(unittest.TestCase):
    def test_predictors_request_only_calibration_examples(self):
        original = investigate.original
        dataset = original.dataset
        requests = []

        def guarded_dataset(task, kind, n):
            requests.append(kind)
            if kind != 'calibration':
                raise AssertionError(f'Predictor requested prohibited split: {kind}')
            return dataset(task, kind, n)

        cfg = original.load()
        task = dict(task_id='audit', seed=6100, rank=2, noise_sd=.75, angle_pi=.125, split='development')
        with patch.object(original, 'dataset', guarded_dataset):
            rows, _ = investigate.predictions(task, 'feedback_only', original.candidates(cfg), cfg)
        self.assertEqual(requests, ['calibration'])
        self.assertEqual(len(rows), 320)
        self.assertEqual({row['method'] for row in rows}, {
            'gaussian_plugin_sgd', 'gaussian_plugin_fullbatch', 'gaussian_oracle_sgd',
            'gaussian_oracle_fullbatch', 'empirical_split_fullbatch', 'original_scalar'})


if __name__ == '__main__':
    unittest.main()
