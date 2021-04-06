import unittest

import torch
import numpy as np
from numpy import log

from src.modules.contrastive_loss import ContrastiveLoss


class TestContrastiveLoss(unittest.TestCase):
    def test_simple_input(self):
        # Input
        unique_ids = \
            [
                [2, 0, 1, 2],
                [6, 7, 6, 6],
            ]
        classifications = \
            [
                [
                    [0.0, 0.3, 0.4, 0.3],
                    [0.3, 0.0, 0.5, 0.2],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.3, 0.2, 0.4, 0.0]
                ],
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.1, 0.4],
                    [0.5, 0.0, 0.0, 0.5],
                    [0.25, 0.25, 0.25, 0.25]
                ]
            ]
        unique_ids = torch.tensor(unique_ids)
        classifications = torch.tensor(classifications)
        log_classifications = classifications.log()

        # Expected outputs (computed partially by hand, steps shown below)
        id_matrix = \
        [
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5, 0.0]
            ]
        ]
        log_classifications_dot_ids = \
        [
            [
                [np.nan,   0.0,    0.0,    log(0.3)],
                [0.0,      np.nan, 0.0,    0.0],
                [np.nan,   np.nan, np.nan, 0.0],
                [log(0.3), 0.0,    0.0,    np.nan]
            ],
            [
                [np.nan,        np.nan, 0.5*log(0.5),  0.5*log(0.5)],
                [0.0,           np.nan, 0.0,           0.0],
                [0.5*log(0.5),  np.nan, np.nan,        0.5*log(0.5)],
                [0.5*log(0.25), 0.0,    0.5*log(0.25), 0.0]
            ]
        ]
        loss_batch_1 = - (log(0.3) + log(0.3))
        loss_batch_2 = - (0.5*log(0.5) + 0.5*log(0.5) + 0.5*log(0.5) + 0.5*log(0.5) + 0.5*log(0.25) + 0.5*log(0.25))
        n_eligible_measurements = 5  # number of measurements with non-zero losses (measurements for which at least one
        # other measurement is from the same object)
        expected_loss = (loss_batch_1 + loss_batch_2) / n_eligible_measurements

        # Actual outputs
        c_loss = ContrastiveLoss()
        actual_loss = c_loss(log_classifications, unique_ids)

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item())
