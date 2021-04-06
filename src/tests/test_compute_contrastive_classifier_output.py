import unittest

import torch
from torch import nn
import numpy as np

from src.modules.contrastive_classifier import ContrastiveClassifier


class TestComputeContrastiveClassifierOutput(unittest.TestCase):

    def test_batch_input(self):
        # Input
        x = [
                [
                    [5.0, 9.0], [4.0, 8.0], [3.0, 3.0], [1.0, 1.0]
                ],
                [
                    [9.0, 2.0], [8.0, 9.0], [6.0, 3.0], [3.0, 0.0]
                ]
            ]
        x = torch.tensor(x, dtype=torch.float32)
        f_w = torch.tensor([[1, 2], [-3, 0]], dtype=torch.float32)
        batch_size, n_measurements, measurement_dim = x.shape

        # Expected output (computed partially by hand, steps shown below)
        z = \
            [
                [
                    [23, -15.0], [20, -12], [9, -9], [3, -3]
                ],
                [
                    [13, -27], [26, -24], [12, -18], [3, -9]
                ]
            ]
        z_norm = \
            [
                [
                    [0.8376, -0.5463], [0.8575, -0.5145], [0.7071, -0.7071], [0.7071, -0.7071]
                ],
                [
                    [0.4338, -0.9010], [0.7348, -0.6783], [0.5547, -0.8321], [0.3162, -0.9487]
                ]
            ]
        # Compute dot products between all z's in a batch (dot-product of a vector with itself should be -inf for the
        # softmax to ignore it)
        dot_prods = \
            [
                [
                    [-np.inf,  0.999313, 0.978555, 0.978555],
                    [0.999313, -np.inf,  0.970141, 0.970141],
                    [0.978555, 0.970141, -np.inf,  0.999980],
                    [0.978555, 0.970141, 0.999980, -np.inf ]
                ],
                [
                    [-np.inf,  0.929904, 0.990351, 0.991946],
                    [0.929904, -np.inf,  0.972006, 0.875847],
                    [0.990351, 0.972006, -np.inf,  0.964809],
                    [0.991946, 0.875847, 0.964809, -np.inf ]
                ]
            ]
        softmaxes = \
            [
                [
                    [0.0000, 0.3380, 0.3310, 0.3310],
                    [0.3398, 0.0000, 0.3301, 0.3301],
                    [0.3319, 0.3291, 0.0000, 0.3391],
                    [0.3319, 0.3291, 0.3391, 0.0000]
                ],
                [
                    [0.0000, 0.3199, 0.3398, 0.3403],
                    [0.3344, 0.0000, 0.3488, 0.3168],
                    [0.3382, 0.3321, 0.0000, 0.3297],
                    [0.3492, 0.3109, 0.3399, 0.0000]
                ]
            ]
        softmaxes = torch.tensor(softmaxes)

        # Real output
        cc = ContrastiveClassifier(measurement_dim=measurement_dim)
        cc.f.weight = nn.Parameter(data=f_w, requires_grad=True)
        y = cc(x).exp()

        # Assert that expect and real output match
        self.assertTrue(torch.allclose(softmaxes, y, rtol=0.001), msg=f'Expected: {softmaxes}\nActual: {y}')

    def test_batch_input_with_masks(self):
        # Input
        x = [
                [
                    [5.0, 9.0], [4.0, 8.0], [3.0, 3.0], [1.0, 1.0]
                ],
                [
                    [9.0, 2.0], [8.0, 9.0], [6.0, 3.0], [3.0, 0.0]
                ]
            ]
        x = torch.tensor(x, dtype=torch.float32)
        f_w = torch.tensor([[1, 2], [-3, 0]], dtype=torch.float32)
        batch_size, n_measurements, measurement_dim = x.shape
        padding_mask = \
            [
                [False, False, True, True],
                [False, False, False, True]
            ]
        padding_mask = torch.tensor(padding_mask)

        # Expected output (computed partially by hand, steps shown below)
        z = \
            [
                [
                    [23, -15.0], [20, -12], [9, -9], [3, -3]
                ],
                [
                    [13, -27], [26, -24], [12, -18], [3, -9]
                ]
            ]
        z_norm = \
            [
                [
                    [0.8376, -0.5463], [0.8575, -0.5145], [0.7071, -0.7071], [0.7071, -0.7071]
                ],
                [
                    [0.4338, -0.9010], [0.7348, -0.6783], [0.5547, -0.8321], [0.3162, -0.9487]
                ]
            ]
        # Compute dot products between all z's in a batch
        # dot-product of a vector with itself (or a padding element) should be -inf for the softmax to ignore it
        dot_prods = \
            [
                [
                    [-np.inf,  0.999313, -np.inf, -np.inf],
                    [0.999313, -np.inf,  -np.inf, -np.inf],
                    [-np.inf,  -np.inf,  -np.inf, -np.inf],
                    [-np.inf,  -np.inf,  -np.inf, -np.inf]
                ],
                [
                    [-np.inf,  0.929904, 0.990351, -np.inf],
                    [0.929904, -np.inf,  0.972006, -np.inf],
                    [0.990351, 0.972006, -np.inf,  -np.inf],
                    [-np.inf,  -np.inf,  -np.inf,  -np.inf]
                ]
            ]
        softmaxes = \
            [
                [
                    [0.0000, 1.0000, 0.0000, 0.0000],
                    [1.0000, 0.0000, 0.0000, 0.0000],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan]
                ],
                [
                    [0.0000, 0.4849, 0.5151, 0.0000],
                    [0.4895, 0.0000, 0.5105, 0.0000],
                    [0.5046, 0.4954, 0.0000, 0.0000],
                    [np.nan, np.nan, np.nan, np.nan]
                ]
            ]
        softmaxes = torch.tensor(softmaxes)

        # Real output
        cc = ContrastiveClassifier(measurement_dim=measurement_dim)
        cc.f.weight = nn.Parameter(data=f_w, requires_grad=True)
        pred = cc(x, padding_mask=padding_mask)
        y = pred.exp()

        # Assert that expect and real output match
        self.assertTrue(torch.allclose(softmaxes, y, rtol=0.001, equal_nan=True), msg=f'Expected: {softmaxes}\nActual: {y}')

