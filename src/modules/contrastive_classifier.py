import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

from typing import Optional


class ContrastiveClassifier(nn.Module):
    def __init__(self, measurement_dim):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.device = 'cpu'
        self.f = nn.Linear(measurement_dim, measurement_dim, bias=False)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Expects a batch with dimensions (BATCH_SIZE x N_MEASUREMENTS x MEASUREMENT_DIM)
        """
        assert len(x.shape) == 3

        batch_size = x.shape[0]
        n_meas = x.shape[1]

        # Compute projection for each measurement and normalize them to unit hypersphere
        z = self.f(x)
        z = F.normalize(z, dim=2)

        # Compute dot-product between all pairs (batch-wise)
        dot_products = z @ z.permute(0, 2, 1)

        # Create mask to ignore diagonal elements (dot-products between vector and itself)
        mask = (torch.eye(n_meas, device=self.device).bool()).repeat(batch_size, 1, 1)

        # Create mask to ignore dot-products between vectors and any padding elements
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
            temp = padding_mask.repeat(1, n_meas, 1)
            padding_mask = temp | temp.transpose(1, 2)
        else:
            padding_mask = torch.zeros(batch_size, n_meas, n_meas, dtype=torch.bool, device=self.device)

        # All elements which are masked are set to -inf, corresponding to zero probability in the predictions
        masked_dots = dot_products.masked_fill(mask | padding_mask, -np.inf)

        probs = masked_dots.log_softmax(2)
        return probs

    def to(self, device):
        super().to(device)
        self.device = device

