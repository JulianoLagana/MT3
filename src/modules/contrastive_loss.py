import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)

    def forward(self, log_classifications, unique_ids) -> Tensor:
        batch_size, n_measurements = unique_ids.shape

        temp = unique_ids.unsqueeze(1).repeat(1, n_measurements, 1)
        id_matrix = (temp == temp.permute(0, 2, 1)).float()

        # Mask diagonal and then perform row-wise normalization
        mask = (torch.eye(n_measurements, device=self.device).bool()).repeat(batch_size, 1, 1)
        id_matrix = id_matrix.masked_fill(mask, 0.0)
        id_matrix = F.normalize(id_matrix, p=1, dim=2)

        # Compute element-wise multiplication between log_classifications and id_matrix (NaNs -> 0.0)
        per_measurement_losses = -log_classifications * id_matrix
        mask = torch.isnan(per_measurement_losses)
        per_measurement_losses = per_measurement_losses.masked_fill(mask, 0.0)
        per_measurement_losses = per_measurement_losses.flatten(0, 1)  # get rid of batch dimension, all measurements are born equal

        # Compute loss
        per_measurement_loss = per_measurement_losses.sum(dim=1)
        n_eligible_measurements = (per_measurement_loss != 0.0).sum()  # number of measurements with non-zero losses,
        # i.e. no. of measurements for which at least one other measurement is from the same object.
        loss = per_measurement_loss.sum()/n_eligible_measurements
        return loss
