import torch
from torch import nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class PointDistanceLoss(nn.Module):
    def __init__(self):
        super(PointDistanceLoss, self).__init__()

    def forward(self, pred_points, target_points):
        """
        Args:
            pred_points (torch.Tensor): Predicted points of shape (batch_size, num_points, 3).
            target_points (torch.Tensor): Target points of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: Point distance loss.
        """
        # Ensure both predicted and target points have the same shape
        assert pred_points.shape == target_points.shape, \
            "Shape mismatch between predicted and target points"
        
        # Compute pairwise distances between predicted and target points
        pairwise_distances = torch.cdist(pred_points, target_points, p=2)

        # Compute the mean distance across all points in each sample of the batch
        mean_distance = torch.mean(pairwise_distances, dim=(1, 2), keepdim=True)

        return mean_distance.squeeze()  # Remove the extra dimensions

