import torch
from torch import nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class CornerBoundingBoxLoss(nn.Module):
    def __init__(self):
        super(CornerBoundingBoxLoss,self).__init__()

    def forward(self, pred_corners, target_corners):
        """
        Args:
            pred_corners (torch.Tensor): Predicted corners of shape (batch_size, 8, 3).
            target_corners (torch.Tensor): Target corners of shape (batch_size, 8, 3).

        Returns:
            torch.Tensor: RMSE loss.
        """
        # Ensure both predicted and target corners have the same shape
        assert pred_corners.shape == target_corners.shape, \
            "Shape mismatch between predicted and target corners"
        eps = 1e-6
        criterion = nn.MSELoss(reduction='none')
        # loss_per_corner = torch.sqrt(criterion(pred_corners, target_corners)+eps)
        loss = criterion(pred_corners, target_corners)
        # loss = torch.mean(loss_per_corner, dim=(1, 2), keepdim=True)
        # loss = (loss.squeeze(1)).squeeze(1)
        return loss
        
        

