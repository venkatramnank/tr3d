import torch
from torch import nn as nn
from ..builder import LOSSES
from ..model_utils.emd.emd import earth_mover_distance

@LOSSES.register_module()
class CornerBoundingBoxEMDLoss(nn.Module):
    '''
    Earth mover distance loss using implementation from https://github.com/daerduoCarey/PyTorchEMD
    '''
    def __init__(self):
        super(CornerBoundingBoxEMDLoss,self).__init__()

    def forward(self, pred_corners, target_corners):
        """
        Args:
            pred_corners (torch.Tensor): Predicted corners of shape (batch_size, 8, 3).
            target_corners (torch.Tensor): Target corners of shape (batch_size, 8, 3).

        Returns:
            torch.Tensor: EMD loss.
        """
        # Ensure both predicted and target corners have the same shape
        assert pred_corners.shape == target_corners.shape, \
            "Shape mismatch between predicted and target corners"
        loss = earth_mover_distance(pred_corners, target_corners, transpose=False)
        return loss
        
        

