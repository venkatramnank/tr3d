# 12D representation
# L2 loss for scale, Huber loss for center, geodesic loss for rotation matrix
# All rotation based calculations is based off on ortho6d so all functions are adapted from https://github.com/papagina/RotationContinuity
import torch
from torch import nn as nn

from mmdet.models.losses.utils import weighted_loss
from physion.external.rotation_continuity.utils import compute_rotation_matrix_from_ortho6d
from ..builder import LOSSES

@LOSSES.register_module()
class l2_geodesic_loss(nn.Module):
    def __init__(self):
        super(l2_geodesic_loss, self).__init__()

    def compute_geodesic_distance_from_two_matrices(self, m1, m2):
        import pdb; pdb.set_trace()
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1)/2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(
            torch.ones(batch).cuda())*-1)
        theta = torch.acos(cos)
        import pdb; pdb.set_trace()
        # theta = torch.min(theta, 2*np.pi - theta)
        return theta

    def l2_loss(self, pred, target):
        # L2 Squared loss
        eps = 1e-6
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(pred, target) + eps
        return loss

    def huber_loss(self, pred, target):
        # Huber loss
        eps = 1e-6
        criterion = nn.HuberLoss(reduction='none')
        loss = criterion(pred, target) 
        return loss

    def rotation_geodesic_loss(self, pred, target):
        gt_r_matrix = compute_rotation_matrix_from_ortho6d(target)
        out_r_matrix = compute_rotation_matrix_from_ortho6d(pred)
        theta = self.compute_geodesic_distance_from_two_matrices(
            gt_r_matrix, out_r_matrix)
        error = theta.mean()
        return error

    def forward(self, pred, target):
        """
        Args:
            pred_corners (torch.Tensor): Predicted corners of shape (batch_size, 12).
            target_corners (torch.Tensor): Target corners of shape (batch_size, 12).

        Returns:
            torch.Tensor: Combined loss of center, scale and rotation.
        """
        alpha = 0.2
        beta = 0.2
        gamma = 0.6
        pred_center = pred[:, :3]
        target_center = target[:, :3]
        pred_scale = pred[:, 3:6]
        target_scale = target[:, 3:6]
        pred_ortho6d = pred[:, 6:]
        target_ortho6d = target[:, 6:]
        # import pdb; pdb.set_trace()
        # if torch.isnan(self.huber_loss(pred_center, target_center)) or torch.isnan(self.huber_loss(pred_scale, target_scale)) :
        #     import pdb; pdb.set_trace()
        # final_error = alpha * self.huber_loss(pred_center, target_center) + beta * self.huber_loss(
        #     pred_scale, target_scale) + gamma*self.rotation_geodesic_loss(pred_ortho6d, target_ortho6d)
        # center_loss = self.huber_loss(pred_center, target_center)
        # scale_loss = self.huber_loss(pred_scale, target_scale)
        # rotation_loss = self.rotation_geodesic_loss(pred_ortho6d, target_ortho6d)
        final_error = self.huber_loss(pred, target)
        return final_error
        # return center_loss, scale_loss, rotation_loss
