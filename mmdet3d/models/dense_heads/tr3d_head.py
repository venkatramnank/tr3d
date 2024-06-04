try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
import mmdet3d.models.losses as losses
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from physion.external.rotation_continuity.utils import compute_rotation_matrix_from_ortho6d
from physion.physion_tools import PointCloudVisualizer, convert_to_world_coords_torch
from physion.physion_nms import *
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="/home/kalyanav/MS_thesis/mmdetection3d/work_dirs/tr3d_physion_config/summary_logs")


@HEADS.register_module()
class TR3DHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 train_cfg=None,
                 test_cfg=None):
        super(TR3DHead, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_classes, in_channels, n_reg_outs)

    def _init_layers(self, n_classes, in_channels, n_reg_outs):
        self.bbox_conv= ME.MinkowskiConvolution(
            in_channels, n_reg_outs , kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))


    # per level
    def _forward_single(self, x):
        # nx = self.bbox_conv1(x)
        reg_final = self.bbox_conv(x).features
        # reg_distance = torch.exp(reg_final[:, 3:6])
        reg_distance = reg_final[:, 3:6]
        reg_center = reg_final[:, :3]
        #TODO: Need to fix this for accepting 6d 
        # reg_angle = reg_final[:, 6:]
        reg_6d_ortho = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_center, reg_distance, reg_6d_ortho), dim=1)
        cls_pred = self.cls_conv(x).features
        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    
    #################***************************##########################
    
    # center
    def _forward_single_center(self, x):
        # nx = self.bbox_conv1(x)
        reg_final = self.bbox_conv(x).features
        # import pdb; pdb.set_trace()
        bbox_pred = reg_final
        cls_pred = self.cls_conv(x).features
        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            #TODO: voxelization? is that the issue?
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points
    
    #################***************************##########################
    
    
    def forward(self, x):
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    
    @staticmethod
    def bbox_to_corners(bbox, pos_points=None):
        #TODO: Use this in iou calculation as well
        """Converts the center, h,w,l,ortho6d format into corners

        Args:
            bbox (Tensor): Input of size (N, 12)
            pos_points : Final locations of shape (N, 3)

        Returns:
            Tensor: Corners (including the center) of shape (N, 8, 3)
        """

        if bbox.shape[-1] != 12:
            return bbox
        
        if bbox.numel() == 0:
            return torch.empty([0, 8, 3], device=bbox.device)
        center = bbox[:, :3]
        if pos_points is not None:
            new_center = center + pos_points
        else:
            new_center = center.clone()
        dims = bbox[:, 3:6]
        #TODO: check order of corners with GT (does not matter)
        #TODO: rotation cannot exceed 90 degree check
        corners_norm = torch.stack([
            torch.Tensor([0.5, 0.5, 0.5]),
            torch.Tensor([0.5, 0.5, -0.5]),
            torch.Tensor([0.5, -0.5, 0.5]),
            torch.Tensor([0.5, -0.5, -0.5]),
            torch.Tensor([-0.5, 0.5, 0.5]),
            torch.Tensor( [-0.5, 0.5, -0.5]),
            torch.Tensor([-0.5, -0.5, 0.5]),
            torch.Tensor([-0.5, -0.5, -0.5])            
        ]).to(device=dims.device, dtype=dims.dtype)
       
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        b = corners.shape[0]
        rotation_matrix = compute_rotation_matrix_from_ortho6d(bbox[:, 6:])
        # import pdb; pdb.set_trace()
        # rotation_matrix = torch.eye(3).unsqueeze(0).expand(b, -1, -1).to(corners.device)
        #TODO: R to Identity
        #Top left corner front, bottom right behind, bottom right front
        #Huber loss (not rmse)
        corners = rotation_matrix@corners.transpose(1,2)
        corners = corners.permute(0,2,1) 
        corners += new_center.view(-1, 1, 3)
        corners_center = torch.concat([corners, new_center.unsqueeze(1)], dim=1)
        # import pdb; pdb.set_trace()
        # # #NOTE: Enable gradient tracking
        # corners.requires_grad_(True)

        # def hook_fn(grad):
        #     print('Gradients passing through bbox_to_corners:', grad)
        # # import pdb; pdb.set_trace()
        # # Register the hook to the tensor
        # corners.register_hook(hook_fn)
        # import pdb; pdb.set_trace()
        
        # print(corners.grad_fn)
        # print(rotation_matrix.grad_fn)
        # corners.retain
        # import pdb; pdb.set_trace()
        
        if corners is not None and corners.grad is not None: import pdb; pdb.set_trace()
        return corners_center
 
    
    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        #NOTE: Mobius OBB parametrization
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0] #NOTE: This addition of points with the offset
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox
        
        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    #########***********************************************************################
    def _center_to_points(self, centers, pos_points=None):
        if pos_points is not None:
            return centers + pos_points
        else:
            return centers
    
    
    # Only centers
    def _loss_single_centers(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.tensor[:,:3], gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            # import pdb; pdb.set_trace()
            if type(self.bbox_loss).__name__ == 'ChamferDistance':
                 # in terms of chamfer distance you get two losses: from source and target. So we add them up!!    
                src_loss, dst_loss = self.bbox_loss(
                                    self._center_to_points(pos_bbox_preds, pos_points),
                                    self._center_to_points(pos_bbox_targets),
                                    # self.bbox_to_corners(pos_bbox_targets) 
                                    )

                bbox_loss = (src_loss + dst_loss).reshape(1)
            else:
                bbox_loss = self.bbox_loss(
                self._center_to_points(pos_bbox_preds, pos_points),
                self._center_to_points(pos_bbox_targets)
                )  
        

        else:
            bbox_loss = None

        return bbox_loss, cls_loss, pos_mask
    
    #########***********************************************************################
    
    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.tensor[:,:3], gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]

            # if pos_bbox_preds.shape[1] == 6:
            #     pos_bbox_targets = pos_bbox_targets[:, :6]

            # bbox_loss = self.bbox_loss(PhysionRandomFrameDataset
            #     self._bbox_to_loss(
            #         self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
            #     self._bbox_to_loss(pos_bbox_targets)) 
            # print(points)
            if type(self.bbox_loss).__name__ == 'ChamferDistance':
                 # in terms of chamfer distance you get two losses: from source and target. So we add them up!!    
                src_loss, dst_loss = self.bbox_loss(
                                    self.bbox_to_corners(pos_bbox_preds, pos_points),
                                    self.bbox_to_corners(pos_bbox_targets),
                                    # self.bbox_to_corners(pos_bbox_targets) 
                                    )

                bbox_loss = (src_loss + dst_loss).reshape(1)
                         
            else:
                # import pdb; pdb.set_trace()
                bbox_loss = self.bbox_loss(
                self.bbox_to_corners(pos_bbox_preds, pos_points),
                self.bbox_to_corners(pos_bbox_targets)
                # self.bbox_to_corners(pos_bbox_targets) 
                )  
                          

        else:
            bbox_loss = None
        # print('bbox_loss: ', bbox_loss)
        # import pdb; pdb.set_trace()
        if bbox_loss is not None and bbox_loss.grad is not None: import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return bbox_loss, cls_loss, pos_mask
    ################*******************************************************************########################
    def _add_pos_points(self, bbox, pos_points):
        bbox[:, :3] = bbox[:, :3] + pos_points
        return bbox
    
    def _loss_single_12d(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.tensor[:,:3], gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            bbox_loss = self.bbox_loss(self._add_pos_points(pos_bbox_preds, pos_points), pos_bbox_targets)
            # center_loss, scale_loss, rotation_loss = self.bbox_loss(self._add_pos_points(pos_bbox_preds, pos_points), pos_bbox_targets)
            
        else:
            # center_loss, scale_loss, rotation_loss = None
            bbox_loss = None
        # print('bbox_loss: ', bbox_loss)
        # import pdb; pdb.set_trace()
        # if bbox_loss is not None and bbox_loss.grad is not None: import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return bbox_loss, cls_loss, pos_mask
    
    ################********************************************************************########################
    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses, pos_masks = [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
        # import pdb; pdb.set_trace()
        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)))
        

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points = self(x)
        
        losses =  self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas)

        return losses

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """

        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
    
    def _nms_corners(self, bboxes_corners, scores, img_meta, bbox_preds):
        """Multi-class NMS for a single scene using corner representations.
        Args:
            bboxes_corners (Tensor): Predicted boxes in corner representation of shape (N_boxes, 12).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes in corner representation.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        nms_bboxes_corners, nms_scores, nms_labels, nms_boxes_pred = [], [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes_corners = bboxes_corners[ids][:, :8, :] # not including the center
            nms_function = self.nms3d_corners
            nms_ids = nms_function(class_bboxes_corners, class_scores, self.test_cfg.iou_thr)
            nms_boxes_pred.append(bbox_preds[nms_ids])
            nms_bboxes_corners.append(class_bboxes_corners[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes_corners.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes_corners):
            nms_bboxes_corners = torch.cat(nms_bboxes_corners, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
            nms_boxes_pred = torch.cat(nms_boxes_pred, dim=0)
        else:
            nms_bboxes_corners = bboxes_corners.new_zeros((0, bboxes_corners.shape[1]))
            nms_scores = bboxes_corners.new_zeros((0,))
            nms_labels = bboxes_corners.new_zeros((0,))

        # if yaw_flag:
        #     box_dim = 12
        #     with_yaw = True
        # else:
        #     raise ValueError("Non-yaw representation not supported for corners.")

        # # Convert corners back to original representation
        # nms_bboxes = corners_to_bboxes(nms_bboxes_corners)

        return nms_bboxes_corners, nms_scores, nms_labels, nms_boxes_pred

    # def _get_bboxes_single_corners(self, bbox_preds_corners, cls_preds, points, img_meta):
    #     scores = torch.cat(cls_preds).sigmoid()
    #     bbox_preds_corners = torch.cat(bbox_preds_corners)
    #     points = torch.cat(points)
    #     max_scores, _ = scores.max(dim=1)
    #     labels = []
    #     n_classes = scores.shape[1]
    #     for i in range(n_classes):
    #         labels.append(
    #             bbox_preds_corners.new_full(
    #                 scores.shape, i, dtype=torch.long))
    #     labels = torch.cat(labels, dim=0)

    #     boxes_corners = bbox_preds_corners  # Assuming bbox_preds_corners are already in corner representation

    #     boxes, scores, labels = self._nms_corners(boxes_corners, scores, img_meta)
    #     return boxes, scores, labels

    def nms3d_corners(self, boxes_corners, scores, iou_threshold: float):
        """3D NMS function using corners representation."""
        assert boxes_corners.size(1) == 8, 'Input boxes corners shape should be (N, 8, 3)'
        order = scores.sort(0, descending=True)[1]

        boxes_corners = boxes_corners[order].contiguous()
        desired_order = torch.LongTensor([1,5,7,3,0,4,6,2]).to(boxes_corners.device)
        rearranged_boxes_corners = torch.index_select(boxes_corners, dim=1, index=desired_order)
        try:
            # intersection_vol, iou_3d_vals = iou_3d(rearranged_boxes_corners, rearranged_boxes_corners, eps=1e-6)
            _, iou_3d_vals = filtered_box3d_overlap(rearranged_boxes_corners, rearranged_boxes_corners, eps=1e-6)
        except ValueError or RuntimeError or TypeError:
            import pdb; pdb.set_trace()
        keep = torch.ones(scores.size(0), dtype=torch.bool, device=boxes_corners.device)
        keep &= filter_boxes(rearranged_boxes_corners,eps=1e-6).to(boxes_corners.device)

        # Iterate over each box
        try:
            for i in range(scores.size(0)):
                if keep[i]:
                    # For each box, compare its IoU with all other boxes
                    for j in range(i + 1, scores.size(0)):
                        if iou_3d_vals[i, j] > iou_threshold:
                            # If IoU exceeds threshold, discard box with lower score
                            if scores[i] < scores[j]:
                                keep[i] = False
                            else:
                                keep[j] = False
        except RuntimeError:
            import pdb; pdb.set_trace()

        # Filter out indices of boxes to keep
        keep_indices = order[keep].contiguous()
        return keep_indices

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid() #TODO: is this the scoring ? Is this alright?
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)
        labels = []
        n_classes = scores.shape[1]
        for i in range(n_classes):
            labels.append(
                bbox_preds.new_full(
                    scores.shape, i, dtype=torch.long))
        labels = torch.cat(labels, dim=0)

        # if len(scores) > self.test_cfg.nms_pre > 0:
        #     _, ids = max_scores.topk(self.test_cfg.nms_pre)
        #     bbox_preds = bbox_preds[ids]
        #     scores = scores[ids]
        #     points = points[ids]
        #     labels = labels[ids]
        
        # boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        
        # print(points)
        boxes_corners = self.bbox_to_corners(bbox_preds, points)
        #TODO: Need to fix NMS for 3d
        boxes_corners, scores, labels, bbox_preds = self._nms_corners(boxes_corners, scores, img_meta, bbox_preds)
        # boxes, scores, labels = self._nms(boxes, scores, img_meta)
        # return boxes, scores, labels
        bbox_preds = img_meta['box_type_3d'](
            bbox_preds,
            box_dim=12,
            with_ortho6d=True)
        return bbox_preds, scores, labels, boxes_corners
    
    #################*************************##################
    
    def _get_bboxes_single_center(self, bbox_preds, cls_preds, points, img_meta):
        # import pdb; pdb.set_trace()
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)
        labels = []
        n_classes = scores.shape[1]
        for i in range(n_classes):
            labels.append(
                bbox_preds.new_full(
                    scores.shape, i, dtype=torch.long))
        labels = torch.cat(labels, dim=0)

        boxes_centers = self._center_to_points(bbox_preds, pos_points=points)

        bbox_preds = img_meta['box_type_3d'](
            boxes_centers,
            box_dim=3)
        return bbox_preds, scores, labels
    
    #################*************************##################
    

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)


@BBOX_ASSIGNERS.register_module() 
class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8) # new tensor 10^8
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))]) #  levels using size of all points across the indices, i.e if two levels, then 0 and 1 values
        points = torch.cat(points) #combining points
        n_points = len(points) 
        n_boxes = len(gt_bboxes) # number of bboxes, usually the objects in that image scene

        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.tensor[:, :3], gt_bboxes.tensor[:, 3:]), dim=1) #essentially the tensor
        
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 12) 
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes) # in our case all 1s
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes) 
        level_condition = label_levels == point_levels
        
        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3] # n_points, n_boxes, 3
        
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1) # sum of square of distances of points from center 
        center_distances = torch.where(level_condition, center_distances, float_max) #where the label levels and point levels match, calulate this distance
        
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1] # k = min(number of points, top pts threshold)
        
        topk_condition = center_distances < topk_distances.unsqueeze(0) # wherever the center distances is less that topk distances

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1) #minimum distance indices of closest center for each point, which bounding  box is nearest to each point

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max) # only topk points distances are kept 
        min_values, min_ids = center_distances.min(dim=1)# then min indices are then calculated considering only top k distances
        min_inds = torch.where(min_values < float_max, min_ids, -1) 
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1) # compare to closest center and only retain then

        return min_inds