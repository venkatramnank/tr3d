# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector
from tools.data_converter.voxelize_mlpointconvformer import *
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from physion.physion_tools import PointCloudVisualizer, convert_to_world_coords


@DETECTORS.register_module()
class MinkSingleStage3DDetector(Base3DDetector):
    r"""Single stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        neck (dict): Config of the neck.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 head,
                 voxel_size,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkSingleStage3DDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.init_weights()

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""

    def extract_feats(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
       # voxelization with voxel size of 0.05 
        # points = [p[voxelize(p[:, :3], self.voxel_size)] for p in points]  # [65536 x 6] [] [] ... b #TODO: visualize it once
        # import pdb; pdb.set_trace()
        # visualizer = PointCloudVisualizer()
        # visualizer.visualize_point_cloud_and_bboxes(points[0].cpu().numpy(), gt_bboxes_3d[0].corners.cpu().numpy(), corners=gt_bboxes_3d[0].corners.reshape(gt_bboxes_3d[0].tensor.shape[0]*8,3).cpu().numpy(), use_points=True, center=gt_bboxes_3d[0].tensor.cpu().numpy()[:,:3], show=True)
        # visualizer.visualize_point_cloud_and_bboxes(points[0].cpu().numpy(), gt_bboxes[0].tensor.cpu().numpy())
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device) 
        # coordinates, features = ME.utils.batch_sparse_collate(
        #     [(p[:, :3], p[:, 3:]) for p in points],
        #     device=points[0].device) 
        #collates all the points in the batch. Total number of points x 4 [batch number, x, y, z]
        # features shape is (total number of points x 3)[r,g,b]
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        ######################################################################################
        # import pdb; pdb.set_trace()
        # # visualizing the points and gt_bboxes_3d to make sure it is alright
        # visualizer = PointCloudVisualizer()
        # visualizer.visualize_point_cloud_and_bboxes(points[0].cpu().numpy(), gt_bboxes_3d[0].corners.cpu().numpy(), corners=gt_bboxes_3d[0].corners.reshape(gt_bboxes_3d[0].tensor.shape[0]*8,3).cpu().numpy(), use_points=True, center=gt_bboxes_3d[0].tensor.cpu().numpy()[:,:3], show=True)
        # visualizer.visualize_point_cloud_and_bboxes(points[0].cpu().numpy(), gt_bboxes_3d[0].corners.cpu().numpy(), use_points=True, center=gt_bboxes_3d[0].tensor.cpu().numpy()[:,:3], show=True)
        #####################################################################################
        x = self.extract_feats(points)
        losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d,
                                         img_metas)
        return losses


    def _box3dcornertoresult(self, bbox_corners, cls_preds):
        result_dict = dict(
            boxes_corners = bbox_corners.to('cpu'),
            cls_preds = [cls_preds[0].to('cpu'), cls_preds[1].to('cpu')]
        )
        return result_dict

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """

        
        x = self.extract_feats(points)
        bbox_list = self.head.forward_test(x, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels, boxes_corners)
            for bboxes, scores, labels, boxes_corners in bbox_list
        ]
        
        # NOTE: for center only
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ]
        # bbox_results = [self._box3dcornertoresult(bbox_corners, cls_preds)
        #                 for bbox_corners, cls_preds in bbox_list]
        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
