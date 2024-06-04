# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp
import sys
sys.path.append("...") 
import mmcv
import numpy as np
from torch.utils.data import Dataset
import os
import io, math
from PIL import Image, ImageOps
import h5py
from ..core.bbox import get_box_type
from .builder import DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline
from physion.store_physion_data import  get_bbox_from_seg_mask, CRUCIAL_OBJECTS_CLASS, CRUCIAL_OBJECTS, get_rotation_matrix_from_quaternion, get_intrinsics_from_projection_matrix
from physion.external.rotation_continuity.utils import get_ortho6d_from_R, compute_rotation_matrix_from_ortho6d_np
from physion.physion_tools import PhysionPointCloudGenerator, PointCloudVisualizer, canonical_to_world_np, world_to_canonical_np, convert_to_world_coords, bbox_to_corners
from mmdet3d.core.bbox.structures.physion_box3d import Physion3DBoxes
import multiprocessing
from tqdm import tqdm
import concurrent
from mmdet3d.core.evaluation.indoor_eval import *
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from mmdet3d.core.points import BasePoints, get_points_type
warnings.filterwarnings("ignore")


@DATASETS.register_module()
class PhysionRandomFrameDataset(Dataset):
    """Physion HDF5 Dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    .. code-block:: none

    [
        {'sample_idx':
         'lidar_points': {'lidar_path': velodyne_path,
                           ....
                         },
         'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
                   'gt_bboxes_3d':  <np.ndarray> (n, 7)
                   'gt_names':  [list]
                   ....
               }
         'calib': { .....}
         'images': { .....}
        }
    ]

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Physion',
                 filter_empty_gt=True,
                 test_mode=False,
                 number_of_frames_per_file = 10,
                 file_client_args=dict(backend='disk')):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if self.test_mode: self.split = 'test' 
        else: self.split = 'train'
        self.total_length = 0
        self.annotations_cache = dict()
        self.iter_initialized = False
        self.file_client_args = file_client_args
        self.number_of_frames_per_file = number_of_frames_per_file
        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb')) 
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)
        # process pipeline

       
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def process_file_frame(self, file_path, img_idx, _file, frame_id, ann_dir):
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as file:
            # Call your processing function with necessary arguments
            return self.get_phys_dict(file, img_idx, _file, frame_id, ann_dir)
        
    def random_select_frames(self, data_files):
        sampled_frames = {}

        for data_file in data_files:
            if data_file[1] not in sampled_frames:
                sampled_frames[data_file[1]] = []
            sampled_frames[data_file[1]].append([data_file[0], data_file[1], data_file[2]])
        # Sample 10 frames from each file, ensuring all files are covered.
        sampled_data = []
        for file, frames in sampled_frames.items():
            if len(frames) >= self.number_of_frames_per_file:
                sampled_data.extend(random.sample(frames, 10))
            else:
                sampled_data.extend(frames)

        # Randomly shuffle the sampled data.
        random.shuffle(sampled_data)

        return sampled_data
    
    def _calculate_bounding_box_extents(self, bbox_corners):
        #NOTE: be wary of the axis
        min_coords = []
        max_coords = []
        for obj_bbox in bbox_corners:
            min_coords_obj = torch.min(obj_bbox, axis=0)
            max_coords_obj = torch.max(obj_bbox, axis=0)
            min_coords.append(min_coords_obj)
            max_coords.append(max_coords_obj)
        return min_coords, max_coords
    
    def _filter_density_and_boxes(self, gt_boxes, densities, threshold):
        filtered_gt_boxes = []
        filtered_densities = []

        for bbox, density in zip(gt_boxes, densities):
            if density >= threshold:
                filtered_gt_boxes.append(bbox)
                filtered_densities.append(density)

        return filtered_gt_boxes, filtered_densities
    
    def density_within_boxes(self, pcd, bbox_corners):
        min_coords, max_coords = self._calculate_bounding_box_extents(bbox_corners)
        density_points_per_obj = []
        for min_coord_per_obj, max_coord_per_obj in zip(min_coords, max_coords):
            points_within_bbox = pcd[(pcd[:, 0] >= min_coord_per_obj.values[0].numpy()) & (pcd[:, 0] <= max_coord_per_obj.values[0].numpy()) &
                                    (pcd[:, 1] >= min_coord_per_obj.values[1].numpy()) & (pcd[:, 1] <= max_coord_per_obj.values[1].numpy()) &
                                    (pcd[:, 2] >= min_coord_per_obj.values[2].numpy()) & (pcd[:, 2] <= max_coord_per_obj.values[2].numpy())]
            density = len(points_within_bbox)
            density_points_per_obj.append(density)
        return density_points_per_obj

    def get_phys_dict(self, file, img_idx, _file, frame_id):
        # file_frame_combined_name = _file.split(".hdf5")[0] + "_" + str(_file_idx) + "_" + frame_id
        file_info = {'_file' :os.path.basename(_file), 'frame_id': frame_id}
        s_obj = file["static"]
        f_obj = file["frames"]
        rgb = f_obj[frame_id]["images"]["_img_cam0"][:]
        image = Image.open(io.BytesIO(rgb))
        image = ImageOps.mirror(image)
        # RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval_new", "image", SPLIT) + "/" + file_frame_combined_name + ".jpg"
        # # RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval", "image") + "/" + file_frame_combined_name + ".jpg"
        # image.save(RGB_IMG_PATH)

        img_width = image.size[0]
        img_height = image.size[1]
        num_segments_in_img = len(s_obj["object_segmentation_colors"][:])

        pcd_info = {
            'num_features': 6,
            'lidar_idx': img_idx 
        }

        # pts_path = 'points_new/{}/'.format(SPLIT) + file_frame_combined_name + '.bin'
        # pts_path = 'points/' + file_frame_combined_name + '.bin'

        # TODO: Verify image_idx
        # image_obj = {
        #     'image_idx': img_idx,
        #     'image_shape': [img_height, img_width],
        #     # 'image' : np.array(image)
        #     # 'image_path': 'image/{}/'.format(SPLIT) + file_frame_combined_name + '.jpg'
        # }

        np_cam = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['camera_matrix_cam0'][:]), (4,4))
        np_proj_mat = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['projection_matrix_cam0'][:]), (4,4))
        
        # TODO: How to remove the labels(bboxes) which have very small amount of points in them?
        pix_T_cam, _, _ = get_intrinsics_from_projection_matrix(np_proj_mat, (img_height, img_width))
        # calib = {
        #     'K': pix_T_cam.astype(np.float32),
        #     'Rt': np_cam.astype(np.float32)
        # }

        pcd_generator = PhysionPointCloudGenerator(hdf5_file_path=os.path.join(_file), frame_number=frame_id, plot=False)
        pcd_points = pcd_generator.run()
        if pcd_points is None: return
        # pcd_points.astype('float32').tofile(os.path.join(STORE_PATH_ROOT, pts_path))
        
        # pcd_new = open3d_convert_rgbd_to_pcd(pix_T_cam, np.array(image), pcd_generator.get_depth_values(f_obj[frame_id]["images"]["_depth_cam0"][:], width=512, height=512, near_plane=0.1, far_plane=100))
        
        bbox_list = []
        location_list = []
        dimensions_list = []
        gt_boxes_upright_depth_list = []
        heading_ang = []
        # names_list = []
        # index_list = []
        # bbox_points_list = []

        for seg_id in range(num_segments_in_img):

            obj_name = s_obj['model_names'][:][seg_id]
            if obj_name in s_obj['distractors'][:]:
                continue
            if obj_name in s_obj['occluders'][:]:
                continue
            if obj_name not in CRUCIAL_OBJECTS:
                continue

            seg_color = s_obj["object_segmentation_colors"][seg_id]
            # object_name = s_obj['model_names'][seg_id].decode('utf-8')
            # Adding to the set in order to see different types of objects
        
            seg = f_obj[frame_id]["images"]["_id_cam0"][:]
            image = Image.open(io.BytesIO(seg))
            image = ImageOps.mirror(image)
            seg_numpy_arr = np.array(image)
            seg_mask = (seg_numpy_arr == seg_color).all(-1)
            seg_mask = seg_mask.astype(np.uint8)
            
            if not np.any(seg_mask):
                # NOTE: Some error in data for pilot_dominoes_0mid_d3chairs_o1plants_tdwroom_0001.hdf5, final seg mask empty
                # warnings.warn('Missing segmentation mask for file: ' + _file + " at frame: " + frame_id) 
                continue

            bbox = get_bbox_from_seg_mask(seg_mask)
            bbox_list.append(bbox)

            location_list.append(f_obj[frame_id]["objects"]["center_cam0"][seg_id])
            # center = f_obj[frame_id]["objects"]["center_cam0"][seg_id]
            # convert_camera_to_world(np_cam, np_proj_mat, center)
            
            front = f_obj[frame_id]["objects"]["front_cam0"][seg_id]
            back = f_obj[frame_id]["objects"]["back_cam0"][seg_id]
            # width_val = abs(front[2] - back[2])

            left = f_obj[frame_id]["objects"]["left_cam0"][seg_id]
            right = f_obj[frame_id]["objects"]["right_cam0"][seg_id]
            # length_val = abs(left[0] - right[0])

            top = f_obj[frame_id]["objects"]["top_cam0"][seg_id]
            bottom = f_obj[frame_id]["objects"]["bottom_cam0"][seg_id]
            # height_val = abs(top[1] - bottom[1])

            center_x = f_obj[frame_id]["objects"]["center_cam0"][seg_id][0]
            center_y = f_obj[frame_id]["objects"]["center_cam0"][seg_id][1]
            center_z = f_obj[frame_id]["objects"]["center_cam0"][seg_id][2]
            

            #TODO: Check quartonion order
            [x,y,z, w] = f_obj[frame_id]["objects"]["rotations_cam0"][seg_id]
            # t = f_obj[frame_id]["objects"]["positions_cam0"][seg_id]
            # scale = s_obj['scale'][seg_id]
            R = get_rotation_matrix_from_quaternion(x,y,z,w)
            ortho6d = get_ortho6d_from_R(R)
            # R_from_ortho = compute_rotation_matrix_from_ortho6d_np(ortho6d.reshape(1, 6)).squeeze(0)
            points = [front, back, left, right, top, bottom]
            points = np.array(points)

            def calculate_bounding_box_dimensions(points):
                
                points = [np.array(point) for point in points]
                length = np.linalg.norm(points[0] - points[1])
                width = np.linalg.norm(points[2] - points[3])
                height = np.linalg.norm(points[4] - points[5])

                return [length, width, height]
            # import pdb; pdb.set_trace()
            bbox_3d_dims = calculate_bounding_box_dimensions(points)
            if bbox_3d_dims[2] < 0.1 or bbox_3d_dims[0] < 0.1 or bbox_3d_dims[1] < 0.1: continue # based on height/width/length remove carpet

           
            dimensions_list.append([bbox_3d_dims[1], bbox_3d_dims[2], bbox_3d_dims[2]])
                
            yaw = math.atan2(2.0*(y*z + x*y), w*w + x*x - y*y - z*z)
            heading_ang.append(yaw)
            # [x, y, z, w, h, l, 6d representation of R]
            #TODO: need to change to scale and rotation
            # gt_boxes_upright_depth = [center_x , center_y, center_z, bbox_3d_dims[1], bbox_3d_dims[2], bbox_3d_dims[0]] + ortho6d.tolist()
            #  SCALE : [w,h,l] / [x,z,y]
            gt_boxes_upright_depth = [center_x, center_y, center_z, bbox_3d_dims[1], bbox_3d_dims[2], bbox_3d_dims[0]] + ortho6d.tolist()
            # gt_boxes_upright_depth = [t[0], t[1], t[2], bbox_3d_dims[1], bbox_3d_dims[2], bbox_3d_dims[0]] + ortho6d.tolist()
            # gt_boxes_upright_depth = [center, front, back, left, right, top, bottom]
            # bbox_points = [center, front, top, back, bottom, left, right]
    
            if len(gt_boxes_upright_depth) == 0 : import pdb; pdb.set_trace()
            gt_boxes_upright_depth_list.append(gt_boxes_upright_depth)
            # bbox_points_list.append(bbox_points)
            # names_list.append(obj_name.decode('utf-8'))
            # index_list.append(CRUCIAL_OBJECTS_CLASS[obj_name])


        """ 
        center = [0, 0.5, 0]
        front = [0, 0.5, 0.5]
        top = [0, 1, 0]
        back = [0, 0.5, -0.5]
        bottom = [0,0,0]
        left = [-0.5, 0.5, 0]
        right = [0.5, 0.5, 0]
        """
        
        # canonical_values = {"center":[0, 0.5, 0],
        #                     "front":[0, 0.5, 0.5],
        #                     "top":[0, 1, 0],
        #                     "back":[0, 0.5, -0.5],
        #                     "bottom":[0,0,0],
        #                     "left":[-0.5, 0.5, 0],
        #                     "right":[0.5, 0.5, 0]}
        
        

        # # gt_world_coords = convert_to_world_coords(gt_boxes_upright_depth_list)
        # # # gt_world_coords = bbox_to_corners(torch.tensor(gt_boxes_upright_depth_list))
        # # visualizer = PointCloudVisualizer()
        # visualizer.visualize_point_cloud_and_bboxes(pcd_points, gt_world_coords, use_points=True, show=True)
        '''
        bbox_corners = bbox_to_corners(torch.tensor(gt_boxes_upright_depth_list))
        density_list = self.density_within_boxes(pcd_points, bbox_corners)
        #TODO: Need to test on filter threshold
        gt_boxes_upright_depth_list, density_list = self._filter_density_and_boxes(gt_boxes_upright_depth_list, density_list, threshold=1000)
        '''
        # visualizer = PointCloudVisualizer()
        # filtered_bbox_corners = bbox_to_corners(torch.tensor(gt_boxes_upright_depth_list))
        # visualizer.visualize_point_cloud_and_bboxes(pcd_points, filtered_bbox_corners, use_points=True, show=True)
        num_segments_in_img = len(gt_boxes_upright_depth_list)
        annos = {
            'gt_num': num_segments_in_img,
            # 'name': np.asarray(names_list),
            'name': np.asarray(['object' for i in range(num_segments_in_img)]),
            # 'bbox': np.asarray(bbox_list),
            # 'location': np.asarray(location_list),
            # 'dimensions': np.asarray(dimensions_list),
            # 'rotation_y': np.asarray(heading_ang),
            'index': np.asarray([i for i in range(num_segments_in_img)]),
            'class': np.asarray([0 for _ in range(num_segments_in_img)], dtype=np.int32),
            # 'class': np.asarray(index_list),
            'gt_boxes_upright_depth': np.asarray(gt_boxes_upright_depth_list)
        }

        assert len(gt_boxes_upright_depth_list) == annos['gt_num']
        return {
            'file_info': file_info,
            'point_cloud': pcd_info,
            'points': pcd_points,
            # 'image': image_obj,
            # 'calib': calib,
            'annos': annos
        }
   
    def load_annotations(self, ann_file):
        """Load annotations from ann_dir.

        Args:
            ann_dir (str): Path of the annotation directory.

        Returns:
            list[dict]: List of annotations.
        """
        print('\n')
        print('Loading File information .....')
        print('\n')
        return mmcv.load(ann_file, file_format='pkl')


    def get_gt_data_info(self, index):
        raw_info = self.data_infos[index]
        with h5py.File(raw_info[1], 'r') as file:
            info = self.get_phys_dict(file, raw_info[0], raw_info[1], raw_info[2])
        if info is None: return
        sample_idx = info['point_cloud']['lidar_idx']
        # assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)
        input_dict['file_info'] = info['file_info']
        if self.modality['use_lidar']:
            # pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['points'] =  self._points_class_builder(info['points'])
            # input_dict['pts_filename'] = pts_filename
            # input_dict['file_name'] = pts_filename


        # if self.modality['use_camera']:
        #     img_filename = osp.join(
        #         osp.join(self.data_root, 'sunrgbd_trainval'),
        #         info['image']['image_path'])
        #     input_dict['img_prefix'] = None
        #     input_dict['img_info'] = dict(filename=img_filename)
        #     calib = info['calib']
        #     rt_mat = calib['Rt']
        #     # follow Coord3DMode.convert_point
        #     rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
        #                        ]) @ rt_mat.transpose(1, 0)
        #     depth2img = calib['K'] @ rt_mat
        #     input_dict['depth2img'] = depth2img

    
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
            return None
        return input_dict
    

    
    def indoor_eval_physion(self,gt_annos,
                    dt_annos,
                    metric,
                    label2cat,
                    logger=None,
                    box_type_3d=None,
                    box_mode_3d=None):
        """Indoor Evaluation modified for physion

        Evaluate the result of the detection.

        Args:
            gt_annos (list[dict]): Ground truth annotations.
            dt_annos (list[dict]): Detection annotations. the dict
                includes the following keys

                - labels_3d (torch.Tensor): Labels of boxes.
                - boxes_3d (:obj:`BaseInstance3DBoxes`):
                    3D bounding boxes in Depth coordinate.
                - scores_3d (torch.Tensor): Scores of boxes.
            metric (list[float]): IoU thresholds for computing average precisions.
            label2cat (dict): Map from label to category.
            logger (logging.Logger | str, optional): The way to print the mAP
                summary. See `mmdet.utils.print_log()` for details. Default: None.

        Return:
            dict[str, float]: Dict of results.
        """
 
        assert len(dt_annos) == len(gt_annos)
        pred = {}  # map {class_id: pred}
        gt = {}  # map {class_id: gt}
        for img_id in range(len(dt_annos)):
            det_anno = dt_annos[img_id]
            gt_anno = gt_annos[img_id]
            if self.get_gt_data_info(gt_anno) is not None:
                gt_anno = self.get_gt_data_info(gt_anno)['ann_info']
            else:
                continue

            if gt_anno is None:
                # import pdb; pdb.set_trace()
                continue

            # NOTE: TESTING ONLY!!!! REMOVE WHEN NOT TESTING!!!
            
            # det_anno['labels_3d'] = torch.tensor(gt_anno['gt_labels_3d'])
            # det_anno['boxes_3d'] = gt_anno['gt_bboxes_3d']
            # det_anno['scores_3d'] = torch.ones(len(gt_anno['gt_labels_3d']))
            
         
            for i in range(len(det_anno['labels_3d'])):
                label = det_anno['labels_3d'].numpy()[i]
                bbox = det_anno['boxes_corners'][i]
                score = det_anno['scores_3d'].numpy()[i]

                if int(label) not in pred:
                    pred[int(label)] = {}
                if img_id not in pred[int(label)]:
                    pred[int(label)][img_id] = []
                if int(label) not in gt:
                    gt[int(label)] = {}
                if img_id not in gt[int(label)]:
                    gt[int(label)][img_id] = []
                pred[int(label)][img_id].append((bbox, score))

            
            if len(gt_anno['gt_labels_3d']) > 0:
                gt_boxes = gt_anno['gt_bboxes_3d']
                labels_3d = gt_anno['gt_labels_3d']
            else:
                gt_boxes = box_type_3d(np.array([], dtype=np.float32))
                labels_3d = np.array([], dtype=np.int64)
            # if gt_anno['gt_num'] != 0:
            #     gt_boxes = box_type_3d(
            #         gt_anno['gt_boxes_upright_depth'],
            #         box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
            #         origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            #     labels_3d = gt_anno['class']
            # else:
            #     gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            #     labels_3d = np.array([], dtype=np.int64)

            for i in range(len(labels_3d)):
                label = labels_3d[i]
                bbox = gt_boxes[i]
                if label not in gt:
                    gt[label] = {}
                if img_id not in gt[label]:
                    gt[label][img_id] = []
                gt[label][img_id].append(bbox)

        rec, prec, ap = eval_map_recall_corners(pred, gt, metric)
        ret_dict = dict()
        header = ['classes']
        table_columns = [[label2cat[label]
                        for label in ap[0].keys()] + ['Overall']]

        for i, iou_thresh in enumerate(metric):
            header.append(f'AP_{iou_thresh:.2f}')
            header.append(f'AR_{iou_thresh:.2f}')
            rec_list = []
            for label in ap[i].keys():
                ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap[i][label][0])
            ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap[i].values())))

            table_columns.append(list(map(float, list(ap[i].values()))))
            table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

            for label in rec[i].keys():
                ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                    rec[i][label][-1])
                rec_list.append(rec[i][label][-1])
            ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.mean(rec_list))

            table_columns.append(list(map(float, rec_list)))
            table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        return ret_dict
    
    
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - points : Points (N x 6)
                - ann_info (dict): Annotation info.
        """
        raw_info = self.data_infos[index]
        with h5py.File(raw_info[1], 'r') as file:
            info = self.get_phys_dict(file, raw_info[0], raw_info[1], raw_info[2])
        if info is None: return
        sample_idx = info['point_cloud']['lidar_idx']
        # assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)
        input_dict['file_info'] = info['file_info']
        if self.modality['use_lidar']:
            # pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['points'] =  self._points_class_builder(info['points'])
            # input_dict['pts_filename'] = pts_filename
            # input_dict['file_name'] = pts_filename


        # if self.modality['use_camera']:
        #     img_filename = osp.join(
        #         osp.join(self.data_root, 'sunrgbd_trainval'),
        #         info['image']['image_path'])
        #     input_dict['img_prefix'] = None
        #     input_dict['img_info'] = dict(filename=img_filename)
        #     calib = info['calib']
        #     rt_mat = calib['Rt']
        #     # follow Coord3DMode.convert_point
        #     rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
        #                        ]) @ rt_mat.transpose(1, 0)
        #     depth2img = calib['K'] @ rt_mat
        #     input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict
    
    def _points_class_builder(self, points):
        attribute_dims = {'color': [3, 4, 5]}
        points_class = get_points_type('DEPTH')
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        return points


   
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        raw_info = self.data_infos[index]
        with h5py.File(raw_info[1], 'r') as file:
            info = self.get_phys_dict(file, raw_info[0], raw_info[1], raw_info[2])
    
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
            
        ######################################################################################################################################################   
        #Testing the new physion box structure
        gt_bboxes_3d = Physion3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        ###################################################################################################################################################### 
        
        # to target box structure
        # gt_bboxes_3d = DepthInstance3DBoxes(
        #     gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['bboxes'] = gt_bboxes_2d
            anns_results['labels'] = gt_labels_3d

        return anns_results


    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    
    def get_gt_annos(self, index):
        gt_data_info = self.get_gt_data_info(index)
        return gt_data_info['ann_info'] if gt_data_info is not None else None
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return {}
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving json
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 indices_to_consider,
                 metric=None,
                 iou_thr=(0.25, 0.5, 0.1),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # from mmdet3d.core.evaluation import indoor_eval
        # from mmdet3d.core.evaluation.indoor_eval import indoor_eval_physion
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        # import pdb; pdb.set_trace()
        # if len(results) != len(self.data_infos): import pdb; pdb.set_trace()
        
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = indices_to_consider
        # for i in range(len(self.data_infos)):
        #     gt_annos.append(i)
        # gt_annos = [info['annos'] for info in self.data_infos]
        assert len(results) == len(indices_to_consider)
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        # ret_dict = indoor_eval(
        #     gt_annos,
        #     results,
        #     iou_thr,
        #     label2cat,
        #     logger=logger,
        #     box_type_3d=self.box_type_3d,
        #     box_mode_3d=self.box_mode_3d)
        # import pdb; pdb.set_trace()
        ret_dict = self.indoor_eval_physion(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def random_sample(self):
        self.data_infos = self.random_select_frames(self.data_infos)
        print("Dataset Sampled!!")
        
    def copy(self):
        copied_dataset = PhysionRandomFrameDataset(
            data_root=self.data_root,
            ann_file=self.ann_file,
            pipeline=self.pipeline,
            classes=self.CLASSES,
            modality=self.modality,
            box_type_3d=self.box_type_3d,
            filter_empty_gt=self.filter_empty_gt,
            test_mode=self.test_mode,
            number_of_frames_per_file=self.number_of_frames_per_file,
            file_client_args=self.file_client_args
        )
        return copied_dataset
    

