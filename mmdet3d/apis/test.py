# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs
import h5py
import numpy as np
from PIL import Image, ImageOps
import io 
from physion.store_physion_data import  get_bbox_from_seg_mask, CRUCIAL_OBJECTS_CLASS, CRUCIAL_OBJECTS, get_rotation_matrix_from_quaternion, get_intrinsics_from_projection_matrix
from physion.external.rotation_continuity.utils import get_ortho6d_from_R, compute_rotation_matrix_from_ortho6d_np
from physion.physion_tools import PhysionPointCloudGenerator, PointCloudVisualizer, canonical_to_world_np, world_to_canonical_np, convert_to_world_coords, bbox_to_corners
from mmdet3d.core.bbox.structures.physion_box3d import Physion3DBoxes
import os, math
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def get_phys_dict(file, img_idx, _file, frame_id):
        # file_frame_combined_name = _file.split(".hdf5")[0] + "_" + str(_file_idx) + "_" + frame_id
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
        
        # TODO: Verify
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
            # import pdb; pdb.set_trace() 
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
        
        
        # # import pdb; pdb.set_trace()
        # # gt_world_coords = convert_to_world_coords(gt_boxes_upright_depth_list)
        # # # gt_world_coords = bbox_to_corners(torch.tensor(gt_boxes_upright_depth_list))
        # # visualizer = PointCloudVisualizer()
        # visualizer.visualize_point_cloud_and_bboxes(pcd_points, gt_world_coords, use_points=True, show=True)

        
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
            'point_cloud': pcd_info,
            'points': pcd_points,
            # 'image': image_obj,
            # 'calib': calib,
            'annos': annos
        }


def get_info(data):
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
    raw_info = data
    with h5py.File(raw_info[1], 'r') as file:
        info = get_phys_dict(file, raw_info[0], raw_info[1], raw_info[2])
    info['filename'] = os.path.splitext(os.path.basename(raw_info[1]))[0]
    return info

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, (data, data_infos) in enumerate(zip(data_loader, vars(dataset)['data_infos'])):
        data_infos = get_info(data_infos)
        gt_data = data_infos['annos']['gt_boxes_upright_depth']
        data['filename'] = data_infos['filename']
        if len(gt_data) == 0:
            import pdb; pdb.set_trace()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    gt_bbox=gt_data,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
