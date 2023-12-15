import warnings
import h5py
import os
from matplotlib.collections import PolyCollection
import numpy as np
import io
from PIL import Image, ImageOps
import pycocotools
import os, json, random
import pickle
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from physion_tools import PhysionPointCloudGenerator
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

global_object_types = set()
CRUCIAL_OBJECTS = [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']
CRUCIAL_OBJECTS_CLASS = {element:index for index, element in enumerate(CRUCIAL_OBJECTS)}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, help="Train or validation split")
    args = parser.parse_args()
    return args

def get_intrinsics_from_projection_matrix(proj_matrix, size=(256, 256)):
    H, W = size
    vfov = 2.0 * math.atan(1.0/proj_matrix[1][1]) * 180.0/ np.pi
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * H / float(H)
    fx = W / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = H / 2.0 / tan_half_vfov
    fl = fx
    sw = 1
    pix_T_cam = np.array([[fx, 0, W / 2.0],
                        [0, fy, H / 2.0],
                                [0, 0, 1]])
    return pix_T_cam, fl, sw

def get_bbox_from_seg_mask(seg_mask):
    a = np.where(seg_mask != 0)
    height = np.max(a[0]) - np.min(a[0])
    width = np.max(a[1]) - np.min(a[1])
    top_left = (np.min(a[1]), np.min(a[0]))
    return [top_left[0], top_left[1], width, height]

def plot_box(points, center):

    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

   # Compute the bounding box
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

    # Define the vertices of the bounding box
    vertices = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ])

    # Define the six faces of the bounding box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]

    # Plot the bounding box
    ax.add_collection3d(Poly3DCollection(faces, linewidths=1, edgecolors='r', alpha=0.1))
    box_dimensions = np.array([max_x - min_x, max_y - min_y, max_z - min_z])

    front, back, left, right, top, bottom = points
    width_val = abs(front[2] - back[2])
    length_val = abs(left[0] - right[0])
    height_val = abs(top[1] - bottom[1])

    plt.show()

def get_phys_dict(img_idx, _file,_file_idx,  filename, frame_id):

    file_frame_combined_name = filename.split(".hdf5")[0] + "_" + str(_file_idx) + "_" + frame_id
    s_obj = file["static"]
    f_obj = file["frames"]
    rgb = f_obj[frame_id]["images"]["_img"][:]
    image = Image.open(io.BytesIO(rgb))
    # image = ImageOps.mirror(image)
    RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval", "image", SPLIT) + "/" + file_frame_combined_name + ".jpg"
    image.save(RGB_IMG_PATH)

    img_width = image.size[0]
    img_height = image.size[1]
    num_segments_in_img = len(s_obj["object_segmentation_colors"][:])

    pcd_info = {
        'num_features': 6,
        'lidar_idx': img_idx
    }

    pts_path = 'points/{}/'.format(SPLIT) + file_frame_combined_name + '.bin'

    # TODO: Verify image_idx
    image_obj = {
        'image_idx': img_idx,
        'image_shape': [img_height, img_width],
        'image_path': 'image/{}/'.format(SPLIT) + file_frame_combined_name + '.jpg'
    }

    np_cam = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['camera_matrix'][:]), (4,4))
    np_proj_mat = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['projection_matrix'][:]), (4,4))
    
    # TODO: Verify
    pix_T_cam, _, _ = get_intrinsics_from_projection_matrix(np_proj_mat, (img_height, img_width))
    calib = {
        'K': pix_T_cam.astype(np.float32),
        'Rt': np_cam.astype(np.float32)
    }

    pcd_generator = PhysionPointCloudGenerator(hdf5_file_path=os.path.join(PHYSION_HDF5_ROOT, _file, filename), frame_number=frame_id, plot=False)
    pcd = pcd_generator.run()

    pcd.astype('float32').tofile(os.path.join(STORE_PATH_ROOT, pts_path))

    bbox_list = []
    location_list = []
    dimensions_list = []
    gt_boxes_upright_depth_list = []
    heading_ang = []
    names_list = []
    index_list = []

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
    
        seg = f_obj[frame_id]["images"]["_id"][:]
        image = Image.open(io.BytesIO(seg))
        # image = ImageOps.mirror(image)
        seg_numpy_arr = np.array(image)
        seg_mask = (seg_numpy_arr == seg_color).all(-1)
        seg_mask = seg_mask.astype(np.uint8)
        if not np.any(seg_mask):
            # NOTE: Some error in data for pilot_dominoes_0mid_d3chairs_o1plants_tdwroom_0001.hdf5, final seg mask empty
            warnings.warn('Missing segmentation mask for file: ' + filename + " at frame: " + frame_id) 
            continue
        
        bbox = get_bbox_from_seg_mask(seg_mask)
        bbox_list.append(bbox)

        location_list.append(f_obj[frame_id]["objects"]["center"][seg_id])

        front = f_obj[frame_id]["objects"]["front"][seg_id]
        back = f_obj[frame_id]["objects"]["back"][seg_id]
        width_val = abs(front[2] - back[2])

        left = f_obj[frame_id]["objects"]["left"][seg_id]
        right = f_obj[frame_id]["objects"]["right"][seg_id]
        length_val = abs(left[0] - right[0])

        top = f_obj[frame_id]["objects"]["top"][seg_id]
        bottom = f_obj[frame_id]["objects"]["bottom"][seg_id]
        height_val = abs(top[1] - bottom[1])

        center_x = f_obj[frame_id]["objects"]["center"][seg_id][0]
        center_y = f_obj[frame_id]["objects"]["center"][seg_id][1]
        center_z = f_obj[frame_id]["objects"]["center"][seg_id][2]

        points = [front, back, left, right, top, bottom]
        # center = [center_x, center_y, center_z]
        # plot_box(points, center)

        points = np.array(points)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

        bbox_3d_dims = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
        dimensions_list.append(bbox_3d_dims)


        #TODO: Check quartonion order
        [w,x,y,z] = f_obj[frame_id]["objects"]["rotations"][seg_id]

        #TODO Check is rotation_y == yaw?
        yaw = math.atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)
        heading_ang.append(yaw)

        #TODO Check (x, y, z, x_size, y_size, z_size, yaw) x_size, y_size, z_size ordering
        gt_boxes_upright_depth = [center_x, center_y, center_z, bbox_3d_dims[0], bbox_3d_dims[1], bbox_3d_dims[2], yaw]
        # gt_boxes_upright_depth = [center_x, center_y, center_z, length_val, height_val, width_val, yaw]
        gt_boxes_upright_depth_list.append(gt_boxes_upright_depth)
        names_list.append(obj_name.decode('utf-8'))
        index_list.append(CRUCIAL_OBJECTS_CLASS[obj_name])

    
        
    # # Visualize point cloud
    # vis = Visualizer(pcd, bbox3d=np.asarray(gt_boxes_upright_depth_list), mode="xyzrgb")
    # vis.show()

    
    num_segments_in_img = len(gt_boxes_upright_depth_list)
    annos = {
        'gt_num': num_segments_in_img,
        'name': np.asarray(names_list),
        'bbox': np.asarray(bbox_list),
        'location': np.asarray(location_list),
        'dimensions': np.asarray(dimensions_list),
        # TODO Verify rotation_y is one angle per object? [1 x num_objs]?
        'rotation_y': np.asarray(heading_ang),
        'index': np.asarray([i for i in range(num_segments_in_img)]),
        # 'class': np.asarray([0 for _ in range(num_segments_in_img)], dtype=np.int32),
        'class': np.asarray(index_list),
        'gt_boxes_upright_depth': np.asarray(gt_boxes_upright_depth_list)
    }

    assert len(gt_boxes_upright_depth_list) == annos['gt_num']

    return {
        'point_cloud': pcd_info,
        'pts_path': pts_path,
        'image': image_obj,
        'calib': calib,
        'annos': annos
    }
    
    # TODO: Verify depth coordinates

if __name__ == "__main__":
    
    args = parse_args()
    SPLIT = args.split
    PHYSION_HDF5_ROOT = "/tr3d_data/physion/HDF5/" + f"{SPLIT}"
    PREV_PKL_FILE_PATH = "/tr3d_data/physion" + f"/{SPLIT}.pkl"
    OBJ_TYPE_LIST_PATH = "/r3d_data/physion" + f"/{SPLIT}.txt"

    STORE_PATH_ROOT = "/tr3d_data/physion"
    data_infos = []
    img_idx = 0
    start = 50
    frames_per_vid = 20
    for _file_idx, _file in enumerate(sorted(os.listdir(PHYSION_HDF5_ROOT))):
        for filename in sorted((os.listdir(os.path.join(PHYSION_HDF5_ROOT, _file)))):
            if os.path.join(PHYSION_HDF5_ROOT, _file, filename).endswith('hdf5'):
                vid_hdf5_path = os.path.join(PHYSION_HDF5_ROOT, _file, filename) 
                print("Looking at : ", os.path.join(_file, filename))
                try:
                    with h5py.File(vid_hdf5_path, 'r') as file:
                        for frame_id in list(file["frames"].keys())[start : start + frames_per_vid]:
                            phys_dict = get_phys_dict(img_idx, _file, _file_idx, filename, frame_id)
                            print(filename, frame_id)
                            print("img_idx: ",img_idx)
                            img_idx += 1
                            data_infos.append(phys_dict)
                except OSError:
                    continue
    print("Storing {} pickle file ....".format(SPLIT) )
    with open(PREV_PKL_FILE_PATH, "wb") as pickle_file:
        pickle.dump(data_infos, pickle_file)
