import h5py
import os
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

SPLIT = "train"
PHYSION_HDF5_ROOT = "/home/kashis/Desktop/Eval7/tr3d/data/physion/Dominoes_testing_HDF5s/Dominoes/" + f"{SPLIT}"
PREV_PKL_FILE_PATH = "/home/kashis/Desktop/Eval7/tr3d/data/physion" + f"/{SPLIT}.pkl"

PHYSION_RGB_PATH = "/home/kashis/Desktop/Eval7/tr3d/physion"

STORE_PATH_ROOT = "/home/kashis/Desktop/Eval7/tr3d/data/physion/"

def meshgrid2d_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def get_depth_values(image: np.array, depth_pass: str = "_depth", width: int = 256, height: int = 256, near_plane: float = 0.1, far_plane: float = 100) -> np.array:
    """
    Get the depth values of each pixel in a _depth image pass.
    The far plane is hardcoded as 100. The near plane is hardcoded as 0.1.
    (This is due to how the depth shader is implemented.)
    :param image: The image pass as a numpy array.
    :param depth_pass: The type of depth pass. This determines how the values are decoded. Options: `"_depth"`, `"_depth_simple"`.
    :param width: The width of the screen in pixels. See output data `Images.get_width()`.
    :param height: The height of the screen in pixels. See output data `Images.get_height()`.
    :param near_plane: The near clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the near clipping plane.
    :param far_plane: The far clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the far clipping plane.
    :return An array of depth values.
    """
    image = np.flip(np.reshape(image, (height, width, 3)), 1)

    # Convert the image to a 2D image array.
    if depth_pass == "_depth":
        depth_values = np.array((image[:, :, 0] + image[:, :, 1] / 256.0 + image[:, :, 2] / (256.0 ** 2)))
    elif depth_pass == "_depth_simple":
        depth_values = image[:, :, 0] / 256.0
    else:
        raise Exception(f"Invalid depth pass: {depth_pass}")
    # Un-normalize the depth values.
    return (depth_values * ((far_plane - near_plane) / 256.0)).astype(np.float32)

def split_intrinsics(pix_T_cam):
    fx, fy, x0, y0 = pix_T_cam[0,0], pix_T_cam[1,1], pix_T_cam[0,2], pix_T_cam[1,2]
    return fx, fy, x0, y0

def Pixels2Camera_np(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    # there is no randomness here

    B, H, W = list(z.shape)

    fx = np.reshape(fx, [B,1,1])
    fy = np.reshape(fy, [B,1,1])
    x0 = np.reshape(x0, [B,1,1])
    y0 = np.reshape(y0, [B,1,1])

    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)

    x = np.reshape(x, [B,-1])
    y = np.reshape(y, [B,-1])
    z = np.reshape(z, [B,-1])
    xyz = np.stack([x,y,z], axis=2)
    return xyz

def depth2pointcloud_np(z, pix_T_cam):
    B, C, H, W = list(z.shape)  # this is 1, 1, H, W
    y, x = meshgrid2d_py(H, W)
    y = np.repeat(y[np.newaxis, :, :], B, axis=0)
    x = np.repeat(x[np.newaxis, :, :], B, axis=0)
    z = np.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera_np(x, y, z, fx, fy, x0, y0)
    return xyz

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

def get_phys_dict(img_idx, filename, frame_id):

    file_frame_combined_name = filename.split(".hdf5")[0] + "_" + frame_id

    s_obj = file["static"]
    f_obj = file["frames"]
    rgb = f_obj[frame_id]["images"]["_img"][:]
    image = Image.open(io.BytesIO(rgb))
    image = ImageOps.mirror(image)
    RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval", "image") + "/" + file_frame_combined_name + ".jpg"
    image.save(RGB_IMG_PATH)

    img_width = image.size[0]
    img_height = image.size[1]
    num_segments_in_img = len(s_obj["object_segmentation_colors"][:])

    pcd_info = {
        'num_features': 6,
        'lidar_idx': 1
    }

    pts_path = 'points/' + file_frame_combined_name + '.bin'

    image_obj = {
        'image_idx': 1,
        'image_shape': [img_height, img_width],
        'image_path': 'image/' + file_frame_combined_name + '.jpg'
    }

    np_cam = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['camera_matrix'][:]), (4,4))
    np_proj_mat = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['projection_matrix'][:]), (4,4))
    
    # TODO: Verify
    pix_T_cam, _, _ = get_intrinsics_from_projection_matrix(np_proj_mat, (img_height, img_width))
    calib = {
        'K': pix_T_cam,
        'Rt': np_cam
    }

    # TODO: Verify depth to point cloud
    depth_img = f_obj[frame_id]["images"]["_depth"][:]
    depth_trans_img = get_depth_values(depth_img, width=img_width, height=img_height)
    pcd = depth2pointcloud_np(depth_trans_img.reshape((1, 1, img_width, img_height)), pix_T_cam)[0]

    # Visualize point cloud
    vis = Visualizer(pcd)
    vis.show()

    pcd.astype('float32').tofile(os.path.join(STORE_PATH_ROOT, pts_path))

    bbox_list = []
    location_list = []
    dimensions_list = []
    gt_boxes_upright_depth_list = []
    heading_ang = []

    for seg_id in range(num_segments_in_img):
        seg_color = s_obj["object_segmentation_colors"][seg_id]
        seg = f_obj[frame_id]["images"]["_id"][:]
        image = Image.open(io.BytesIO(seg))
        image = ImageOps.mirror(image)
        seg_numpy_arr = np.array(image)
        seg_mask = (seg_numpy_arr == seg_color).all(-1)
        seg_mask = seg_mask.astype(np.uint8)
        if not np.any(seg_mask):
            # NOTE: Some error in data for pilot_dominoes_0mid_d3chairs_o1plants_tdwroom_0001.hdf5, final seg mask empty
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

        dim = [height_val, width_val, length_val]
        dimensions_list.append(dim)

        center_x = f_obj[frame_id]["objects"]["center"][seg_id][0]
        center_y = f_obj[frame_id]["objects"]["center"][seg_id][1]
        center_z = f_obj[frame_id]["objects"]["center"][seg_id][2]

        #TODO: Check quartonion order
        [w,x,y,z] = f_obj[frame_id]["objects"]["rotations"][seg_id]

        #TODO Check is rotation_y == yaw?
        yaw = math.atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)
        heading_ang.append(yaw)

        #TODO Check (x, y, z, x_size, y_size, z_size, yaw) x_size, y_size, z_size ordering
        gt_boxes_upright_depth = [center_x, center_y, center_z, length_val, height_val, width_val, yaw]
        gt_boxes_upright_depth_list.append(gt_boxes_upright_depth)
        

    annos = {
        'gt_num': num_segments_in_img,
        'name': np.asarray(['object' for _ in range(num_segments_in_img)]),
        'bbox': np.asarray(bbox_list),
        'location': np.asarray(location_list),
        'dimensions': np.asarray(dimensions_list),
        # TODO Verify rotation_y is one angle per object? [1 x num_objs]?
        'rotation_y': np.asarray(heading_ang),
        'index': np.asarray([i for i in range(num_segments_in_img)]),
        'class': np.asarray([0 for _ in range(num_segments_in_img)]),
        'gt_boxes_upright_depth': np.asarray(gt_boxes_upright_depth_list)
    }

    return {
        'point_cloud': pcd_info,
        'pts_path': pts_path,
        'image': image_obj,
        'calib': calib,
        'annos': annos
    }

    # TODO: Verify depth coordinates




data_infos = []
img_idx = 0
start = 50
frames_per_vid = 50
for filename in (sorted(os.listdir(PHYSION_HDF5_ROOT))):
    vid_hdf5_path = os.path.join(PHYSION_HDF5_ROOT, filename)
    with h5py.File(vid_hdf5_path, 'r') as file:
        for frame_id in list(file["frames"].keys())[start : start + frames_per_vid]:
            phys_dict = get_phys_dict(img_idx, filename, frame_id)
            print(filename, frame_id)
            img_idx += 1
            data_infos.append(phys_dict)

print("")
with open(PREV_PKL_FILE_PATH, "wb") as pickle_file:
    pickle.dump(data_infos, pickle_file)
