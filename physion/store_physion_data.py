import warnings
import h5py
import os
import torch
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
from physion.physion_tools import PhysionPointCloudGenerator, PointCloudVisualizer, canonical_to_world_np, world_to_canonical_np, convert_to_world_coords, bbox_to_corners
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from physion.external.rotation_continuity.utils import get_ortho6d_from_R, compute_rotation_matrix_from_ortho6d_np

global_object_types = set()
CRUCIAL_OBJECTS = [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']
CRUCIAL_OBJECTS_CLASS = {element:index for index, element in enumerate(CRUCIAL_OBJECTS)}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, help="Train or validation split")
    args = parser.parse_args()
    return args

def get_intrinsics_from_projection_matrix(proj_matrix, size=(512, 512)):
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
    
    
def convert_camera_to_world(camera_matrix, projection_matrix, obj_2D):
    """
    Convert 2D coordinates in camera space to 3D coordinates in world space.

    Args:
        camera_matrix (numpy.ndarray): Camera matrix for the transformation from image to camera coordinates.
        projection_matrix (numpy.ndarray): Projection matrix for the transformation from camera to world coordinates.
        obj_2D (numpy.ndarray): Array of 2D coordinates in camera space with shape (num_points, 3),
                                where each row represents (x, y, depth).

    Returns:
        numpy.ndarray: Array of transformed 3D coordinates with shape (num_points, 3).
    """

    if len(obj_2D.shape) == 1:
        obj_2D = obj_2D.reshape(1, 3)
    # Ensure the input array has the correct shape
    if obj_2D.shape[1] != 3:
        raise ValueError("obj_2D should have shape (num_points, 3)")
    
    # Rearrange the input array to match the expected order of coordinates
    obj_2D = np.concatenate([obj_2D[:, 1:2], obj_2D[:, 0:1], obj_2D[:, 2:3]], axis=1).astype(np.float32)

    # Normalize and scale coordinates
    obj_2D[:, 1] = 1 - obj_2D[:, 1] / camera_matrix[1, 1]
    obj_2D[:, 0] = obj_2D[:, 0] / camera_matrix[0, 0]
    obj_2D[:, :2] = obj_2D[:, :2] * 2 - 1

    # Transform 2D coordinates to 3D using projection matrix
    obj_3D = np.concatenate([obj_2D[:, :2] * obj_2D[:, 2:3],
                             obj_2D[:, 2:3],
                             (obj_2D[:, 2:3] - 1.0 * projection_matrix[2, 3]) / projection_matrix[2, 2] * projection_matrix[3, 2]],
                             axis=1)

    # Invert projection matrix to get world coordinates
    obj_3D = np.linalg.inv(projection_matrix) @ obj_3D.T

    # Invert camera matrix to transform to world coordinates
    obj_3D = (np.linalg.inv(camera_matrix) @ obj_3D).T

    return obj_3D[:, :3]

def open3d_convert_rgbd_to_pcd(pix_t_cam, img_array, dep_array):
    """Use open3d to convert RGBD to PCD

    Args:
        pix_t_cam (np.ndarray): Intirnsics from Projection matrix 
        img_array (np.ndarray): RGB image
        dep_array (np.ndarray): Depth Image
    """
    intrinsic = o3d.camera.PinholeCameraIntrinsic(512,512,pix_t_cam[0,0],pix_t_cam[1,1],pix_t_cam[0,2],pix_t_cam[1,2])
    img = o3d.geometry.Image(img_array.astype(np.uint8))
    depth = o3d.geometry.Image(dep_array.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd_points = np.asarray(pcd.points)
    pcd_color = np.asarray(pcd.colors)
    return np.concatenate([pcd_points, pcd_color], axis=1)


def create_3d_bbox(center, length, width, height, yaw):
    # Create a rotation matrix based on yaw angle
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])

    # Define the corners of the bounding box in object space
    corners = np.array([[-length / 2, -width / 2, -height / 2],
                        [length / 2, -width / 2, -height / 2],
                        [length / 2, width / 2, -height / 2],
                        [-length / 2, width / 2, -height / 2],
                        [-length / 2, -width / 2, height / 2],
                        [length / 2, -width / 2, height / 2],
                        [length / 2, width / 2, height / 2],
                        [-length / 2, width / 2, height / 2]])

    # Apply rotation to the corners
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Translate the rotated corners to the specified center
    translated_corners = rotated_corners + center

    # Define the edges of the bounding box
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a line set for the bounding box
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(translated_corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    return line_set





def visualize_bbox_points(points, colors, bbox_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    bbox_colors = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    pcd.points.extend(o3d.utility.Vector3dVector(bbox_points))
    pcd.colors.extend(o3d.utility.Vector3dVector(bbox_colors))
    o3d.visualization.draw_geometries([pcd])



#########################################################################################################################


# def visualize_bbox_points_with_lines(points, colors, corner_box):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Create a LineSet for bounding box edges
#     lines = [[0, 1], [1, 2], [2, 3], [0, 3],
#              [4, 5], [5, 6], [6, 7], [4, 7],
#              [0, 4], [1, 5], [2, 6], [3, 7]]

#     # Use the same color for all lines
#     line_colors = [[1, 0, 0] for _ in range(len(lines))]

#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(corner_box)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(line_colors)

#     # Create a visualization object and window
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     # Display the point cloud and bounding box edges
#     vis.add_geometry(pcd)
#     vis.add_geometry(line_set)

#     # Run the visualization loop
#     vis.run()




# code to convert [x,y,z,h,w,l,r] ---> bbox
# def box_center_to_corner(box):
#     # To return
#     corner_boxes = np.zeros((8, 3))

#     translation = box[0:3]
#     h, w, l = box[3], box[4], box[5]
#     rotation = box[6]

#     # Create a bounding box outline
#     bounding_box = np.array([
#         [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
#         [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
#         [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

#     # Standard 3x3 rotation matrix around the Z axis
#     rotation_matrix = np.array([
#         [np.cos(rotation), -np.sin(rotation), 0.0],
#         [np.sin(rotation), np.cos(rotation), 0.0],
#         [0.0, 0.0, 1.0]])

#     # Repeat the [x, y, z] eight times
#     eight_points = np.tile(translation, (8, 1))

#     # Translate the rotated bounding box by the
#     # original center position to obtain the final box
#     corner_box = np.dot(
#         rotation_matrix, bounding_box) + eight_points.transpose()

#     return corner_box.transpose()

def apply_transform(points, R, t, scale):
    """
    Apply rotation, translation, and scaling to a set of 3D points.

    Args:
    - points (numpy.ndarray): 3D points matrix, each column is a point.
    - R (numpy.ndarray): 3x3 rotation matrix.
    - t (numpy.ndarray): 1x3 translation vector.
    - scale (numpy.ndarray): 1x3 scaling factors for each axis.

    Returns:
    - transformed_points (numpy.ndarray): Transformed 3D points matrix.
    """
    # Ensure input data is in the correct format
    points = np.array(points)
    R = np.array(R)
    t = np.array(t)
    scale = np.array(scale)

    # Create a scaling matrix
    scale_matrix = np.diag(scale)

    # Apply scaling, rotation, and translation
    transformed_points = np.dot(R, points.T).T + t
    return transformed_points

def get_rotation_matrix(x,y,z,w):
    return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])


def get_rotation_matrix_from_quaternion(x,y,z,w):
    rot = R.from_quat([x,y,z,w])
    return (rot.as_matrix())
    


def get_phys_dict(file, img_idx, _file,_file_idx, frame_id):
    file_frame_combined_name = _file.split(".hdf5")[0] + "_" + str(_file_idx) + "_" + frame_id
    s_obj = file["static"]
    f_obj = file["frames"]
    rgb = f_obj[frame_id]["images"]["_img_cam0"][:]
    image = Image.open(io.BytesIO(rgb))
    image = ImageOps.mirror(image)
    RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval_new", "image", SPLIT) + "/" + file_frame_combined_name + ".jpg"
    # RGB_IMG_PATH = os.path.join(STORE_PATH_ROOT, "phys_trainval", "image") + "/" + file_frame_combined_name + ".jpg"
    image.save(RGB_IMG_PATH)

    img_width = image.size[0]
    img_height = image.size[1]
    num_segments_in_img = len(s_obj["object_segmentation_colors"][:])

    pcd_info = {
        'num_features': 6,
        'lidar_idx': img_idx + 9361
    }

    pts_path = 'points_new/{}/'.format(SPLIT) + file_frame_combined_name + '.bin'
    # pts_path = 'points/' + file_frame_combined_name + '.bin'

    # TODO: Verify image_idx
    image_obj = {
        'image_idx': img_idx + 9361,
        'image_shape': [img_height, img_width],
        'image_path': 'image/{}/'.format(SPLIT) + file_frame_combined_name + '.jpg'
    }

    np_cam = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['camera_matrix_cam0'][:]), (4,4))
    np_proj_mat = np.reshape(np.asarray(f_obj[frame_id]['camera_matrices']['projection_matrix_cam0'][:]), (4,4))
    
    # TODO: Verify
    pix_T_cam, _, _ = get_intrinsics_from_projection_matrix(np_proj_mat, (img_height, img_width))
    calib = {
        'K': pix_T_cam.astype(np.float32),
        'Rt': np_cam.astype(np.float32)
    }

    pcd_generator = PhysionPointCloudGenerator(hdf5_file_path=os.path.join(PHYSION_HDF5_ROOT, _file), frame_number=frame_id, plot=False)
    pcd_points = pcd_generator.run()
    if pcd_points is None: return
    pcd_points.astype('float32').tofile(os.path.join(STORE_PATH_ROOT, pts_path))
    
    # pcd_new = open3d_convert_rgbd_to_pcd(pix_T_cam, np.array(image), pcd_generator.get_depth_values(f_obj[frame_id]["images"]["_depth_cam0"][:], width=512, height=512, near_plane=0.1, far_plane=100))

    bbox_list = []
    location_list = []
    dimensions_list = []
    gt_boxes_upright_depth_list = []
    heading_ang = []
    names_list = []
    index_list = []
    bbox_points_list = []
    gt_boxes_world_coords = []

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
            warnings.warn('Missing segmentation mask for file: ' + _file + " at frame: " + frame_id) 
            continue
        # import pdb; pdb.set_trace() 
        bbox = get_bbox_from_seg_mask(seg_mask)
        bbox_list.append(bbox)

        location_list.append(f_obj[frame_id]["objects"]["center_cam0"][seg_id])
        center = f_obj[frame_id]["objects"]["center_cam0"][seg_id]
        # convert_camera_to_world(np_cam, np_proj_mat, center)
        
        front = f_obj[frame_id]["objects"]["front_cam0"][seg_id]
        back = f_obj[frame_id]["objects"]["back_cam0"][seg_id]
        width_val = abs(front[2] - back[2])

        left = f_obj[frame_id]["objects"]["left_cam0"][seg_id]
        right = f_obj[frame_id]["objects"]["right_cam0"][seg_id]
        length_val = abs(left[0] - right[0])

        top = f_obj[frame_id]["objects"]["top_cam0"][seg_id]
        bottom = f_obj[frame_id]["objects"]["bottom_cam0"][seg_id]
        height_val = abs(top[1] - bottom[1])

        center_x = f_obj[frame_id]["objects"]["center_cam0"][seg_id][0]
        center_y = f_obj[frame_id]["objects"]["center_cam0"][seg_id][1]
        center_z = f_obj[frame_id]["objects"]["center_cam0"][seg_id][2]
        

        #TODO: Check quartonion order
        [x,y,z, w] = f_obj[frame_id]["objects"]["rotations_cam0"][seg_id]
        t = f_obj[frame_id]["objects"]["positions_cam0"][seg_id]
        scale = s_obj['scale'][seg_id]
        R = get_rotation_matrix_from_quaternion(x,y,z,w)
        ortho6d = get_ortho6d_from_R(R)
        R_from_ortho = compute_rotation_matrix_from_ortho6d_np(ortho6d.reshape(1, 6)).squeeze(0)
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
        bbox_points = [center, front, top, back, bottom, left, right]
 
        if len(gt_boxes_upright_depth) == 0 : import pdb; pdb.set_trace()
        gt_boxes_upright_depth_list.append(gt_boxes_upright_depth)
        bbox_points_list.append(bbox_points)
        names_list.append(obj_name.decode('utf-8'))
        index_list.append(CRUCIAL_OBJECTS_CLASS[obj_name])


    """ 
    center = [0, 0.5, 0]
    front = [0, 0.5, 0.5]
    top = [0, 1, 0]
    back = [0, 0.5, -0.5]
    bottom = [0,0,0]
    left = [-0.5, 0.5, 0]
    right = [0.5, 0.5, 0]
    """
    
    canonical_values = {"center":[0, 0.5, 0],
                        "front":[0, 0.5, 0.5],
                        "top":[0, 1, 0],
                        "back":[0, 0.5, -0.5],
                        "bottom":[0,0,0],
                        "left":[-0.5, 0.5, 0],
                        "right":[0.5, 0.5, 0]}
    
    
    # import pdb; pdb.set_trace()
    # gt_world_coords = convert_to_world_coords(gt_boxes_upright_depth_list)
    # # gt_world_coords = bbox_to_corners(torch.tensor(gt_boxes_upright_depth_list))
    # visualizer = PointCloudVisualizer()
    # visualizer.visualize_point_cloud_and_bboxes(pcd_points, gt_world_coords, use_points=True, show=True)

    
    num_segments_in_img = len(gt_boxes_upright_depth_list)
    annos = {
        'gt_num': num_segments_in_img,
        # 'name': np.asarray(names_list),
        'name': np.asarray(['object' for i in range(num_segments_in_img)]),
        'bbox': np.asarray(bbox_list),
        'location': np.asarray(location_list),
        'dimensions': np.asarray(dimensions_list),
        'rotation_y': np.asarray(heading_ang),
        'index': np.asarray([i for i in range(num_segments_in_img)]),
        'class': np.asarray([0 for _ in range(num_segments_in_img)], dtype=np.int32),
        # 'class': np.asarray(index_list),
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
    
  

if __name__ == "__main__":
    
    args = parse_args()
    SPLIT = args.split
    # PHYSION_HDF5_ROOT = "/home/kalyanav/Downloads/support_all_movies" + f"{SPLIT}"
    PHYSION_HDF5_ROOT = "/media/kalyanav/Venkat/support_data/support_all_movies/"+ f"{SPLIT}"
    PREV_PKL_FILE_PATH = "/media/kalyanav/Venkat/support_data" + f"/{SPLIT}.pkl"
    OBJ_TYPE_LIST_PATH = "//media/kalyanav/Venkat/support_data" + f"/{SPLIT}.txt"
    # PREV_PKL_FILE_PATH = "/media/kalyanav/Venkat/support_data" + ".pkl"
    # OBJ_TYPE_LIST_PATH = "/media/kalyanav/Venkat/support_data" + ".txt"

    # PHYSION_RGB_PATH = "/home/kashis/Desktop/Eval7/tr3d/physion"

    STORE_PATH_ROOT = "/media/kalyanav/Venkat/support_data"
    data_infos = []
    img_idx = 0
    start = 50
    frames_per_vid = 50

    for _file_idx, _file in enumerate(sorted(os.listdir(PHYSION_HDF5_ROOT))):
        if 'pilot_towers_nb4_fr015_SJ000_gr01_mono1_dis0_occ0_boxroom_stable_0274' in _file: #TODO use dominoes only or 'collision' in _file
            # for filename in sorted((os.listdir(os.path.join(PHYSION_HDF5_ROOT, _file)))):
                if os.path.join(PHYSION_HDF5_ROOT, _file).endswith('hdf5'):
                    import pdb; pdb.set_trace()
                    vid_hdf5_path = os.path.join(PHYSION_HDF5_ROOT, _file)
                    print("Looking at : ", os.path.join(_file))
                    try:
                        with h5py.File(vid_hdf5_path, 'r') as file:
                            for frame_id in [key for key in file["frames"].keys()]:
                                
                                phys_dict = get_phys_dict(file, img_idx, _file, _file_idx, frame_id)
                                
                                if phys_dict is None:
                                    print("Broken input ......")
                                    continue
                                print(_file, frame_id)
                                print("img_idx: ",img_idx)
                                img_idx += 1
                                if phys_dict['annos']['gt_boxes_upright_depth'].size != 0:
                                    data_infos.append(phys_dict)
                    except OSError:
                        continue
    print("Storing {} pickle file ....".format(SPLIT) )
    with open(PREV_PKL_FILE_PATH, "wb") as pickle_file:
        pickle.dump(data_infos, pickle_file)
    
    import re

    # Define the regular expression pattern to match HDF5 filenames
    pattern = r'pilot_towers_nb\d+_fr\d+_SJ\d+_mono\d+_dis\d+_occ\d+_.*?\.hdf5'

    # Open the text file and read its contents
    file_path = '/home/kalyanav/Downloads/file_names_missed_out.txt'  # Update with the path to your text file
    with open(file_path, 'r') as file:
        text = file.read()

    # Find all matches in the text
    hdf5_filenames = re.findall(pattern, text)
    
    for _file_idx, _file in enumerate(sorted(hdf5_filenames)):
        if 'tower' in _file: #TODO use dominoes only or 'collision' in _file
            # for filename in sorted((os.listdir(os.path.join(PHYSION_HDF5_ROOT, _file)))):
                if os.path.join(PHYSION_HDF5_ROOT, _file).endswith('hdf5'):
                    vid_hdf5_path = os.path.join(PHYSION_HDF5_ROOT, _file)
                    print("Looking at : ", os.path.join(_file))
                    try:
                        with h5py.File(vid_hdf5_path, 'r') as file:
                            for frame_id in [key for key in file["frames"].keys()][start : start + frames_per_vid]:
                                
                                phys_dict = get_phys_dict(file, img_idx, _file, _file_idx, frame_id)
                                
                                if phys_dict is None:
                                    print("Broken input ......")
                                    continue
                                print(_file, frame_id)
                                print("img_idx: ",img_idx+9361)
                                img_idx += 1
                                if phys_dict['annos']['gt_boxes_upright_depth'].size != 0:
                                    data_infos.append(phys_dict)
                    except OSError:
                        continue
    print("Storing {} pickle file ....".format(SPLIT) )
    with open(PREV_PKL_FILE_PATH, "ab+") as pickle_file:
        pickle.dump(data_infos, pickle_file)
