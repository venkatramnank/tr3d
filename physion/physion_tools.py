# Tools for Physion data

import PIL.Image as Image
from PIL import ImageOps
import numpy as np
import math
import io
import h5py
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from physion.external.rotation_continuity.utils import get_ortho6d_from_R, compute_rotation_matrix_from_ortho6d_np, compute_rotation_matrix_from_ortho6d
import torch



class PhysionPointCloudGenerator:
    """Point cloud generator for a single frame 
    """

    def __init__(self, hdf5_file_path, frame_number, plot=False):
        self.hdf5_file_path = hdf5_file_path
        if not isinstance(frame_number, str):
            self.frame_number = '{:04d}'.format(frame_number)
        else:
            self.frame_number = frame_number
        self.hf = h5py.File(self.hdf5_file_path, 'r')

        self.object_ids = self.hf["static"]["object_ids"][:].tolist()
        self.object_names = self.hf["static"]["model_names"][:]
        self.scales = self.hf["static"]["scale"][:]
        self.colors = self.hf['static']['color'][:]
        self.segmentation_colors = self.hf['static']['object_segmentation_colors'][:]
        self.projection_matrix = np.array(
            self.hf['frames'][self.frame_number]['camera_matrices']['projection_matrix']).reshape(4, 4)
        self.camera_matrix = np.array(
            self.hf['frames'][self.frame_number]['camera_matrices']['camera_matrix']).reshape(4, 4)

        self.img_array = self.io2image(
            self.hf["frames"][self.frame_number]["images"]["_img"][:])
        self.img_height = self.img_array.shape[0]
        self.img_width = self.img_array.shape[1]
        self.seg_array = self.io2image(
            self.hf["frames"][self.frame_number]["images"]["_id"][:])
        #NOTE: Changed the dep_array ==> divide by 10
        self.dep_array = self.get_depth_values(
            self.hf["frames"][self.frame_number]["images"]["_depth"][:], width=self.img_width, height=self.img_height, near_plane=0.1, far_plane=100, depth_pass='_depth')
        self.dep_array = np.where(self.dep_array > 80, 0, self.dep_array) #NOTE: Removal of far away depth
        self.positions = self.hf["frames"][self.frame_number]["objects"]["positions"][:]
        self.rotations = self.hf["frames"][self.frame_number]["objects"]["rotations"][:]
        self.plot = plot
        
    def io2image(self, tmp):
        """Converts bytes format to numpy array of image

        Args:
            tmp (numpy.ndarray): Bytes array of image

        Returns:
            numpy.ndarray: H x W x 3 image
        """
        image = Image.open(io.BytesIO(tmp))
        image = ImageOps.mirror(image)

        image_array = np.array(image)
        return image_array

    def get_depth_values(self, image: np.array, depth_pass: str = "_depth", width: int = 256, height: int = 256, near_plane: float = 0.1, far_plane: float = 100) -> np.array:
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
        # image = np.reshape(image, (height, width, 3))
        
        # Convert the image to a 2D image array.
        if depth_pass == "_depth":

            depth_values = np.array(
                (image[:, :, 0] + image[:, :, 1] / 256.0 + image[:, :, 2] / (256.0 ** 2)))
        elif depth_pass == "_depth_simple":
            depth_values = image[:, :, 0] / 256.0
        else:
            raise Exception(f"Invalid depth pass: {depth_pass}")
        # Un-normalize the depth values.
        return (depth_values * ((far_plane - near_plane) / 256.0)).astype(np.float32)

    def get_intrinsics_from_projection_matrix(self, proj_matrix, size=(256, 256)):
        """Gets intrisic matrices

        Args:
            proj_matrix (np.array): Projection matrix
            size (tuple, optional): Size of image. Defaults to (256, 256).

        Returns:
            numpy.ndarray, float, int: pixel to camera projection, focal length, sensor width
        """
        H, W = size
        vfov = 2.0 * math.atan(1.0/proj_matrix[1][1]) * 180.0 / np.pi
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

    def depth_to_z(self, z, focal_length, sensor_width):
        """calculates and returns the corresponding 3D coordinates in the camera space.

        Args:
            z (numpy.ndarray):  A 3D array representing depth values
            focal_length (float): focal length
            sensor_width (int): sensor width

        Returns:
            numpy.ndarray: 3D coordinates
        """

        z = np.array(z)
        assert z.ndim >= 3
        h, w, _ = z.shape[-3:]
        pixel_centers_x = (
            np.arange(-w/2, w/2, dtype=np.float32) + 0.5) / w * sensor_width
        pixel_centers_y = (
            np.arange(-h/2, h/2, dtype=np.float32) + 0.5) / h * sensor_width
        squared_distance_from_center = np.sum(np.square(np.meshgrid(
            pixel_centers_x,  # X-Axis (columns)
            pixel_centers_y,  # Y-Axis (rows)
            indexing="xy",
        )), axis=0)

        depth_scaling = np.sqrt(
            1 + squared_distance_from_center / focal_length**2)
        depth_scaling = depth_scaling.reshape(
            (1,) * (z.ndim - 3) + depth_scaling.shape + (1,))

        return z / depth_scaling

    def meshgrid2d_py(self, Y, X):
        grid_y = np.linspace(0.0, Y-1, Y)
        grid_y = np.reshape(grid_y, [Y, 1])
        grid_y = np.tile(grid_y, [1, X])

        grid_x = np.linspace(0.0, X-1, X)
        grid_x = np.reshape(grid_x, [1, X])
        grid_x = np.tile(grid_x, [Y, 1])

        return grid_y, grid_x

    def extract_rgbd_from_physion_frame(self, frame_images):
        frame_image = np.array(frame_images.get('_img'))
        rgb_frame_array = self.io2image(frame_image)
        return rgb_frame_array

    def extract_depth_from_physion_frame(self, frame_images):
        frame_depth_array = np.array(frame_images.get('_depth'))
        return frame_depth_array

    # def convert_2D_to_3D(self, obj_2D, camera_matrix, projection_matrix, target_resolution=(256, 256)):
    #     """
    #     Convert 2D coordinates to 3D coordinates using camera and projection matrices.

    #     Args:
    #         obj_2D (numpy.ndarray): Array of 2D coordinates with shape (num_points, 3),
    #                                 where each row represents (x, y, depth).
    #         camera_matrix (numpy.ndarray): Camera matrix for the 3D-to-2D projection.
    #         projection_matrix (numpy.ndarray): Projection matrix for transforming normalized device coordinates.
    #         target_resolution (tuple, optional): Target resolution of the 2D coordinates. Defaults to (256, 256).

    #     Returns:
    #         numpy.ndarray: Array of transformed 3D coordinates with shape (num_points, 3).
    #     """
    #     obj_num = obj_2D.shape[0]
    #     obj_2D = np.concatenate([obj_2D[:, 1:2],
    #                             obj_2D[:, 0:1],
    #                             obj_2D[:, 2:3],
    #                              ], axis=1).astype(np.float32)

    #     obj_2D[:, 1] = 1 - obj_2D[:, 1]/target_resolution[1]
    #     obj_2D[:, 0] = obj_2D[:, 0]/target_resolution[0]
    #     obj_2D[:, :2] = obj_2D[:, :2] * 2 - 1

    #     obj_3D = np.concatenate([obj_2D[:, :2] * obj_2D[:, 2:3],
    #                              obj_2D[:, 2:3],
    #                             (obj_2D[:, 2:3] - 1.0 *
    #                              projection_matrix[2, 3])
    #                             / projection_matrix[2, 2] * projection_matrix[3, 2]],
    #                             axis=1)

    #     obj_3D = np.linalg.inv(projection_matrix) @ obj_3D.T #image coodinate normalization

    #     obj_3D = (np.linalg.inv(camera_matrix) @ obj_3D).T
    #     return obj_3D[:, :3]
    
    def convert_2D_to_3D(self, obj_2D, camera_matrix, projection_matrix, target_resolution=(256, 256)):
        obj_num = obj_2D.shape[0]    
        # obj_2D[:, 1] = 1 - obj_2D[:, 1]/target_resolution[1]
        # obj_2D[:, 0] = obj_2D[:, 0]/target_resolution[0]
        obj_2D = np.concatenate([obj_2D[:, 1:2],
                                obj_2D[:, 0:1],
                                obj_2D[:, 2:3],
                                ], axis=1).astype(np.float32)
        
        obj_2D[:, 1] = 1 - obj_2D[:, 1]/target_resolution[1]
        # obj_2D[:, 1] = obj_2D[:, 1]/target_resolution[1]
        obj_2D[:, 0] = 1-  obj_2D[:, 0]/target_resolution[0]

        obj_2D[:, :2] = obj_2D[:, :2] * 2 - 1
        # print(obj_2D)
        obj_3D =  np.concatenate([obj_2D[:, :2] * obj_2D[:, 2:3],
                                obj_2D[:, 2:3]], 
                                axis=1)
        
        projection_matrix_new=np.concatenate([projection_matrix[:2,:3],
                                                projection_matrix[3:4,:3],
                                                ],axis=0)
        # print(projection_matrix_new)
        # print(np.linalg.inv(projection_matrix_new))
        obj_3D = np.linalg.inv(projection_matrix_new) @ obj_3D.T

        ones = np.ones((1,obj_num))
        obj_3D = np.concatenate((obj_3D, ones), axis=0)   
        obj_3D = (np.linalg.inv(camera_matrix) @ obj_3D ).T
        return obj_3D[:, :3]

    def background_pc(self, size, ind_i_all, ind_j_all, true_z_f, rgb_f, camera_matrix, projection_matrix):
        """
        Generate a point cloud representing the background of a scene based on input parameters.

        Args:
            size (int): Size of the grid (assumes a square grid).
            ind_i_all (list): List of 1D arrays containing row indices for each grid cell.
            ind_j_all (list): List of 1D arrays containing column indices for each grid cell.
            true_z_f (numpy.ndarray): 2D array representing the true depth values for each pixel.
            rgb_f (numpy.ndarray): 3D array representing the RGB values for each pixel.
            camera_matrix (numpy.ndarray): Camera matrix for the transformation from image to camera coordinates.
            projection_matrix (numpy.ndarray): Projection matrix for the transformation from camera to world coordinates.

        Returns:
            tuple: A tuple containing two elements:
                - background_depth_point_world (numpy.ndarray): 2D array representing the 3D coordinates of background points.
                - background_rgb_value (numpy.ndarray): 2D array representing the RGB values of background points.
        """
        def calArray2dDiff(array_0, array_1):
            array_0_rows = array_0.view(
                [('', array_0.dtype)] * array_0.shape[1])
            array_1_rows = array_1.view(
                [('', array_1.dtype)] * array_1.shape[1])

            return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])

        ind_i_all = np.concatenate(ind_i_all, axis=0)
        ind_j_all = np.concatenate(ind_j_all, axis=0)
        ind_o_all = np.concatenate(
            (ind_i_all[:, np.newaxis], ind_j_all[:, np.newaxis]), axis=1)

        i_all = np.concatenate(
            [np.ones(size).astype(int) * i for i in range(size)])
        j_all = np.concatenate([np.arange(size).astype(int)
                               for i in range(size)])
        ind_all = np.concatenate(
            (i_all[:, np.newaxis], j_all[:, np.newaxis]), axis=1)

        ind_b_all = calArray2dDiff(ind_all, ind_o_all)

        background_z_value = true_z_f[ind_b_all[:, 0], ind_b_all[:, 1]]
        background_rgb_value = rgb_f[ind_b_all[:, 0], ind_b_all[:, 1], :]
        background_depth_point_img = np.concatenate(
            [ind_b_all[:, 0][:, np.newaxis], ind_b_all[:, 1][:, np.newaxis], background_z_value[:, np.newaxis]], 1)
        background_depth_point_world = self.convert_2D_to_3D(
            background_depth_point_img, camera_matrix, projection_matrix, target_resolution=(256, 256))

        return background_depth_point_world, background_rgb_value

    
    def pcd_visualizer_with_color(self, pcd_array, color_array):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array)
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        o3d.visualization.draw_geometries([pcd])
    
    def run(self):
        """Builds point cloud from depth images and rgb images

        Returns:
            np.array: PCD [points x 6]; xyzrgb
        """
        
        obj_depth_point_world_f = []
        obj_partial_rgb_f = []
        ind_i_all, ind_j_all = [], []

        for idx, obj_id in enumerate(self.object_ids):
            # obj_name = self.hf['static']['model_names'][:][idx]
            # if obj_name in self.hf['static']['distractors'][:]:
            #     continue
            # if obj_name in self.hf['static']['occluders'][:]:
            #     continue

            selected_mask = np.logical_and.reduce(
                (
                    self.seg_array[:, :,
                                   0] == self.segmentation_colors[idx, 0],
                    self.seg_array[:, :,
                                   1] == self.segmentation_colors[idx, 1],
                    self.seg_array[:, :,
                                   2] == self.segmentation_colors[idx, 2],
                )
            )

            if np.sum(selected_mask) == 0:
                continue

            ind_i, ind_j = np.nonzero(selected_mask)
            ind_i_all.append(ind_i)
            ind_j_all.append(ind_j)
            z_value = self.dep_array[ind_i, ind_j]
            obj_depth_point_img = np.concatenate(
                [ind_i[:, np.newaxis], ind_j[:, np.newaxis], z_value[:, np.newaxis]], 1).astype(np.float32)
            obj_rgb_value = self.img_array[ind_i, ind_j, :]

            obj_depth_point_world = self.convert_2D_to_3D(obj_depth_point_img, self.camera_matrix, self.projection_matrix, target_resolution=(
                self.img_array.shape[0], self.img_array.shape[1]))

            obj_depth_point_world_f.append(obj_depth_point_world)
            obj_partial_rgb_f.append(obj_rgb_value)

        obj_depth_point_world_f = np.concatenate(
            obj_depth_point_world_f, axis=0)
        obj_partial_rgb_f = np.concatenate(obj_partial_rgb_f, axis=0)

        background_depth_point_world, background_rgb_value = self.background_pc(self.img_array.shape[0],
                                                                                ind_i_all,
                                                                                ind_j_all,
                                                                                self.dep_array,
                                                                                self.img_array,
                                                                                self.camera_matrix,
                                                                                self.projection_matrix)
        complete_pcd_world = np.concatenate(
            [obj_depth_point_world_f, background_depth_point_world], axis=0)
        complete_pcd_colors = np.concatenate(
            [obj_partial_rgb_f, background_rgb_value], axis=0)
        complete_pcd_colors = complete_pcd_colors/255.
        complete_pcd = np.concatenate(
            [complete_pcd_world, complete_pcd_colors], axis=1)

        if self.plot:
            self.pcd_visualizer_with_color(complete_pcd_world, complete_pcd_colors)

        return complete_pcd


class PointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def create_point_cloud(self, points, color, corners=None, center = None):
        if corners is not None:
            print("corners included")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(color)
            bbox_colors = np.array([[1.0, 0.0, 0.0]] * corners.shape[0])
            corner_points = o3d.utility.Vector3dVector(corners)
            corner_colors = o3d.utility.Vector3dVector(bbox_colors)
            pcd.points.extend(corner_points)
            pcd.colors.extend(corner_colors)
        elif corners is not None and center is not None:
            print("corners included")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(color)
            bbox_colors = np.array([[1.0, 0.0, 0.0]] * corners.shape[0])
            center_color = np.array([[1.0, 0.0, 0.0]]* center.shape[0])
            corner_points = o3d.utility.Vector3dVector(corners)
            corner_colors = o3d.utility.Vector3dVector(bbox_colors)
            print("center included")
            pcd.points.extend(corner_points)
            pcd.colors.extend(corner_colors)
            pcd.points.extend(o3d.utility.Vector3dVector(center))
            pcd.colors.extend(o3d.utility.Vector3dVector(center_color))
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def create_3d_bbox(self, center=None, dimensions=None, rotation_matrix=None, bbox_points=None, use_rot=False, use_points=False):
        if use_rot:
            bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation_matrix, extent=dimensions)
            bbox.color = (0,1,0)
            # bbox.translate(translation)
        
        elif use_points:
            bbox_points = np.array(bbox_points).reshape(7,3)
            bbox_points_vector = o3d.utility.Vector3dVector(bbox_points)
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(bbox_points_vector)
            bbox.color = (0,1,0)

        else:
            bbox = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3, 3), extent=dimensions)
        return bbox

    def visualize_point_cloud_and_bboxes(self, points, gt_bboxes_list, center = None, corners=None, bbox_points_list=None, use_points=False):
        
        pcd = self.create_point_cloud(points[:, :3], color = points[:, 3:], corners = corners, center = center)
        self.vis.add_geometry(pcd)

        for gt_bbox_info in (gt_bboxes_list):
            # if bbox_points:
            #     bbox_colors = np.array([[0.0, 1.0, 0.0]] * 7)
            #     pcd.points.extend(o3d.utility.Vector3dVector(bbox_points))
            #     pcd.colors.extend(o3d.utility.Vector3dVector(bbox_colors))
            if len(gt_bbox_info) == 10:
                center, dimensions, quartenion = (
                    gt_bbox_info[:3],
                    gt_bbox_info[3:6],
                    gt_bbox_info[6:10]
                )
                dimensions = (np.array(dimensions).transpose(1,2,0)).tolist()
                rot = R.from_quat([quartenion[0], quartenion[1], quartenion[2], quartenion[3]])
                bbox = self.create_3d_bbox(
                    np.array(center).reshape(3, 1),
                    np.array(dimensions).reshape(3, 1),
                    rot.as_matrix(),
                    use_rot=True,
                    use_points=False,
                )
            #using the ortho6D representation
            elif len(gt_bbox_info) == 12:
                center, dimensions, ortho6d = (
                    gt_bbox_info[:3],
                    gt_bbox_info[3:6],
                    gt_bbox_info[6:]
                )
                # dimensions = (np.array(dimensions)[[1,2,0]]).tolist()
                dimensions = (np.array(dimensions)).tolist()
                rot = compute_rotation_matrix_from_ortho6d_np(np.array(ortho6d).reshape(1, 6)).squeeze(0)
                bbox = self.create_3d_bbox(
                    np.array(center).reshape(3, 1),
                    np.array(dimensions).reshape(3, 1),
                    rotation_matrix=rot,
                    use_rot=True,
                    use_points=False,
                    
                ) 
            elif use_points:
                bbox = self.create_3d_bbox(bbox_points=gt_bbox_info, use_points=True, use_rot=False)                  
            else:
                print("Error in dimension of input, unable to visualize")
                exit()
            self.vis.add_geometry(bbox)

        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_up([0, -1, 0])
        self.vis.get_view_control().set_lookat([1, 1, 1])

        self.vis.run()
        self.vis.destroy_window()


#TODO: need to write properties for canonical space to world coordinates and vice versa in terms of utils
#TODO: Need to store the canonical values correctly and since it is same across, it can be passed to the corners calculation correctly (directly)


def canonical_to_world_np(points, R, trans, scale):
    return R @ (np.diag(scale) @ points) + trans
    
def world_to_canonical_np(points, R, trans, scale):
    return np.linalg.inv(np.diag(scale))@(np.linalg.inv(R)@(points - trans))

def world_to_canonical(points, R, trans, scale):
        return torch.matmul(torch.inverse(torch.diag_embed(scale)), torch.matmul(torch.inverse(R), (points - trans)))

def canonical_to_world(points, R, trans, scale):
        return torch.matmul(R, torch.matmul(torch.diag_embed(scale), points.unsqueeze(2))) + trans.unsqueeze(2)



# class cameraUtilsTensor():
#     def __init__(self, rot, trans, scale):
#         self.R = get_ortho6d_from_R(rot)
#         self.trans = trans
#         self.scale = torch.diag(scale)
        
#     def world_to_canonical(self, points):
#         return torch.matmul(torch.inverse(self.scale), torch.matmul(torch.inverse(R), (points - self.trans)))
        
#     def canonical_to_world(self, points):
#         return torch.matmul(self.R, torch.matmul(self.scale, points)) + self.trans
        
        
def convert_to_world_coords(gt_boxes_upright_depth_list):
    canonical_values = {"center":[0, 0.5, 0],
                    "front":[0, 0.5, 0.5],
                    "top":[0, 1, 0],
                    "back":[0, 0.5, -0.5],
                    "bottom":[0,0,0],
                    "left":[-0.5, 0.5, 0],
                    "right":[0.5, 0.5, 0]}
    gt_world_coords = []
    for objects_info in gt_boxes_upright_depth_list:
        points_world_coord = []
        if not isinstance(objects_info, np.ndarray):
            objects_info = np.array(objects_info)
        for point, val in canonical_values.items():
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            points_world_coord.append(canonical_to_world_np(points=val,
                                                             R=compute_rotation_matrix_from_ortho6d_np(objects_info[6:].reshape(1, 6)),
                                                             trans=objects_info[:3],
                                                             scale=objects_info[3:6]))
        gt_world_coords.append(points_world_coord)
    return gt_world_coords
    

def convert_to_world_coords_torch(gt_boxes_upright_depth_list):
    canonical_values = {"center":[0, 0.5, 0],
                    "front":[0, 0.5, 0.5],
                    "top":[0, 1, 0],
                    "back":[0, 0.5, -0.5],
                    "bottom":[0,0,0],
                    "left":[-0.5, 0.5, 0],
                    "right":[0.5, 0.5, 0]}
    gt_world_coords = []
    
    # Convert canonical values to torch tensors
    canonical_values_torch = {key: torch.tensor(value, dtype=torch.float32) for key, value in canonical_values.items()}

    for objects_info in gt_boxes_upright_depth_list:
        points_world_coord = []
        
        # Convert objects_info to numpy array if it's not already
        if not isinstance(objects_info, np.ndarray):
            objects_info = np.array(objects_info)
        
        # Convert objects_info to torch tensor
        objects_info_torch = torch.tensor(objects_info, dtype=torch.float32)
        
        for point, val in canonical_values_torch.items():
            points_world_coord.append(canonical_to_world(points=val,
                                                         R=compute_rotation_matrix_from_ortho6d(objects_info_torch[6:].reshape(1, 6)),
                                                         trans=objects_info_torch[:3],
                                                         scale=objects_info_torch[3:6]))
        
        gt_world_coords.append(points_world_coord)
    
    return gt_world_coords
    