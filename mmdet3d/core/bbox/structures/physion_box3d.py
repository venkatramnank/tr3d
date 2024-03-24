# Box 3d based on depth_box3d and base_box3d for Physion dataset to include ortho 6D
import numpy as np
import torch
import warnings
from abc import abstractmethod

from mmdet3d.core.points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis
from physion.external.rotation_continuity.utils import compute_rotation_matrix_from_ortho6d
from physion.physion_tools import canonical_to_world
from physion.physion_nms import iou_3d

class Physion3DBoxes(object):
    def __init__(self, tensor, box_dim = 12, with_ortho6d=True, origin=(0.5, 0.5, 0)):

        # modified_box_dim = box_dim 
        # super().__init__(tensor, box_dim=modified_box_dim, with_yaw=False, origin=origin)
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        self.tensor = tensor.clone()

        # if origin != (0.5, 0.5, 0):
        #     dst = self.tensor.new_tensor((0.5, 0.5, 0))
        #     src = self.tensor.new_tensor(origin)
        #     self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

        self.with_ortho6d = with_ortho6d
        self.rotation_matrix = compute_rotation_matrix_from_ortho6d(tensor[:, 6:])
        self.box_dim = box_dim
        
        
    @property
    def orth6d(self):
        """
        ortho6d representation
        torch.Tensor: Size dimensions of each box in shape (N, 6).
        """
        return self.tensor[:, 6:]

    # @property
    # def yaw(self):
    #     """torch.Tensor: Size dimensions of each box in shape (N, ).""" 
    #     return self.euler_angles[:, 2]
    
    # @property
    # def pitch(self):
    #     return self.euler_angles[:, 1]
    
    # @property
    # def roll(self):
    #     return self.euler_angles[:, 0]
    
    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center
    
    
    
    # @property 
    # def corners(self):
    #     """torch.Tensor: Coordinates of corners of all the boxes
    #     in shape (N, 8, 3).

    #     Convert the boxes to corners in clockwise order, in form of
    #     ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

    #     .. code-block:: none

    #                                        up z
    #                         front y           ^
    #                              /            |
    #                             /             |
    #               (x0, y1, z1) + -----------  + (x1, y1, z1)
    #                           /|            / |
    #                          / |           /  |
    #            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
    #                         |  /      .   |  /
    #                         | / origin    | /
    #            (x0, y0, z0) + ----------- + --------> right x
    #                                       (x1, y0, z0)
    #     """
    #     if self.tensor.numel() == 0:
    #         return torch.empty([0, 8, 3], device=self.tensor.device)
    #     dims = self.tensor[:, 3:6]
        
    #     """corners
    #     the coordinate system is [x,z,y],
    #     with center being [0, 0.5, 0],

        
    #     Then we multiply the scale in the form of [x,z,y] to get the cuboids, Then apply rotation and translation.
    #     """
    #     corners_norm = torch.stack([
    #         torch.Tensor([-0.5, 0, -0.5]),
    #         torch.Tensor([-0.5, 1, -0.5]),
    #         torch.Tensor([-0.5, 0, 0.5]),
    #         torch.Tensor([-0.5, 1, 0.5]),
    #         torch.Tensor([0.5, 0, 0.5]),
    #         torch.Tensor([0.5, 1, 0.5]),
    #         torch.Tensor([0.5, 0, -0.5]),
    #         torch.Tensor([0.5, 1, -0.5])            
    #     ]).to(device=dims.device, dtype=dims.dtype)
    #     # corners_norm = torch.from_numpy(
    #     #     np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
    #     #         device=dims.device, dtype=dims.dtype)

    #     # corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        

    #     # use relative origin (0.5, 0.5, 0.5)
    #     # corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
    #     corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    #     corners = self.rotation_matrix@corners.transpose(1,2)
    #     corners = corners.permute(0,2,1) 
    #     corners += self.bottom_center.view(-1, 1, 3)
    #     return corners
    

    
    @property 
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)
        dims = self.tensor[:, 3:6]
        
        """corners
        the coordinate system is [x,z,y],
        with center being [0, 0.5, 0],

        Then we multiply the scale in the form of [x,z,y] to get the cuboids, Then apply rotation and translation.
        """
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

        # corners_norm = torch.from_numpy(
        #     np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
        #         device=dims.device, dtype=dims.dtype)

        # corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        

        # use relative origin (0.5, 0.5, 0.5)
        # corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        # self.rotation_matrix = torch.diag(torch.tensor([1, 1, -1], dtype=self.rotation_matrix.dtype)) @ self.rotation_matrix
        corners = self.rotation_matrix@corners.transpose(1,2)
        corners = corners.permute(0,2,1) 
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners
    
    
    
    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation
            in XYWHR format, in shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]
  
  
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`DepthInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self, src=Box3DMode.PHYSION, dst=dst, rt_mat=rt_mat)
    

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]


    @property
    def height(self):
        """torch.Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 4]

    @property
    def top_height(self):
        """torch.Tensor:
            A vector with the top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor:
            A vector with bottom's height of each box in shape (N, )."""
        return self.tensor[:, 2]
    
    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        # TODO: Need to build canonical to world
        return self.tensor[:, :3]
    
    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for clearer usage.

        Returns:
            torch.Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center
    

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor


    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'
    

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`PhysionInstance3DBoxes`: A new object of
                :class:`PhysionInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_ortho6d=self.with_ortho6d)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_ortho6d=self.with_ortho6d)


    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device
    

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            box_dim=self.box_dim,
            with_ortho6d=self.with_ortho6d)
    # def rotate(self, angle, points=None):
    #     pass
  


    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim, with_ortho6d=self.with_ortho6d)
        
        
    def copy(self):
        """Create a copy of the Physion3DBoxes object.

        Returns:
            Physion3DBoxes: A copy of the Physion3DBoxes object.
        """
        return Physion3DBoxes(
            tensor=self.tensor.clone(),  # Assuming tensor is a torch.Tensor
            box_dim=self.box_dim,
            with_ortho6d=self.with_ortho6d
        )
        
    @classmethod
    def overlaps(self, boxes1, boxes2, mode='iou'):
        assert isinstance(boxes1, Physion3DBoxes)
        assert isinstance(boxes2, Physion3DBoxes)
        assert type(boxes1) == type(boxes2)
        boxes1_corners = boxes1.corners
        boxes2_corners = boxes2.corners
        desired_order = torch.index_select([[6, 2, 1, 5, 7, 3, 0, 4]]).to(boxes1.device)
        rearranged_boxes1_corners = torch.index_select(boxes1_corners, dim=1, index=desired_order)
        rearranged_boxes2_corners = torch.index_select(boxes2_corners, dim=1, index=desired_order)
        _, iou = iou_3d(rearranged_boxes1_corners, rearranged_boxes2_corners, eps=1e-6)
        
        return iou