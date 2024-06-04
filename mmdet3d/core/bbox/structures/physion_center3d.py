# Box 3d based on depth_box3d and base_box3d for Physion dataset  with center only
import numpy as np
import torch
import warnings
from abc import abstractmethod

from mmdet3d.core.points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis
from physion.external.rotation_continuity.utils import compute_rotation_matrix_from_ortho6d
from physion.physion_tools import canonical_to_world
from physion.physion_nms import *

class Physion3DCenter(object):
    def __init__(self, tensor, box_dim = 3):

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
        self.box_dim = box_dim


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
            box=self, src=Box3DMode.PHYSIONCENTER, dst=dst, rt_mat=rt_mat)
    
    
    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with world center of each box in shape (N, 3)."""
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
                box_dim=self.box_dim)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim)


    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device
    

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            box_dim=self.box_dim)
  


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
            new_tensor, box_dim=self.box_dim)
        
        
    def copy(self):
        """Create a copy of the Physion3DCenter object.

        Returns:
            Physion3DCenter: A copy of the Physion3DCenter object.
        """
        return Physion3DCenter(
            tensor=self.tensor.clone(),  # Assuming tensor is a torch.Tensor
            box_dim=self.box_dim
        )
    