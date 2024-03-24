# import torch

# def iou(boxes_preds, boxes_labels):
#     """
#     IOU based on corners of the 3D bounding boxes
#     """
#     box1_corners = boxes_preds  # Shape: B x 8 x 3
#     box2_corners = boxes_labels  # Shape: B x 8 x 3

#     box1_min = torch.min(box1_corners, dim=1)[0]
#     box1_max = torch.max(box1_corners, dim=1)[0]
    
#     box2_min = torch.min(box2_corners, dim=1)[0]
#     box2_max = torch.max(box2_corners, dim=1)[0]

#     x1 = box1_min[..., 0]
#     y1 = box1_min[..., 1]
#     z1 = box1_min[..., 2]
#     x2 = box2_min[..., 0]
#     y2 = box2_min[..., 1]
#     z2 = box2_min[..., 2]

#     # Need clamp(0) in case they do not intersect, then we want intersection to be 0
#     intersection = (torch.min(box1_max[..., 0], box2_max[..., 0]) - x1).clamp(0) * \
#                    (torch.min(box1_max[..., 1], box2_max[..., 1]) - y1).clamp(0) * \
#                    (torch.min(box1_max[..., 2], box2_max[..., 2]) - z1).clamp(0)

#     box1_volume = abs((box1_max[..., 0] - x1) * (box1_max[..., 1] - y1) * (box1_max[..., 2] - z1))
#     box2_volume = abs((box2_max[..., 0] - x2) * (box2_max[..., 1] - y2) * (box2_max[..., 2] - z2))

#     return intersection / (box1_volume + box2_volume - intersection + 1e-6)

# def nms_3d(bboxes, iou_threshold, score_threshold, scores):
#     #TODO: need to modify accordingly
#     n_classes = scores.shape[1]
#     nms_bboxes, nms_scores, nms_labels = [], [], []
#     nms_indices = []
#     for i in range(n_classes):
#         ids = scores[:, i] > score_threshold
#         if not ids.any():
#                 continue
        
#         class_scores = scores[ids, i]
#         class_bboxes = bboxes[ids]
#         sorted_indices = torch.argsort(class_scores, 0,True)
#         sorted_class_bboxes = class_bboxes[sorted_indices]

#         while sorted_class_bboxes:
#             chosen_box = sorted_class_bboxes[0]
#             sorted_class_bboxes = sorted_class_bboxes[1:]

#             selected_indices = [
#                 idx
#                 for idx, box in enumerate(bboxes)
#                 if idx in ids and 
#                 iou(chosen_box, box) < iou_threshold
#             ]
#             nms_indices.extend(selected_indices)
#         nms_bboxes.append(class_bboxes[nms_indices])
#         nms_scores.append(class_scores[nms_indices])
#         nms_labels.append(
#         bboxes.new_full(
#                     class_scores[nms_indices].shape, i, dtype=torch.long))

        
#     num_boxes = len(bboxes)
    
#     # Filter out bboxes below the score threshold
#     bboxes = [box for box in bboxes if box[1] > score_threshold]

#     # Sort bboxes based on class scores
#     bboxes = sorted(bboxes, key=scores, reverse=True)
#     bboxes_after_nms = []

#     while bboxes:
#         chosen_box = bboxes.pop(0)

#         # Use torch.tensor for intersection_over_union calculation
#         bboxes = [
#             box
#             for box in bboxes
#             if box[0] != chosen_box[0] or  # Check class prediction
#             (box[1] > score_threshold and  # Check score threshold
#              intersection_over_union(
#                 torch.tensor(chosen_box[2:]),
#                 torch.tensor(box[2:]),
#                 box_format=box_format,
#             ) < iou_threshold)
#         ]

#         bboxes_after_nms.append(chosen_box)

#     # Pad with zeros if the number of selected boxes is less than the original number
#     bboxes_after_nms += [[0] * len(chosen_box) for _ in range(num_boxes - len(bboxes_after_nms))]

#     return bboxes_after_nms


from pytorch3d.ops import box3d_overlap
import torch
import torch.nn.functional as F

def iou_3d(boxes1, boxes2, eps):
    """Non differentiable 3d overlap for calculation of IOU using  
    https://pytorch3d.org/docs/iou3d
 
    Args:
        box1 (tensor): box1 of shape (M,8,3)
        box2 (tensor): box2 of shape (N,8,3)
    """
    intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2, eps)
    return intersection_vol, iou_3d # M x N shape 


_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]
def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:

    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return 


def filter_boxes(boxes,eps):
    # Check coplanar and nonzero area
    coplanar_mask = []
    nonzero_mask = []
    for box in boxes:
        try:
            _check_coplanar(box.unsqueeze(0),eps=eps)
            coplanar_mask.append(True)
        except ValueError:
            coplanar_mask.append(False)
        try:
            _check_nonzero(box.unsqueeze(0),eps=eps)
            nonzero_mask.append(True)
        except:
            nonzero_mask.append(False)
    
    coplanar_mask = torch.tensor(coplanar_mask)
    nonzero_mask = torch.tensor(nonzero_mask)

    return coplanar_mask & nonzero_mask

def filtered_box3d_overlap(boxes1, boxes2, eps=1e-4):
    # Filter boxes1 and boxes2
    mask1 = filter_boxes(boxes1,eps=eps)
    mask2 = filter_boxes(boxes2,eps=eps)
    filtered_boxes1 = boxes1[mask1]
    filtered_boxes2 = boxes2[mask2]

    # Compute overlap for filtered boxes
    vol, iou = box3d_overlap(filtered_boxes1, filtered_boxes2, eps)

    # Map indices back to original boxes
    original_indices1 = torch.arange(len(boxes1))[mask1]
    original_indices2 = torch.arange(len(boxes2))[mask2]

    # Construct output tensors with original shape
    vol_output = torch.zeros((len(boxes1), len(boxes2)), dtype=vol.dtype, device=vol.device)
    iou_output = torch.zeros((len(boxes1), len(boxes2)), dtype=iou.dtype, device=iou.device)
    
    vol_output[original_indices1[:, None], original_indices2[None, :]] = vol
    iou_output[original_indices1[:, None], original_indices2[None, :]] = iou

    return vol_output, iou_output