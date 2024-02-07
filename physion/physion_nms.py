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

def iou_3d(boxes1, boxes2):
    """Non differentiable 3d overlap for calculation of IOU using  
    https://pytorch3d.org/docs/iou3d
 
    Args:
        box1 (tensor): box1 of shape (M,8,3)
        box2 (tensor): box2 of shape (N,8,3)
    """
    intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
    return intersection_vol, iou_3d # M x N shape 


def nms_3d(bboxes, iou_threshold, score_threshold, scores):
    import pdb; pdb.set_trace()
    n_classes = scores.shape[1]
    nms_bboxes, nms_scores, nms_labels = [], [], []
    nms_indices = []
    for i in range(n_classes):
        ids = scores[:, i] > score_threshold
        if not ids.any():
                continue
        
        class_scores = scores[ids, i]
        class_bboxes = bboxes[ids]
        sorted_indices = torch.argsort(class_scores, 0,True)
        sorted_class_bboxes = class_bboxes[sorted_indices]

        while sorted_class_bboxes:
            chosen_box = sorted_class_bboxes[0]
            sorted_class_bboxes = sorted_class_bboxes[1:]

            selected_indices = [
                idx
                for idx, box in enumerate(bboxes)
                if idx in ids and 
                iou_3d(chosen_box, box) < iou_threshold
            ]
            nms_indices.extend(selected_indices)
        nms_bboxes.append(class_bboxes[nms_indices])
        nms_scores.append(class_scores[nms_indices])
        nms_labels.append(
        bboxes.new_full(
                    class_scores[nms_indices].shape, i, dtype=torch.long))