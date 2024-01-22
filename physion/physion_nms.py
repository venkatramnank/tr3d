import torch

def iou(boxes_preds, boxes_labels):
    """
    IOU based on corners of the 3D bounding boxes
    """
    box1_corners = boxes_preds  # Shape: B x 8 x 3
    box2_corners = boxes_labels  # Shape: B x 8 x 3

    box1_min = torch.min(box1_corners, dim=1)[0]
    box1_max = torch.max(box1_corners, dim=1)[0]
    
    box2_min = torch.min(box2_corners, dim=1)[0]
    box2_max = torch.max(box2_corners, dim=1)[0]

    x1 = box1_min[..., 0]
    y1 = box1_min[..., 1]
    z1 = box1_min[..., 2]
    x2 = box2_min[..., 0]
    y2 = box2_min[..., 1]
    z2 = box2_min[..., 2]

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (torch.min(box1_max[..., 0], box2_max[..., 0]) - x1).clamp(0) * \
                   (torch.min(box1_max[..., 1], box2_max[..., 1]) - y1).clamp(0) * \
                   (torch.min(box1_max[..., 2], box2_max[..., 2]) - z1).clamp(0)

    box1_volume = abs((box1_max[..., 0] - x1) * (box1_max[..., 1] - y1) * (box1_max[..., 2] - z1))
    box2_volume = abs((box2_max[..., 0] - x2) * (box2_max[..., 1] - y2) * (box2_max[..., 2] - z2))

    return intersection / (box1_volume + box2_volume - intersection + 1e-6)

def nms_3d(bboxes, iou_threshold, score_threshold, scores):
    #TODO: need to modify accordingly
    assert type(bboxes) == list

    num_boxes = len(bboxes)
    
    # Filter out bboxes below the score threshold
    bboxes = [box for box in bboxes if box[1] > score_threshold]

    # Sort bboxes based on class scores
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # Use torch.tensor for intersection_over_union calculation
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or  # Check class prediction
            (box[1] > score_threshold and  # Check score threshold
             intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold)
        ]

        bboxes_after_nms.append(chosen_box)

    # Pad with zeros if the number of selected boxes is less than the original number
    bboxes_after_nms += [[0] * len(chosen_box) for _ in range(num_boxes - len(bboxes_after_nms))]

    return bboxes_after_nms