import torch

def nms(boxes, iou_threshold):
    
    # boxes = sorted(boxes, key=lambda x :x[-1], reverse=True)
    indexes = boxes[:,-1].argsort(descending = True)
    new_boxes = boxes[indexes]
    print(new_boxes)
    
    keep_boxes = []

    while(len(new_boxes) > 1):
        _box = new_boxes[0]
        keep_boxes.append(_box)
        _boxes = new_boxes[1:]

        new_boxes = _boxes[torch.where(iou(_box, _boxes) < iou_threshold)]

    if len(new_boxes) == 1:
        keep_boxes.append(new_boxes[0])

    return torch.vstack(keep_boxes)

def iou(box, boxes):

    box_area = abs(box[0] - box[2]) * abs(box[1] - box[3])
    boxes_area = abs(boxes[:, 0] - boxes[:, 2]) * abs(boxes[:, 1] - boxes[:, 3])

    left_x = torch.maximum(box[0], boxes[:, 0])
    left_y = torch.maximum(box[1], boxes[:, 1])
    right_x = torch.minimum(box[2], boxes[:, 2])
    right_y = torch.minimum(box[3], boxes[:, 3])

    inter_area = torch.maximum(right_x - left_x, torch.Tensor([0])) * torch.maximum(right_y - left_y, torch.Tensor([0]))
    union_area = box_area + boxes_area - inter_area

    return inter_area/union_area

if __name__ == "__main__":
    box = torch.tensor([0,0,4,4])
    boxes = torch.tensor([[0,0,4,4, 0.1], [1,1,5,5, 0.8], [5,5,10,10, 0.23]])
    print(iou(box, boxes))
    print("*"*20)

    boxes = torch.tensor([[0,0,4,4, 0.9], [0,1,5,5, 0.8], [5,5,10,10, 0.23], [4,5,11,21, 0.54], [10,11,34,33, 0.78]])
    print(nms(boxes, 0.35))