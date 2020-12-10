import torch
import numpy as np
import cv2
from collections import Counter


def intersection_over_union(predictions, labels, box_format="midpoint"):
    """
    Calculates the intersection divided by the union of the given data.

    :param predictions: Predicted bounding boxes, with shape (..., 4)
    :type predictions: torch.Tensor
    :param labels: Actual bounding boxes, with shape (..., 4)
    :type labels: torch.Tensor
    :param box_format: "midpoint"|"corners", "midpoint" if boxes are (x,y,w,h) and "corners" if boxes are (x1,y1,x2,y2)
    :type box_format: str
    :return: Intersection over union for all examples.
    :rtype: torch.Tensor
    """

    # predictions.shape = (batches, 4)
    # labels.shape = (batches, 4)

    if box_format == "midpoint":
        pred_x1 = predictions[..., 0:1] - (predictions[..., 2:3] / 2)
        pred_y1 = predictions[..., 1:2] - (predictions[..., 3:4] / 2)
        pred_x2 = predictions[..., 0:1] + (predictions[..., 2:3] / 2)
        pred_y2 = predictions[..., 1:2] + (predictions[..., 3:4] / 2)

        label_x1 = labels[..., 0:1] - (labels[..., 2:3] / 2)
        label_y1 = labels[..., 1:2] - (labels[..., 3:4] / 2)
        label_x2 = labels[..., 0:1] + (labels[..., 2:3] / 2)
        label_y2 = labels[..., 1:2] + (labels[..., 3:4] / 2)

    if box_format == "corners":
        pred_x1 = predictions[..., 0:1]
        pred_y1 = predictions[..., 1:2]
        pred_x2 = predictions[..., 2:3]
        pred_y2 = predictions[..., 3:4]

        label_x1 = labels[..., 0:1]
        label_y1 = labels[..., 1:2]
        label_x2 = labels[..., 2:3]
        label_y2 = labels[..., 3:4]

    if box_format != "midpoint" and box_format != "corners":
        raise ValueError(f"Illegal Argument format={box_format}.")

    x1 = torch.max(pred_x1, label_x1)
    y1 = torch.max(pred_y1, label_y1)

    x2 = torch.min(pred_x2, label_x2)
    y2 = torch.min(pred_y2, label_y2)

    # the clamp keeps values from going negative when there is no overlap
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)

    # union should not double count the intersection. 1e-10 added for numerical stability
    return intersection / (pred_area + label_area - intersection + 1e-10)


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    # bboxes = [[class, prob, x1, y1, x2, y2], [], []], length is number of bboxes
    assert type(bboxes) == list
    assert box_format == "corners" or box_format == "midpoint"

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes_nms = []  # bboxes after non max suppression

    bboxes = sorted(
        bboxes, key=lambda x: x[1], reverse=True
    )  # sorting by prob, highest to lowest

    while bboxes:
        current_box = bboxes.pop(0)
        # keep the box if it is from a different class or its iou is higher than the threshold
        bboxes = [
            box
            for box in bboxes
            if box[0] != current_box[0]
            or intersection_over_union(
                torch.Tensor(current_box[2:]),
                torch.Tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_nms.append(current_box)

    return bboxes_nms


def mean_average_precision(
    pred_boxes, label_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
    # pred_boxes = [[train_ind, class_pred, prob, x1, y1, x2, y2], [], []], len=num bboxes
    average_precisions = []
    epsilon = 1e-10  # for numerical stability

    for cls in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == cls:  # if it is the current class
                detections.append(detection)
        for label in label_boxes:
            if label[1] == cls:
                ground_truths.append(label)

        # counts number of occurrences of each train_ind, effectively counting number of bboxes per image
        num_bboxes = Counter([box[0] for box in ground_truths])

        # this is to help keep track of bboxes we have already looked at
        for ind, num in num_bboxes.items():
            # ind is train_ind, num is num of bboxes for train_ind
            num_bboxes[ind] = torch.zeros(num)
        # num_bboxes = {0:torch.Tensor([0,0,0]), 1:torch.Tensor([0,0]),...}

        # sort by prob highest to lowest
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true = len(ground_truths)

        for detection_ind, detection in enumerate(detections):
            # only grabbing same indexed bboxes, ie, only comparing the predicted and actual
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            num_ground_truths = len(ground_truth_img)
            best_iou = 0
            best_ground_truth_ind = 0

            for ind, ground_truth in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(ground_truth[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_ind = ind

            if (
                best_iou > iou_threshold
                and num_bboxes[detection[0]][best_ground_truth_ind] == 0
            ):  # if this is best bbox and hasn't already been covered
                TP[detection_ind] = 1
                num_bboxes[detection[0]][best_ground_truth_ind] = 1

            else:
                FP[detection_ind] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)  # cumulative sum
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true + epsilon)
        recalls = torch.cat(
            (torch.tensor([0]), recalls)
        )  # added for numerical stability integrating at point 0,1

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))

        average_precisions.append(
            torch.trapz(precisions, recalls)
        )  # for integration, torch.trapz takes (y, x)

    return sum(average_precisions) / len(average_precisions)


def get_bboxes(
    loader,
    model,
    iou_threshold,
    prob_threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    split_size=7,
    num_boxes=2,
    num_classes=20,
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_ind = 0

    for batch_ind, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        # 1 for num_boxes because the labels always only have one bounding box
        true_bboxes = cellboxes_to_boxes(labels, split_size, 1, num_classes)
        bboxes = cellboxes_to_boxes(predictions, split_size, num_boxes, num_classes)

        for batch_ind in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[batch_ind],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_ind] + nms_box)

            for box in true_bboxes[batch_ind]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_ind] + box)

            train_ind += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    img = np.array(image)

    for box in boxes:
        # box: (x_midpoint, y_midpoint, width, height)
        assert len(box) == 4, "Got more values than in x, y, w, h in a box!"
        top_left_x = box[0] - box[2] / 2
        top_left_y = box[1] - box[2] / 2
        bottom_right_x = box[0] + box[2] / 2
        bottom_right_y = box[1] + box[2] / 2

        cv2.rectangle(
            img,
            (top_left_x, top_left_y),
            (bottom_right_x, bottom_right_y),
            color=(255, 0, 0),
        )

    cv2.imshow("Predictions", img)
    cv2.waitKey(0)


def convert_cellboxes(predictions, split_size=7, num_boxes=2, num_classes=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """
    print("num_boxes:", num_boxes)

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(
        batch_size, split_size, split_size, -1
    )  # -1 bc size can be num_classes + 5 * num_boxes for the predictions or num_class + 5 for the labels


    print("predictions.shape:", predictions.shape)


    # bboxes.shape (batches, num_boxes, split_size, split_size, 4)
    bboxes = torch.stack(
        [
            predictions[..., num_classes + 1 + (5 * i) : num_classes + 5 + (5 * i)]
            for i in range(num_boxes)
        ],
        dim=1,
    )  # dim=1 to put the num_boxes into the second dimension rather than the first, which is reserved for batches

    # class predictions, class_scores.shape = (batches, num_boxes, split_size, split_size)
    class_scores = torch.cat(
        [predictions[..., num_classes + 5 * i].unsqueeze(0) for i in range(num_boxes)],
        dim=0,
    )

    # shape = (batches, split_size, split_size, 1)
    best_boxes_ind = class_scores.argmax(0).unsqueeze(-1)
    inverted_best_mask = 1 - best_boxes_ind

    # choosing the best boxes. best_boxes.shape = (batches, num_boxes, split_size, split_size, 4)
    # unsqueezing at dim=1 to make shape align with bboxes, which have the second dim of num_boxes
    best_boxes = bboxes * inverted_best_mask.unsqueeze(1)

    # cell_indices.shape = (batches, 1, split_size, split_size, 1). Is just 0,1,2,3...split_size all the way
    # the final unsqueeze is to make the dims line up with the num_boxes
    cell_indices = (
        torch.arange(split_size).repeat(batch_size, split_size, 1).unsqueeze(-1)
    ).unsqueeze(1)
    # x.shape = (batches, num_boxes, split_size, split_size, 1)
    x = 1 / batch_size * (best_boxes[..., :1] + cell_indices)
    # y.shape = (batches, num_boxes, split_size, split_size, 1)
    y = 1 / batch_size * (best_boxes[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    # w_y.shape = (batches, num_boxes, split_size, split_size, 2)
    w_y = 1 / batch_size * best_boxes[..., 2:4]

    # converted_bboxes.shape = (batches, num_boxes, split_size, split_size, 4)
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    # predicted_class.shape = (batches, 1, split_size, split_size, 1)
    predicted_class = predictions[..., :num_classes].argmax(-1).unsqueeze(1).unsqueeze(-1)
    # predicted_class.shape = (batches, num_boxes, split_size, split_size, 1)
    predicted_class = torch.cat([predicted_class] * num_boxes, dim=1)

    # confidences.shape = (batches, 1, split_size, split_size, 1)
    confidences = torch.stack([predictions[..., num_classes + 5 * i] for i in range(num_boxes)]).permute(1, 2, 3, 0).unsqueeze(1)
    # best_confidence.shape = (batches, 1, split_size, split_size, 1)
    best_confidence = torch.amax(confidences, dim=-1, keepdim=True)  # choosing the highest confidence
    # best_confidence.shape = (batches, num_boxes, split_size, split_size, 1)
    best_confidence = torch.cat([best_confidence] * num_boxes, dim=1)


    print("predicted_class.shape:", predicted_class.shape)
    print("best_confidence.shape:", best_confidence.shape)
    print("converted_bboxes.shape:", converted_bboxes.shape)


    # converted_preds.shape = (batches, num_boxes, split_size, split_size, 6)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, split_size=7, num_boxes=2, num_classes=20):
    converted_pred = convert_cellboxes(out, split_size, num_boxes, num_classes).reshape(
        out.shape[0], split_size * split_size, -1
    )
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(split_size * split_size):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="YOLO_checkpoint.pth"):
    print("<=================>\nSaving checkpoint\n<=================>")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("<=================>\nLoading checkpoint\n<=================>")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
