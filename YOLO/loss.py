import torch
import torch.nn as nn
from YOLO.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(
        self,
        split_size=7,
        num_boxes=2,
        num_classes=20,
        lambda_coord=5,
        lambda_noobj=0.5,
    ):
        super(YoloLoss, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5 * self.B)
        ious_list = []

        # 0-19 for classes, 20 is prob obj1, 21-24 -> (x,y,w,h) obj1, 25 is prob obj2, 26-30 -> (x,y,w,h) obj2
        for box in range(self.B):
            iou = intersection_over_union(
                    predictions[..., self.C + 1 + 5 * box : self.C + 5 + 5 * box],
                    target[..., self.C + 1 :],
                )
            print("\npred in to iou func:", predictions[..., self.C + 1 + 5 * box : self.C + 5 + 5 * box].shape)
            print("\ntargets in to iou func:", target[..., self.C + 1 :].shape)
            print("\niou.shape:", iou.shape)
            ious_list.append(iou)

        # TODO shape of ious is off
        # ious.shape = (batches, B, S, S, 1)
        ious = torch.stack(ious_list, dim=1)

        ### OLD ###
        # ious.shape = (batches, B)
        # ious = torch.cat([iou.unsqueeze(0) for iou in ious], dim=0)
        # print("\nious.shape:", ious.shape)
        ### OLD ###


        # find the largest iou, the best prediction; returns max, argmax
        iou_max, best_box_ind = torch.max(ious, dim=0)

        # 1 if object exists
        # unsqueeze is to keep last dim after only choosing one element
        exists = target[..., 20:21]  # equivalent to target[..., 20].unsqueeze(3)

        # IMPORTANT SHAPES
        # predictions.shape = (batches, S, S, C + 5 * B)
        # target.shape = (batches, S, S, C + 5)
        # exists.shape = (batches, S, S, 1), where last dim is 0 or 1 depending on existence
        # ious.shape = (batches, B)

        # ====================== #
        #         Losses         #
        # ====================== #

        # Here our goal is to only calculate loss for the responsible box, the one with the highest iou
        coord_loss = 0
        obj_loss = 0
        no_obj_loss = 0

        relevant_boxes_mask = (
            exists * predictions
        )  # will distribute the exists across final dim of predictions
        relevant_target_mask = exists * target

        inverse_boxes_mask = (1 - exists) * predictions
        inverse_target_mask = (1 - exists) * target

        for box in range(self.B):
            # ====================== #
            #     No Object Loss     #
            # ====================== #
            # calculated first as it is done for all boxes

            # calculating difference between predicted and actual probabilities for all objects, if it doesn't exist
            pred_no_obj = inverse_boxes_mask[
                ..., self.C + 5 * box : self.C + 1 + 5 * box
            ]
            target_no_obj = inverse_target_mask[..., self.C : self.C + 1]
            no_obj_loss += self.mse(pred_no_obj, target_no_obj)

            # only calculate for responsible/best box
            print("\nbox:", box)
            print("best_box_ind:", best_box_ind.shape)
            print("best_box_ind last:", best_box_ind[..., 0])
            # TODO broken:
            if box != best_box_ind:
                continue

            # ====================== #
            #  Box Coordinates Loss  #
            # ====================== #

            # pred_coords = (x, y, w, h)
            pred_coords = relevant_boxes_mask[
                ..., self.C + 1 + 5 * box : self.C + 5 + 5 * box
            ]
            target_coords = relevant_target_mask[..., self.C + 1 :]

            # square rooting the width and heights
            # in case the model predicts negatives for width/height, we must take absolute value before the square root
            # additionally, we need to multiply back that sign after so that the gradient is in the correct direction
            # finally, the 1e-10 is added for stability as d/dx(x^1/2) at 0 approaches infinity.
            pred_coords[..., 2:] = torch.sign(pred_coords[..., 2:]) * torch.sqrt(
                torch.abs(pred_coords[..., 2:]) + 1e-10
            )
            # we don't need to worry about the width/height being negative for the targets
            target_coords[..., 2:] = torch.sqrt(target_coords[..., 2:])

            # the mse will find the loss for both center points and widths/heights
            # TODO: I have not flattened the data, that may mess things up. Check back later. Same for other MSE
            coord_loss += self.mse(pred_coords, target_coords)

            # ====================== #
            #       Object Loss      #
            # ====================== #

            # calculating difference between predicted and actual probabilities for the best object, if it exists
            pred_obj = relevant_boxes_mask[..., self.C + 5 * box : self.C + 1 + 5 * box]
            target_obj = relevant_target_mask[..., self.C : self.C + 1]
            obj_loss += self.mse(pred_obj, target_obj)

        # ====================== #
        #       Class Loss       #
        # ====================== #
        class_loss = self.mse(exists * predictions[..., :self.C], exists * target[..., :self.C])

        # ====================== #
        #       Total Loss       #
        # ====================== #

        loss = self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss

        return loss


