import sys
import unittest
import torch

sys.path.append("ML/Pytorch/object_detection/metrics/")
from YOLO.utils import *


class TestIntersectionOverUnion(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
        self.t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
        self.t1_correct_iou = 1 / 7

        self.t2_box1 = torch.tensor([0.95, 0.6, 0.5, 0.2])
        self.t2_box2 = torch.tensor([0.95, 0.7, 0.3, 0.2])
        self.t2_correct_iou = 3 / 13

        self.t3_box1 = torch.tensor([0.25, 0.15, 0.3, 0.1])
        self.t3_box2 = torch.tensor([0.25, 0.35, 0.3, 0.1])
        self.t3_correct_iou = 0

        self.t4_box1 = torch.tensor([0.7, 0.95, 0.6, 0.1])
        self.t4_box2 = torch.tensor([0.5, 1.15, 0.4, 0.7])
        self.t4_correct_iou = 3 / 31

        self.t5_box1 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t5_box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t5_correct_iou = 1

        # (x1,y1,x2,y2) format
        self.t6_box1 = torch.tensor([2, 2, 6, 6])
        self.t6_box2 = torch.tensor([4, 4, 7, 8])
        self.t6_correct_iou = 4 / 24

        self.t7_box1 = torch.tensor([0, 0, 2, 2])
        self.t7_box2 = torch.tensor([3, 0, 5, 2])
        self.t7_correct_iou = 0

        self.t8_box1 = torch.tensor([0, 0, 2, 2])
        self.t8_box2 = torch.tensor([0, 3, 2, 5])
        self.t8_correct_iou = 0

        self.t9_box1 = torch.tensor([0, 0, 2, 2])
        self.t9_box2 = torch.tensor([2, 0, 5, 2])
        self.t9_correct_iou = 0

        self.t10_box1 = torch.tensor([0, 0, 2, 2])
        self.t10_box2 = torch.tensor([1, 1, 3, 3])
        self.t10_correct_iou = 1 / 7

        self.t11_box1 = torch.tensor([0, 0, 3, 2])
        self.t11_box2 = torch.tensor([1, 1, 3, 3])
        self.t11_correct_iou = 0.25

        self.t12_bboxes1 = torch.tensor(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 3, 2],
            ]
        )
        self.t12_bboxes2 = torch.tensor(
            [
                [3, 0, 5, 2],
                [3, 0, 5, 2],
                [0, 3, 2, 5],
                [2, 0, 5, 2],
                [1, 1, 3, 3],
                [1, 1, 3, 3],
            ]
        )
        self.t12_correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])

        # Accept if the difference in iou is small
        self.epsilon = 0.001

    def test_both_inside_cell_shares_area(self):
        iou = intersection_over_union(self.t1_box1, self.t1_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t1_correct_iou) < self.epsilon))

    def test_partially_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t2_box1, self.t2_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t2_correct_iou) < self.epsilon))

    def test_both_inside_cell_shares_no_area(self):
        iou = intersection_over_union(self.t3_box1, self.t3_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t3_correct_iou) < self.epsilon))

    def test_midpoint_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t4_box1, self.t4_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t4_correct_iou) < self.epsilon))

    def test_both_inside_cell_shares_entire_area(self):
        iou = intersection_over_union(self.t5_box1, self.t5_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t5_correct_iou) < self.epsilon))

    def test_box_format_x1_y1_x2_y2(self):
        iou = intersection_over_union(self.t6_box1, self.t6_box2, box_format="corners")
        self.assertTrue((torch.abs(iou - self.t6_correct_iou) < self.epsilon))

    def test_additional_and_batch(self):
        ious = intersection_over_union(
            self.t12_bboxes1, self.t12_bboxes2, box_format="corners"
        )
        all_true = torch.all(
            torch.abs(self.t12_correct_ious - ious.squeeze(1)) < self.epsilon
        )
        self.assertTrue(all_true)


sys.path.append("ML/Pytorch/object_detection/metrics/")


class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c1_boxes = [[1, 1, 0.5, 0.45, 0.4, 0.5], [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

        self.t2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

        self.t3_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c3_boxes = [[1, 1, 0.5, 0.5, 0.2, 0.4], [2, 0.8, 0.25, 0.35, 0.3, 0.1]]

        self.t4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

    def test_remove_on_iou(self):
        bboxes = non_max_suppression(
            self.t1_boxes, prob_threshold=0.2, iou_threshold=7 / 20, box_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = non_max_suppression(
            self.t2_boxes, prob_threshold=0.2, iou_threshold=7 / 20, box_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = non_max_suppression(
            self.t3_boxes, prob_threshold=0.2, iou_threshold=7 / 20, box_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = non_max_suppression(
            self.t4_boxes, prob_threshold=0.2, iou_threshold=9 / 20, box_format="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_preds = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t1_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t1_correct_mAP = 1

        self.t2_preds = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t2_targets = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t2_correct_mAP = 1

        self.t3_preds = [
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t3_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t3_correct_mAP = 0

        self.t4_preds = [
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

        self.t4_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t4_correct_mAP = 5 / 18

        self.epsilon = 1e-4

    def test_all_correct_one_class(self):
        mean_avg_prec = mean_average_precision(
            self.t1_preds,
            self.t1_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t1_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_correct_batch(self):
        mean_avg_prec = mean_average_precision(
            self.t2_preds,
            self.t2_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t2_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_wrong_class(self):
        mean_avg_prec = mean_average_precision(
            self.t3_preds,
            self.t3_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.t3_correct_mAP - mean_avg_prec) < self.epsilon)

        mean_avg_prec = mean_average_precision(
            self.t3_preds,
            self.t3_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.t3_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_one_inaccurate_box(self):
        mean_avg_prec = mean_average_precision(
            self.t4_preds,
            self.t4_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t4_correct_mAP - mean_avg_prec) < self.epsilon)


if __name__ == "__main__":
    unittest.main()
