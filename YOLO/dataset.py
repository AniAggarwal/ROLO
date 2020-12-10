from pathlib import Path
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset

# labels are shaped as so: class_num, x_midpoint, y_midpoint, width, height


class VOCDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        label_dir,
        split_size=7,
        num_boxes=2,
        num_classes=20,
        transforms=None,
    ):
        self.annotations = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir).absolute()
        self.label_dir = Path(label_dir).absolute()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.img_dir / self.annotations.iloc[index, 0]
        label_path = self.label_dir / self.annotations.iloc[index, 1]
        boxes = []

        with open(label_path.as_posix()) as f:
            for box in f.readlines():
                features = box.replace("\n", "").split()
                # don't want to make the class values into floats. You have to do int(float(i)) to ensure no errors
                features = [
                    float(i) if float(i) != int(float(i)) else int(i) for i in features
                ]
                boxes += [features]

        image = cv2.imread(img_path.as_posix())
        boxes = torch.Tensor(boxes)

        if self.transforms:
            image, boxes = self.transforms(image, boxes)

        # for every box in S*S grid, label the midpoints and classes
        label_matrix = torch.zeros(
            self.S, self.S, self.C + 5
        )  # labels only have 1 box per cell
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # finding which cell row and column this box is in
            row, col = int(self.S * y), int(self.S * x)
            # now finding the position relative to the cell
            x_cell, y_cell = self.S * x - col, self.S * y - row

            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[row, col, self.C] == 0:  # if no obj in row, col
                label_matrix[row, col, self.C] = 1  # there is an obj in this cell
                label_matrix[row, col, class_label] = 1  # the obj is of this class
                label_matrix[row, col, self.C + 1 :] = torch.Tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )  # and this is the position of the obj

        return image, label_matrix
