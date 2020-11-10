import torch
from torch import nn


# in format "layer type": (details, more_details)
# "Conv": (square kernel size, filter count, stride, padding)
# "Max Pool": (square kernel size, stride)
architecture_config = [
    ["Conv", 7, 64, 2, 3],
    ["Max Pool", 2, 2],
    ["Conv", 3, 192, 1, 1],
    ["Max Pool", 2, 2],
    ["Conv", 1, 128, 1, 0],
    ["Conv", 3, 256, 1, 1],
    ["Conv", 1, 256, 1, 0],
    ["Conv", 3, 512, 1, 1],
    ["Max Pool", 2, 2],
    ["Conv", 1, 256, 1, 0],
    ["Conv", 3, 512, 1, 1],
    ["Conv", 1, 256, 1, 0],
    ["Conv", 3, 512, 1, 1],
    ["Conv", 1, 256, 1, 0],
    ["Conv", 3, 512, 1, 1],
    ["Conv", 1, 256, 1, 0],
    ["Conv", 3, 512, 1, 1],
    ["Conv", 1, 512, 1, 0],
    ["Conv", 3, 1024, 1, 1],
    ["Max Pool", 2, 2],
    ["Conv", 1, 512, 1, 0],
    ["Conv", 3, 1024, 1, 1],
    ["Conv", 1, 512, 1, 0],
    ["Conv", 3, 1024, 1, 1],
    ["Conv", 3, 1024, 1, 1],
    ["Conv", 3, 1024, 2, 1],
    ["Conv", 3, 1024, 1, 1],
    ["Conv", 3, 1024, 1, 1],
]


class CNNLayer(nn.Module):
    # args such as kernel size, stride, padding, etc will be passed in as kwargs
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, architecture=architecture_config, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels

        self.backbone = self._create_backbone(self.architecture)
        self.fc_layers = self._create_fc(**kwargs)

    def forward(self, x):
        # return self.fc_layers(self.flatten(self.backbone(x)))
        x = self.backbone(x)
        return self.fc_layers(x)

    def _create_backbone(self, architecture):
        layers = []
        in_channels = self.in_channels
        self.out_channels_backbone = None

        for layer in architecture:
            if layer[0] == "Conv":
                layers.append(
                    CNNLayer(
                        in_channels,
                        layer[2],
                        kernel_size=layer[1],
                        stride=layer[3],
                        padding=layer[4],
                    )
                )
                in_channels = layer[2]
                self.out_channels_backbone = layer[2]

            elif layer[0] == "Max Pool":
                layers.append(nn.MaxPool2d(kernel_size=layer[1], stride=layer[2]))

        return nn.Sequential(*layers)  # the * unpacks the list into args

    def _create_fc(self, split_size, num_boxes, num_classes):
        # TODO: split_size doesn't work with sizes other than 7 bc of model architecture

        return nn.Sequential(
            nn.Flatten(start_dim=1),  # starts flatten at images rather than batch size
            # 4096 nodes in linear layer in paper
            nn.Linear(self.out_channels_backbone * split_size * split_size, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # 5 because each box has 4 coordinates and 1 probability
            # will be reshaped to split, split, 30 for our data
            nn.Linear(496, split_size * split_size * (num_classes + 5 * num_boxes)),
        )


def test_architecture(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    print(model)
    x = torch.randn((10, 3, 448, 448))
    print(model(x).shape)


if __name__ == "__main__":
    test_architecture()
