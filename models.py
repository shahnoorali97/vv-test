import numpy as np
import torch
from collections import OrderedDict


class ModelTask1(torch.nn.Module):
    def __init__(self):
        super(ModelTask1, self).__init__()

        self.layers = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", torch.nn.Conv2d(1, 6, 5)),
                    ("conv_1", torch.nn.Conv2d(6, 9, 5)),
                    ("maxpool_0", torch.nn.MaxPool2d(3, 3)),
                    ("batchnorm_0", torch.nn.BatchNorm2d(9)),
                    ("flatten", torch.nn.Flatten()),
                    ("dense_0", torch.nn.Linear(324, 64)),
                    ("activation_0", torch.nn.ReLU()),
                    ("dense_1", torch.nn.Linear(64, 64)),
                    ("activation_1", torch.nn.ReLU()),
                    ("dense_2", torch.nn.Linear(64, 64)),
                    ("activation_2", torch.nn.ReLU()),
                    ("output", torch.nn.Linear(64, 10)),
                    ("softmax", torch.nn.Softmax(dim=1)),
                ]
            )
            # OrderedDict(
            #     [
            #         ("conv_0", torch.nn.Conv2d(1, 3, 5)),
            #         ("maxpool_0", torch.nn.MaxPool2d(2, 2)),
            #         ('batchnorm_0', torch.nn.BatchNorm2d(3)),
            #         ('flatten', torch.nn.Flatten()),
            #         ('dense_0', torch.nn.Linear(432, 256)),
            #         ('activation_0', torch.nn.Tanh()),
            #         ('output', torch.nn.Linear(256, 10)),
            #         ('softmax', torch.nn.Softmax(dim=1))
            #     ]
            # )
        )

    def forward(self, x):
        return self.layers(x)


class ModelTask3(torch.nn.Module):
    def __init__(self):
        super(ModelTask3, self).__init__()

        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", torch.nn.Conv2d(1, 9, 5)),
                    ("conv_1", torch.nn.Conv2d(9, 6, 5)),
                    ("avgpool_0", torch.nn.AvgPool2d(5, 4)),
                    ("dropout_0", torch.nn.Dropout(p=0.2)),
                    ("flatten", torch.nn.Flatten()),
                ]
            )
        )

        self.linear = torch.nn.Sequential(
            OrderedDict(
                [
                    ("linear_0", torch.nn.Linear(96, 256)),
                    ("layernorm_1", torch.nn.LayerNorm(256)),
                    ("activation_0", torch.nn.ReLU()),
                    ("linear_1", torch.nn.Linear(256, 128)),
                    ("layernorm_2", torch.nn.LayerNorm(128)),
                    ("activation_1", torch.nn.ReLU()),
                    ("linear_2", torch.nn.Linear(128, 64)),
                    ("layernorm_3", torch.nn.LayerNorm(64)),
                    ("activation_2", torch.nn.Tanh()),
                ]
            )
        )

        self.output = torch.nn.Sequential(
            OrderedDict(
                [
                    ("output", torch.nn.Linear(64, 10)),
                    ("softmax", torch.nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)

        return self.output(x)


class ModelTask4(torch.nn.Module):
    def __init__(self):
        super(ModelTask4, self).__init__()

        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", torch.nn.Conv2d(1, 9, 5)),
                    ("conv_1", torch.nn.Conv2d(9, 6, 5)),
                    ("avgpool_0", torch.nn.AvgPool2d(5, 4)),
                    ("dropout_0", torch.nn.Dropout(p=0.2)),
                    ("flatten", torch.nn.Flatten()),
                ]
            )
        )

        self.linear = torch.nn.Sequential(
            OrderedDict(
                [
                    ("linear_0", torch.nn.Linear(96, 256)),
                    ("layernorm_1", torch.nn.LayerNorm(256)),
                    ("activation_0", torch.nn.ReLU()),
                    ("linear_1", torch.nn.Linear(256, 128)),
                    ("layernorm_2", torch.nn.LayerNorm(128)),
                    ("activation_1", torch.nn.ReLU()),
                    ("linear_2", torch.nn.Linear(128, 64)),
                    ("layernorm_3", torch.nn.LayerNorm(64)),
                    ("activation_2", torch.nn.ReLU()),
                ]
            )
        )

        self.output = torch.nn.Sequential(
            OrderedDict(
                [
                    ("output", torch.nn.Linear(64, 10)),
                ]
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)

        return self.output(x)
