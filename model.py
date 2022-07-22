# -*- codeing = utf-8 -*-
# @Time : 2022/7/11 17:55
# @Author : 骆龙飞
# @File : model.py
# @Software: PyCharm

# 搭建神经网络
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),

            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),

            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),

            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    input = torch.ones(64, 3, 32, 32)
    model = MyModel()
    output = model(input)
    print(output, output.shape)
