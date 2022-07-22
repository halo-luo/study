# -*- codeing = utf-8 -*-
# @Time : 2022/7/11 17:35
# @Author : 骆龙飞
# @File : train.py
# @Software: PyCharm

import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MyModel

train_data = torchvision.datasets.CIFAR10("./root", train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True)
test_data = torchvision.datasets.CIFAR10("./root", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为 ： {}".format(train_data_size))
print(f"测试数据集的长度为 ： {test_data_size}")

train_loader = DataLoader(train_data, batch_size=64, drop_last=True)
test_loader = DataLoader(test_data, batch_size=64, drop_last=True)

writer = SummaryWriter("log_train")

model = MyModel()
# model = model.cuda()
# print(model)

# 损失函数
loss_cel = nn.CrossEntropyLoss()

# 优化器
op = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 要训练的轮数
epoch = 10

for i in range(10):
    print('-' * 10, "第 {} 轮训练开始 ".format(i + 1), '-' * 10)

    model.train()
    for data in train_loader:
        imgs, tags = data
        outputs = model(imgs)
        loss_result = loss_cel(outputs, tags)
        op.zero_grad()
        loss_result.backward()
        op.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("第 {} 轮训练，loss 值为 ：{}".format(total_train_step, loss_result.item()))
            writer.add_scalar("train_loss", loss_result.item(), total_train_step)

    # 测试
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_test_accuracy = 0
        for data in test_loader:
            imgs, tags = data
            outputs = model(imgs)
            loss = loss_cel(outputs, tags)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == tags).sum()
            total_test_accuracy += accuracy

        total_test_step += 1
        print("测试集上的 loss 为：{}".format(total_test_loss))
        writer.add_scalar("test_total_loss", total_test_loss, total_test_step)
        print("测试集上的正确率 accuracy 为: {}".format(total_test_accuracy))
        writer.add_scalar("test_accuracy", total_test_accuracy, total_test_step)

    torch.save(model, "model_{}.pth".format(i + 1))
    # torch.save(model.state_dict(),"model_dict_{}.pth".format(i+1))

writer.close()

'''
PS C:\Python\python file\.git\objects> git reset --hard HEAD
fatal: this operation must be run in a work tree
PS C:\Python\python file\.git\objects> cd ../
PS C:\Python\python file\.git> git reset --hard HEAD
fatal: this operation must be run in a work tree
PS C:\Python\python file\.git> cd ../
PS C:\Python\python file> git reset --hard HEAD
HEAD is now at 4e038bf Initial commit
PS C:\Python\python file> git reset --hard HEAD
HEAD is now at 4e038bf Initial commit
PS C:\Python\python file> git reset --hard HEAD fc609ada
fatal: Cannot do hard reset with paths.
PS C:\Python\python file> git reset --hard fc609ada
Updating files: 100% (1122/1122), done.
HEAD is now at fc609ad Initial commit
PS C:\Python\python file>
'''
