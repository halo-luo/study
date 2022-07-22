# -*- codeing = utf-8 -*-
# @Time : 2022/7/12 10:18
# @Author : 骆龙飞
# @File : test.py
# @Software: PyCharm

import torch
import torchvision
from PIL import Image
from model import MyModel

img = Image.open("1.png")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)

img = torch.reshape(img, (1, 3, 32, 32))

model = torch.load("./model_loss/model_10_loss_1.1595516204833984.pth", map_location=torch.device('cpu'))

model.eval()
with torch.no_grad():
    output = model(img)
images = torch.reshape(img, (3, 32, 32))
trans = torchvision.transforms.ToPILImage()
image = trans(images)
image.show()
print(output)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(output.argmax(1))
