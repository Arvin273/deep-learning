import torch
from torch import nn
from torchvision import datasets, transforms
from model import LeNet
from PIL import Image
import argparse

# 数据预处理（与训练时相同）
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

detect_data = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
model.load_state_dict(torch.load('./lenet_mnist_best.pth'))

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
show = transforms.ToPILImage()

for i in range(5):
    images, labels = detect_data[i][0], detect_data[i][1]
    show(images).show()
    images = torch.unsqueeze(images, dim=0).to(device)

    with torch.no_grad():
        output = model(images)
        predicted = classes[torch.argmax(output).item()]
        print(f"predicted: {predicted}  actual: {labels}")


