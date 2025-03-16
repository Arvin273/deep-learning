import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1
        '''
        如何计算特征图经过卷积核之后的尺寸，计算公式是什么
        outSize = [(特征图size - 卷积核size + 2 * padding) / stride] + 1
        '''
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        # 池化层1
        self.pool1 = nn.AvgPool2d(2, 2)
        # 卷积层2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化层2
        self.pool2 = nn.AvgPool2d(2, 2)
        # 卷积层3
        self.conv3 = nn.Conv2d(16, 120, 5)
        # 激活函数
        self.relu = nn.ReLU()
        # 展平
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
