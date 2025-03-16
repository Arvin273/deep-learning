import torch
from torch import nn
from torch import optim
from model import LeNet
from torchvision import datasets, transforms

'''
torchvision是一个用于计算机视觉任务的库，提供了很多的数据集、模型架构
以及图像转换工具。在处理图像数据时，通常需要对原始图像进行一些预处理操作
以便更好地适配模型训练
'''

'''
数据预处理
Compose是PyTorch提供的一个类，作用是把多个图像操作组合成一个流水线（pipeline）
传入的是一个列表，列表中的每个元素都是一个转换操作，可以添加多个操作，现在只有一个操作ToTensor()
transforms.ToTensor()是将图像转换成张量，同时会把图像的像素值从0-255的范围归一化到0-1
'''
data_transform = transforms.Compose(
    [transforms.ToTensor()]
)

'''
datasets.MNIST 是 PyTorch 提供的一个内置数据集，用于加载 MNIST 手写数字数据集。
root='./data' 指定数据集存储的路径，如果路径不存在，PyTorch 会自动创建。
train=True 表示加载的是训练集，如果设置为 False，则加载的是测试集。
download=True 如果数据集没有下载，PyTorch 会自动下载。
transform=data_transform 应用了前面定义的 data_transform 预处理操作（将图片转换为张量）。
'''
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)

'''
DataLoader 是 PyTorch 提供的工具，可以对数据集进行批量加载和打乱。
dataset=trainset 指定使用的数据集。
batch_size=32 每次取 32 张图片作为一个批次。
shuffle=True 打乱数据顺序，增加训练的随机性，防止模型过拟合。
'''
train_dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)

#判断当前系统是否有 GPU (CUDA)，如果有，则使用 GPU 进行加速；否则使用 CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
LeNet() 是定义的一个卷积神经网络模型 (LeNet-5)。
.to(device) 将模型移动到指定的设备 (GPU 或 CPU)。
'''
model = LeNet().to(device)

'''
定义损失函数
CrossEntropyLoss() 是交叉熵损失函数，常用于分类任务。
'''
loss_fn = nn.CrossEntropyLoss()

'''
优化器是用来更新参数的
SGD 是随机梯度下降 (Stochastic Gradient Descent) 优化器。
model.parameters() 是 PyTorch 内部的机制，model通过继承 nn.Module，自动收集了模型中的所有可训练参数，供优化器使用。
parameters()中获取了哪些内容？
    卷积层 conv1 的权重和偏置
    卷积层 conv2 的权重和偏置
    卷积层 conv3 的权重和偏置
    全连接层 fc1、fc2的权重和偏置
lr=0.001 是学习率。
momentum=0.9 是动量项，可以帮助加快收敛速度。
'''
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 定义训练函数
def train(model, train_dataloader, loss_fn, optimizer, device, epochs=20):
    best_accuracy = 0  # 初始化最佳准确率
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        total_loss = 0
        for images, labels in train_dataloader:
            # 一个循环加载batch_size张图片（32张）
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = loss_fn(outputs, labels)

            # 反向传播
            # 梯度清零，防止之前的梯度造成影响
            optimizer.zero_grad()
            # 从损失函数开始，逐层计算损失函数对每个模型参数的梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

        # 每个 epoch 结束后评估模型
        accuracy = evaluate(model, test_dataloader, device)

        # 如果当前准确率高于最佳准确率，则保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, 'lenet_mnist_best.pth')
            print(f"Model saved with accuracy: {accuracy:.2f}%")


# 定义评估函数
def evaluate(model, test_dataloader, device):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            # 得到模型预测结果
            outputs = model(images)
            # 拿到概率最大的类别（一批图片）
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# 定义保存模型函数
def save_model(model, filename='lenet_mnist.pth'):
    torch.save(model.state_dict(), filename)

# 训练和评估模型
train(model, train_dataloader, loss_fn, optimizer, device, epochs=20)