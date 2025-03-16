# 深度学习训练过程

## 1. 数据准备
### 数据收集
收集与任务相关的数据（如图像、文本、音频等）

### 数据预处理
1.清洗数据（去除噪音...)
2.标准化或归一化数据
3.数据增强
如：
data_transform = transforms.Compose(
    [transforms.ToTensor()]
)

### 数据集划分
将数据集划分为训练集、验证集和测试集（如7：2：1）

### 数据加载
使用数据加载器（如Pytorch的DataLoader）将数据分批加载到模型中
如：
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)

## 2. 模型设计
### 选择模型架构
根据任务选择合适的模型（如卷积神经网络 CNN 用于图像分类，循环神经网络 RNN 用于序列数据）。
可以使用预训练模型（如 ResNet、BERT）进行迁移学习。

### 定义模型结构
使用框架（如 PyTorch、TensorFlow）定义模型的结构（如层数、激活函数、损失函数等）。
如：
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

## 3. 训练配置
1.选择损失函数（如交叉熵损失用于分类任务，均方误差用于回归任务）
2.选择优化器（如SGD、Adam）并设置学习率，优化器是用来更新参数的
3.设置超参数：设置批量大小（batch size）、学习率、训练轮数等超参数
4.将模型和数据移动到GPU上

如下：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 20


## 4. 模型训练
1.前向传播：将输入数据传递给模型，计算输出
2.计算损失：使用损失函数计算模型输出与真实标签之间的误差
3.反向传播：计算损失对模型参数的梯度
4.参数更新：使用优化器更新模型的权重和偏置

如下：
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


## 5. 验证与调优
1.验证模型：在验证集上评估模型的性能，计算验证损失和准确率
2.调整超参数：根据验证结果调整学习率、批量大小等超参数
3.防止过拟合：使用正则化技术或早停来防止过拟合

## 6. 模型测试
1.测试模型：在测试集上评估模型的性能，计算损失和准确率
2.分析结果：分析模型的错误，找出改进的方向

## 7. 其他
1.训练可以有多轮（epochs）,一轮有多批(batch)，batch = 数据总数量 / batch_size(一批多少个数据)  
2.每处理完一批batch数据，进行一次参数更新
3.每处理完一轮epoch，进行一次验证，如果在验证集上的准确率提高，则保存当前的权重为best.pth