import torch
from torch import nn
from torch import optim
from model import AlexNet
from torchvision import datasets, transforms

# 1.数据预处理（转换成张量，resize）
data_transform = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ]
)

# 2.加载数据集
# 训练集
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)

# 验证集
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=True)

# 判断当前系统是否有 GPU (CUDA)，如果有，则使用 GPU 进行加速；否则使用 CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)

# 3.定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 4.定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 5.开始训练
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

        # 输出这一轮(epoch)的平均损失值
        avg_loss = round(total_loss / len(train_dataloader), 4)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

        # 每个 epoch 结束后评估模型
        accuracy = evaluate(model, test_dataloader, device)
        # 如果当前准确率高于最佳准确率，则保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {accuracy:.2f}%")

# 定义评估函数
def evaluate(model, test_dataloader, device):

    model.eval()  # 设置为评估模式
    # 正确样本数
    correct = 0
    # 总样本数
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            # outputs的形状[32, 10]
            # 对于FashionMNIST数据集，类别数是10，一批32张图片
            # 因此形状是[32, 10]
            outputs = model(images)

            '''
                torch.max(outputs, 1) 会返回两个值：
                第一个值是最大概率值（我们不关心）。
                第二个值是最大值所在的索引，即预测的类别 predicted。
                ✅示例：
                    outputs = [
                        [0.1, 0.3, 0.7],  # 第0张图片，预测类别是2
                        [0.8, 0.1, 0.1],  # 第1张图片，预测类别是0
                        [0.2, 0.5, 0.3],  # 第2张图片，预测类别是1
                    ]
                    torch.max(outputs, 1) 返回的是：
                    (predicted_value, predicted_index) = (tensor([0.7, 0.8, 0.5]), tensor([2, 0, 1]))
                也就是把tensor[2,0,1]赋值给了predicted
            '''
            _, predicted = torch.max(outputs, 1)

            '''
                labels.size(0) 获取的是 labels 张量的第 0 维度的长度，也就是 当前批次(batch)中的样本数量。
                
                为什么是 size(0) 而不是 size(1)？
                因为：
                
                第 0 维度 (dim=0) 是 批次维度，表示样本数量。
                第 1 维度 (dim=1) 是类别（对于分类任务通常是10个类别）
            '''
            total += labels.size(0)

            '''
                 统计预测正确的样本数量
                correct += (predicted == labels).sum().item()
                (predicted == labels) 是一个布尔张量：
                [True, False, True, True, False, ...]
                
                .sum() 会计算布尔值为 True 的数量，即当前批次中预测正确的样本数量。
                .item() 将张量转换为一个具体的python数值。
            '''
            correct += (predicted == labels).sum().item()

    # 计算正确率
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    # 训练
    train(model, train_dataloader, loss_fn, optimizer, device, epochs=20)
