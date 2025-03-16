import torch
from torchvision import datasets, transforms
from model import AlexNet

# 数据预处理（与训练时相同）
data_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

# 加载数据集
detect_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
detect_dataloader = torch.utils.data.DataLoader(dataset=detect_data, batch_size=32, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 把模型放在设备上
model = AlexNet().to(device)
# 加载模型权重/参数
model.load_state_dict(torch.load('./best_model.pth'))

# 类别标签
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 显示图片组件
show = transforms.ToPILImage()

# 遍历前五张图片
for i in range(50):
    # 获取第 i 张图片和它对应的标签
    images, labels = detect_data[i][0], detect_data[i][1]

    # 先显示图片
    # show(images).show()

    '''
        ✅ 4. 增加一个维度
        images = torch.unsqueeze(images, dim=0).to(device)
        torch.unsqueeze() 是 增加一个批次维度，因为模型的输入需要是 [batch_size, channel, height, width]。
        假设图片原本的形状是 [1, 28, 28]，使用 unsqueeze() 后变成 [1, 1, 28, 28]。
        dim=0 表示在第 0 维度（batch）增加一个维度。
        to(device) 是把图片数据放到 GPU 上加速计算。
    '''
    images = torch.unsqueeze(images, dim=0).to(device)

    # 禁用梯度计算 (避免显存浪费)
    with torch.no_grad():
        '''
            output 是一个包含 10个类别概率 的张量（对于 FashionMNIST 数据集）：
            output -> tensor([0.1, 0.2, 0.05, 0.6, ...])  # 10个类别的概率
        '''
        output = model(images)
        '''
            output -> tensor([0.1, 0.2, 0.05, 0.6, ...])  # 10个类别的概率
            torch.argmax(output) 获取的是 概率最大的类别索引，比如 3。
            .item() 是把张量转换为具体的数值。
            classes 是类别名称列表
        '''
        predicted = classes[torch.argmax(output).item()]
        print(f"predicted: {predicted}  actual: {classes[labels]}")