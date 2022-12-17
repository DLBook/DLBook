'''
参考资料：
PyTorch官方文档 https://pytorch.org/docs/stable/index.html
孙玉林，余本国. PyTorch深度学习入门与实践 (案例视频精讲) 37-83
吴茂贵，郁明敏，杨本法，李涛，张粤磊. Python深度学习 基于PyTorch 67-99
'''

'''
神经网络模型的整个训练过程
'''
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 预处理
transfrom = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.ToTensor()
])
# 准备训练集数据和验证集
train_dataset = datasets.MNIST(root='./data',transform=transfrom,train=True,download=True)
val_dataset = datasets.MNIST(root='./data',transform=transfrom,train=False,download=True)

# 训练集dataloader，验证集dataloader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False
)

#################################################################
# 2.网络部分
from torch import nn
class Net(nn.Module):
    # 初始化
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear = nn.Linear(in_features=288, out_features=10, bias=True)

    # 定义前向计算过程
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x= self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        # 将特征展平
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x

net = Net()
#################################################
# 3.1损失函数
loss_fn = nn.CrossEntropyLoss()

#################################################
# 3.2优化器，学习率调整器
from torch import optim

# 使用SGD优化器，初始学习率为0.0001
optimizer = optim.SGD(net.parameters(), lr=0.0001)
# 使用StepLR学习率调整方式，每隔5个Epoch学习率变为原来的0.1倍
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

################################
# 3.3迭代训练10个Epoch
num_epochs = 10
print("Start Train!")
for epoch in range(1, num_epochs+1):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    # 训练阶段，设为训练状态
    net.train()
    for batch_index, (imgs,labels) in enumerate(train_loader):
        output = net(imgs)
        loss = loss_fn(output,labels) #计算损失
        optimizer.zero_grad()# 梯度清零
        loss.backward()# 反向传播
        optimizer.step()# 梯度更新
        # 计算精确度
        _,predict = torch.max(output,dim=1)
        corec = (predict == labels).sum().item()
        acc = corec / imgs.shape[0]
        train_loss += loss.item()
        train_acc += acc
    # 验证阶段
    net.eval()
    for batch_index, (imgs, labels) in enumerate(val_loader):
        output = net(imgs)
        loss = loss_fn(output,labels)
        _, predict = torch.max(output, dim=1)
        corec = (predict == labels).sum().item()
        acc = corec / imgs.shape[0]
        val_loss += loss.item()
        val_acc += acc
    # 依次输出当前epoch、总的num_epochs，
    # 训练过程中当前epoch的训练损失、训练精确度、验证损失、验证精确度和学习率
    print(epoch, '/', num_epochs, train_loss/len(train_loader),
          train_acc/len(train_loader), val_loss/len(val_loader),
          val_acc/len(val_loader),optimizer.param_groups[0]['lr'])

    scheduler.step()


'''
一些常见的transforms变换
'''
# # 导入transforms模块
# from torchvision import transforms
#
# size = 32 # 或 (32, 32)
# p = 0.5
# mean= [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]
# padding = 1 # 或 (1, 1) 或 (1, 1, 1, 1)
# fill = 0
# brightness = 0
# contrast = 0
# saturation = 0
# hue = 0
#
# # 将PIL图像调整为给定大小
# transforms.Resize(size)
# # 依据给定的size从PIL图像中心裁剪
# transforms.CenterCrop(size)
# # 在PIL图像上随机裁剪出给定大小
# transforms.RandomCrop(size)
# # 将PIL图像裁剪为随机大小和宽高比，然后resize到给定大小
# transforms.RandomResizedCrop(size)
# # PIL图像依概率p水平翻转，p默认值为0.5
# transforms.RandomHorizontalFlip(p)
# # 在PIL图像四周使用fill值进行边界填充，填充像素个数为padding
# transforms.Pad(padding, fill)
# # 对PIL图像进行高斯模糊
# transforms.GaussianBlur(kernel_size, sigma)
# # 调整PIL图像的亮度、对比度、饱和度、色调
# transforms.ColorJitter(brightness, contrast, saturation, hue)
# # PIL图像依概率p随即变为灰度图，p默认值为0.5
# transforms.RandomGrayscale(p)
# # 将PIL图像或者ndarray转换为Tensor，并且归一化至[0-1]
# transforms.ToTensor()
# # 用平均值和标准偏差归一化张量
# transforms.Normalize(mean, std)
# # 将Tensor或者ndarray数据转换为PIL图像
# transforms.ToPILImage()
#
# '''
# 对数据集中的每个图像执行：
#     1）大小调整至32×32大小，
#     2）依0.5的概率进行水平翻转，
#     3）最后将PIL图像变为Tensor数据
# '''
# transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ToTensor()
# ])

'''
预处理并加载MNIST数据集
'''
# # 使用torchvision.datasets包下面的MNIST数据集类获得MNIST数据集
# from torchvision.datasets import MNIST
# from torchvision import transforms
# from torch.utils.data import DataLoader
#
# # 定义数据增强
# transform = transforms.Compose([
#     transforms.Resize(32),
#     # transforms.RandomHorizontalFlip(0.5),
#     transforms.ToTensor()
# ])
# train_dataset = MNIST(
#     root='./data',# 数据集的存放或下载地址
#     transform=transform,#数据增强
#     train=True,# 是否为训练集
#     download=True# 是否下载，如果上述地址已存在该数据集则不下载
# )
# test_dataset = MNIST(
#     root='./data',
#     transform=transform,
#     train=True,
#     download=True
# )
# # 将预处理好的数据集变为可迭代对象，每次使用一个batch数量的数据
# train_loader = DataLoader(
#     dataset=train_dataset,# 数据集
#     batch_size=16,# batch大小
#     shuffle=True# 是否打乱顺序后取出
# )
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=8,
#     shuffle=False
# )

'''
查看预处理后的数据
'''
# # 查看预处理后的一个MNIST数据及其标签
# print(train_dataset[0])
# # 查看预处理后的一个MNIST数据的形状
# print(train_dataset[0][0].shape)
# # 取出一个batch
# batch_data, batch_label = next(iter(train_loader))
# # 查看一个batch数据的形状
# print(batch_data.shape)
# # 查看一个batch数据对应的标签的形状
# print(batch_label.shape)

'''
数据集数据可视化
'''
# import matplotlib.pyplot as plt
# # 得到一个batch数量的MNIST数据及其对应的标签
# batch_data, batch_label = next(iter(train_loader))
# fig = plt.figure()
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(batch_data[i][0])
#     plt.title("{}".format(batch_label[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

'''
自定义数据集
'''
# import torch
# from torch.utils.data import Dataset
#
# class MyDataset(Dataset):
#     # 初始化方法
#     def __init__(self):
#         # 由3个4维向量组成的数据集
#         self.data_list = torch.tensor([[0, 1, 2, 3],
#                                     [4, 5, 6, 7],
#                                     [8, 9, 0, 1]])
#         # 对应的标签
#         self.label_list = torch.tensor([0, 1, 2])
#
#     def __len__(self):
#         return len(self.data)
#
#     # 根据索引每次取一个数据
#     def __getitem__(self, index):
#         data = self.data_list[index]
#         label = self.label_list[index]
#         return data, label
#
# # 获取自定义数据集的数据
# dataset = MyDataset()
# # 取出第一个数据及其标签
# print(dataset[0])

'''
使用ImageFolder
'''
'''
—————data
    |—————train
    |    |——————class1
    |    |       |————class1_data1   
    |    |       |————class1_data2
    |    |       |————class1_···
    |    |——————class2
    |    |       |————class2_data1   
    |    |       |————class2_data2
    |    |       |————class2_···
    |    ···
    |—————test

# 根据以上文件结构使用ImageFolder
from torchvision.datasets import ImageFolder
train_dataset = ImageFolder(
    root="./data/train/",
    transform=transform
)
test_dataset = ImageFolder(
    root="./data/test/",
    transform=transform
)
'''

'''
创建深度神经网络
'''
# from torch import nn
#
# class Net(nn.Module):
#     # 初始化
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
#         self.relu1 = nn.ReLU()
#         self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0)
#         self.relu2 = nn.ReLU()
#         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.linear = nn.Linear(in_features=288, out_features=10, bias=True)
#
#     # 定义前向计算过程
#     def forward(self,x):
#         # 假设输入的样本大小为[N, 1, 28, 28]
#         # 输出大小[N, 16, 28, 28]
#         x = self.conv1(x)
#         x = self.relu1(x)
#         # 输出大小[N, 16, 14, 14]
#         x= self.max_pool1(x)
#         # 输出大小[N, 8, 12, 12]
#         x = self.conv2(x)
#         x = self.relu2(x)
#         # 输出大小[N, 8, 6, 6]
#         x = self.max_pool2(x)
#         # 将特征展平，变为[N, 8*6*6]，即[N, 288]
#         x = x.view(x.shape[0],-1)
#         # 输出大小[N, 10]
#         x = self.linear(x)
#         return x

'''
损失函数
'''
# import torch
# from torch import nn
#
# # 设置MSE损失
# loss_fn = nn.MSELoss()
# # 假设网络输出output，batch_size为2
# output = torch.randn(2, 10, requires_grad=True)
# # 真实值，与output的形状相同
# target = torch.randn(2, 10)
# loss = loss_fn(output, target)
# loss.backward()
# nn.LeakyReLU

'''
两种不同的优化器使用方式
'''
# net = Net()
# # 方式一：
# # 使用Adam优化器，各层参数的学习率统一设置为0.001
# optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
#
# # 方式二：为不同的层定义表不同的学习率
# optimizer = torch.optim.Adam(
#     [{"params": net.conv1.parameters(), "lr": 0.01},
#      {"params": net.conv2.parameters(), "lr": 0.001},
#      {"params": net.linear.parameters()}],
#     lr=1e-5
# )

'''
优化器在神经网络训练过程中的使用格式
'''
# for input, target in dataloader:
#     optimizer.zero_grad()         # 梯度清零
#     output = net(input)           # 前向计算预测值
#     loss = loss_fn(input, target) # 使用损失函数loss_fn计算损失
#     loss.backward()               # 损失反向传播
#     optimizer.step()              # 使用优化器更新网络参数


'''
优化器和学习率调整方式的使用
'''
# # 创建神经网络
# net = Net()
# # 设置优化器
# optimizer = torch.optim.Adam(params=net.parameters(), lr=0.1)
# # 设置学习率调整方式
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# for epoch in range(100):
#     for input, target in dataloader:
#         optimizer.zero_grad()  # 梯度清零
#         output = net(input)  # 前向计算预测值
#         loss = loss_fn(input, target)  # 使用损失函数loss_fn计算损失
#         loss.backward()  # 损失反向传播
#         optimizer.step()  # 使用优化器更新网络参数
#     scheduler.step() # 一个epoch结束，更新学习率

# 1.准备数据

