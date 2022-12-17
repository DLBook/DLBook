

# 导入所需要的包
import os
import wandb
import random
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

'''
数据集格式处理
'''
# # 对应其中类别
# classes = ['NE','HA','AN','DI','FE','SA','SU']
# folder_names = ['train', 'test']
# # 未划分的数据集地址
# src_data_folder = "./jaffe"
# # 划分后的数据集保存地址
# target_data_folder = "./data/jaffe"
# # 划分比例
# train_scale = 0.8
# # 在目标目录下创建训练集和验证集文件夹
# for folder_name in folder_names:
#     folder_path = os.path.join(target_data_folder, folder_name)
#     os.mkdir(folder_path)
#     # 在folder_path目录下创建类别文件夹
#     for class_name in classes:
#         class_folder_path = os.path.join(folder_path, class_name)
#         os.mkdir(class_folder_path)
#
# # 获取所有的图片
# files = os.listdir(src_data_folder)
# data = [file for file in files if file.endswith('tiff') and not file.startswith('.')]
# # 随机打乱图片顺序
# random.shuffle(data)
# # 统计保存各类图片数量
# class_sum = dict.fromkeys(classes, 0)
# for file in data:
#     class_sum[file[3:5]] += 1
#
# # 记录训练集各类别图片的个数
# class_train = dict.fromkeys(classes, 0)
# # 记录测试集各类别图片的个数
# class_test = dict.fromkeys(classes, 0)
# # 遍历每个图片划分
# for file in data:
#     # 得到原图片目录地址
#     src_img_path = os.path.join(src_data_folder, file)
#     # 如果训练集中该类别个数未达划分数量，则复制图片并分配进入训练集
#     if class_train[file[3:5]] < class_sum[file[3:5]]*train_scale:
#         target_img_path = os.path.join(os.path.join(target_data_folder, 'train'), file[3:5])
#         shutil.copy2(src_img_path, target_img_path)
#         class_train[file[3:5]] += 1
#     # 否则，进入测试集
#     else:
#         target_img_path = os.path.join(os.path.join(target_data_folder, 'test'), file[3:5])
#         shutil.copy2(src_img_path, target_img_path)
#         class_test[file[3:5]] += 1
#
# # 输出标明数据集划分情况
# for class_name in classes:
#     print("-" * 10)
#     print("{}类共{}张图片,划分完成:".format(class_name, class_sum[class_name]))
#     print("训练集：{}张，测试集：{}张".format(class_train[class_name], class_test[class_name]))

# 数据加载及预处理
# transforms.Compose()组合定义图像预处理操作
transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载数据
traindata = ImageFolder(
    root="./data/jaffe/train",
    transform=transform_train
)
testdata = ImageFolder(
    root="./data/jaffe/test",
    transform=transform_test
)
trainloader = DataLoader(
    traindata, batch_size=4, shuffle=True)
testloader = DataLoader(
    testdata, batch_size=2, shuffle=False)

# 基础的残差模块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        # inplace为True，将计算得到的值直接覆盖之前的值,可以节省时间和内存
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.downsample = None
        if ch_out != ch_in:
            # 如果输入通道数和输出通道数不相同，使用1×1的卷积改变通道数
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample != None:
            identity = self.downsample(x)
        out += identity
        relu = nn.ReLU()
        out = relu(out)
        return out

# 实现ResNet网络
class ResNet(nn.Module):
    # 初始化；block：残差块结构；layers：残差块层数；num_classes：输出层神经元即分类数
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        # 改变后的通道数
        self.channel = 64
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, self.channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 残差网络的四个残差块堆
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，也是输出层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 用于堆叠残差块
    def _make_layer(self, block, ch_out, blocks, stride=1):
        layers = []
        layers.append(block(self.channel, ch_out, stride))
        self.channel = ch_out * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.channel, ch_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ResNet18生成方法
def resnet18(num_classes = 1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


# 定义网络的预训练
def train(net, train_loader, test_loader, device, l_r = 0.0002, num_epochs=25,):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='Facial Expression Recognition', resume='allow', anonymous='must')
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
    # 将网络移动到指定设备
    net = net.to(device)
    # 正式开始训练
    for epoch in range(num_epochs):
        # 保存一个Epoch的损失
        train_loss = 0
        # 计算准确度
        test_corrects = 0
        # 设置模型为训练模式
        net.train()
        for step, (imgs, labels) in enumerate(train_loader):
            # 训练使用的数据移动到指定设备
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = net(imgs)
            # 计算损失
            loss = criterion(output, labels)
            # 将梯度清零
            optimizer.zero_grad()
            # 将损失进行后向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()
            train_loss += loss.item()
        # 设置模型为验证模式
        net.eval()
        for step, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = net(imgs)
            pre_lab = torch.argmax(output, 1)
            corrects = (torch.sum(pre_lab == labels.data).double() / imgs.size(0))
            test_corrects += corrects.item()
        # 一个Epoch结束时，使用wandb保存需要可视化的数据
        experiment.log({
            'epoch':epoch,
            'train loss': train_loss / len(train_loader),
            'test acc': test_corrects / len(test_loader),
        })
        print('Epoch: {}/{}'.format(epoch, num_epochs-1))
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss / len(train_loader)))
        print('{} Test Acc:{:.4f}'.format(epoch, test_corrects / len(test_loader)))
        # 保存此Epoch训练的网络的参数
        torch.save(net.state_dict(), './net.pth')

if __name__ == "__main__":
    # 定义训练使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet18(num_classes=7)
    train(net, trainloader, testloader, device, l_r=0.00003, num_epochs=30)
