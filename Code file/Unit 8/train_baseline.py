# 基础的LeNet网络精确度，作为对比实验
# 导入所需的包
import torch
import wandb
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 定义LeNet-5网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 输入3×32×32的Cifar10图像，经过卷积后输出大小为6×28×28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=1, bias=True)
        # 卷积操作后使用Tanh激活函数,激活函数不改变其大小
        self.tanh1 = nn.Tanh()
        # 使用最大池化进行下采样，输出大小为6×14×14
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2),stride=2)
        # 输出大小为16×10×10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, bias=True)
        self.tanh2 = nn.Tanh()
        # 输出大小为16×5×5
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 两个卷积层和最大池化层后接三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        #第三个全连接层是输出层，输出单元个数即是数据集中类的个数，Cifar1数据集有10个类
        self.classifier = nn.Linear(84, 10)
    # 定义前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        # 在全连接操作前将数据平展开
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.tanh4(x)
        output = self.classifier(x)
        return output




# 定义网络的预训练
def train(net, train_loader, test_loader, device, l_r=0.0003, num_epochs=10 ):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='LeNet5', resume='allow', anonymous='must')
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
            'epoch': epoch,
            'train loss': train_loss / len(train_loader),
            'test acc': test_corrects / len(test_loader),
        })
        # scheduler.step()
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss / len(train_loader)))
        print('{} Test Acc:{:.4f}'.format(epoch, test_corrects / len(test_loader)))
        # 保存此Epoch训练的网络的参数
        torch.save(net.state_dict(), './lenet5.pth')


# 使用Compose容器组合定义图像预处理方式
transf = transforms.Compose([
    # 改变图像大小
    transforms.Resize(32),
    # 将给定图片转为shape为(C, H, W)的tensor
    transforms.ToTensor()
])
# 数据准备
traindata = CIFAR10(
    # 数据集的地址
    root="./datasets",
    # 是否为训练集，True为训练集
    train=True,
    # 使用数据预处理
    transform=transf,
    # 是否需要下载， True为需要下载
    download=True
)
testdata = CIFAR10(
    root="./datasets",
    train=False,
    transform=transf,
    download=True
)
# 定义数据加载器
trainloader = DataLoader(
    # 需要加载的数据
    traindata,
    # 定义batch大小
    batch_size=128,
    # 是否打乱顺序，True为打乱顺序
    shuffle=True
)
testloader = DataLoader(
    testdata,
    batch_size=128,
    shuffle=False
)

net = LeNet5()
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train (net,trainloader, testloader, device, l_r=0.0003, num_epochs=50)
