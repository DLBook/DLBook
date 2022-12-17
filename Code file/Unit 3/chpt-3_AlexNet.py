# 导入所需的包
import torch
import wandb
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 使用Compose容器组合定义图像预处理方式
transf = transforms.Compose([
    # 改变图像大小
    transforms.Resize(224),
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

# 定义AlexNet网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 输入3×224×224的Cifar10图像
        self.conv1 = nn.Sequential(
            # 卷积操作后输出数据大小为96×55×55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 池化操作后输出数据大小为96×27×27
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            # 输出大小为256×27×27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 输出大小为256×13×13
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            # 输出大小为384×13×13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            # 输出大小为384×13×13
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            # 输出大小为256×13×13
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 输出大小为256×6×6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4096, 10)

    # 定义前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.classifier(x)
        return output


# 定义网络的预训练
def train(net, train_loader, test_loader, device, l_r = 0.0002, num_epochs=25,):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='AlexNet', resume='allow', anonymous='must')
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
    net = AlexNet()
    train(net, trainloader, testloader, device, l_r=0.0003, num_epochs=10)
