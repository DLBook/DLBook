# 导入所需的包
import torch
import wandb
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

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

# 定义VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d((2, 2), 2)
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d((2, 2), 2)
        # 第五个卷积块
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),nn.ReLU(),
        )
        self.pool5 = nn.MaxPool2d((2, 2), 2)
        # 全连接层部分
        self.output = nn.Sequential(
            nn.Linear(512*7*7, 4096),nn.ReLU(),nn.Dropout(),
            nn.Linear(4096, 4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        x = x.view(x.size(0), -1)
        outer = self.output(x)
        return outer


# 定义网络的预训练
def train(net, train_loader, test_loader, device, l_r = 0.0002, num_epochs=25,):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='VGG16', resume='allow', anonymous='must')
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
    # 使用自定义的VGG16类实现VGG16网络
    # 由于CIFAR10只有10种类别，所以修改VGG网络的输出层神经元数量，num_classes = 10
    net = VGG16(num_classes = 10)
    train(net, trainloader, testloader, device, l_r=0.0003, num_epochs=10)
