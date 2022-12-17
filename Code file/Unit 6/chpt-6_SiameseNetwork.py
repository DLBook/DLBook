'''
参考资料：
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/tree/master
Contrastive Loss Base on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Dataset from: AT&T Laboratories, Cambridge. http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip.
'''

# 导入所需的包
import torch
import wandb
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils
import numpy as np
import random
from PIL import Image

# 训练集数据
train_dir = "./att_faces/train/"  # 训练集地址
traindata = torchvision.datasets.ImageFolder(root=train_dir)
# 测试集数据
test_dir = "./att_faces/test/"  # 训练集地址
testdata = torchvision.datasets.ImageFolder(root=test_dir)

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        super().__init__()
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def __getitem__(self, index):
        # 随机选一幅图片，img0_tuple的结构是（图片路径，类别）
        img0_tuple = self.imageFolderDataset.imgs[index]
        # 保证同类样本约占一半
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        # 读取图片
        img0 = Image.open(img0_tuple[0]).convert("L")
        img1 = Image.open(img1_tuple[0]).convert("L")
        # 数据变换
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        return img0, img1, label


# 数据预处理
transform = transforms.Compose([transforms.Resize((108, 108)),
                                transforms.ToTensor()])
# 训练集
siamese_dataset_train = SiameseNetworkDataset(
    imageFolderDataset=traindata,
    transform=transform
)
trainloader = DataLoader(siamese_dataset_train, shuffle=True, batch_size=32)
# 测试集
siamese_dataset_test = SiameseNetworkDataset(
    imageFolderDataset=testdata,
    transform=transform
)
testloader = DataLoader(siamese_dataset_test, shuffle=False, batch_size=8)

# 用来展示一幅tensor图像，输入是(C,H,W)
def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    # 设置插入的文本格式
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 构造孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 输入（108，108）的图片，输出大小是（54，54）
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            # 输出大小是（52，52）
            nn.Conv2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 输出大小是（26，26）
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 输出大小：（24，24）
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 24 * 24, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    # 单个分支的前向传播
    def forward_once(self, x):
        output = self.conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    # 前向传播
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Contrastive Loss
def contrastiveLoss(output1, output2, label, margin = 2.0):
    # 欧氏距离
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    # 类内损失
    within_loss = (1 - label) * torch.pow(euclidean_distance, 2)
    # 类间损失
    between_loss = label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    loss_contrastive = torch.mean(within_loss + between_loss)
    return loss_contrastive


# 定义网络的预训练
def train(net, train_loader, test_loader, criterion, device, num_epochs, l_r=0.0002):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='SiameseNetwork', resume='allow', anonymous='must')
    # 将网络移动到指定设备
    net = net.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
    for epoch in range(num_epochs):
        # 保存一个Epoch的损失
        train_loss = 0
        test_loss = 0
        # 设置模型为训练模式
        net.train()
        for step, (img0, img1, label) in enumerate(train_loader):
            # 训练使用的数据移动到指定设备
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output0, output1 = net(img0, img1)
            # 计算损失
            loss = criterion(output0, output1, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        net.eval()
        for step, (img0, img1, label) in enumerate(test_loader):
            # 训练使用的数据移动到指定设备
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output0, output1 = net(img0, img1)
            # 计算损失
            loss = criterion(output0, output1, label)
            test_loss += loss.item()
        # 使用wandb保存需要可视化的数据
        experiment.log({
            'Epoch': epoch,
            'train loss': train_loss / len(train_loader),
            'test loss': test_loss / len(test_loader),
        })
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('{} Train Loss:{:.4f}, Test Loss{:.4f}'.format(epoch,
                                                             train_loss / len(train_loader),
                                                             test_loss / len(test_loader)))
        # 保存此Epoch训练的网络的参数
        torch.save(net.state_dict(), './net.pth')

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SiameseNetwork()
    # 训练过程
    train(net, trainloader, testloader, contrastiveLoss, device, l_r=0.0005, num_epochs=100)
