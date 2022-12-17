"""
代码参考：https://zhuanlan.zhihu.com/p/32506912
"""
"""
数据预处理
"""
# 导入所需的包
import os
import torch
import wandb
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 数据集中的所有类别名称
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# 每个类的颜色的RGB值
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

# 将RGB标签图片转变为单通道的分类类别值图，返回Numpy数据
def image2label(image, colormap):
    cm2lbl = np.zeros(256 ** 3)  # 每个像素点有0～255的选择，RGB 三个通道
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引
    image = np.array(image, dtype='int32')
    idx = (image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

# 将单通道的分类类别值图转变为RGB标签图片，返回Numpy数据，类型为uint8
def label2image(data, colormap):
    cm = np.array(colormap).astype('uint8')
    data = data.numpy()
    return cm[data]

# 定义使用的数据集
class MyDataset(Dataset):
    # 初始化数据集方法；
    # root：数据集根目录；train：是否是训练集，布尔型；size：裁剪大小，元组
    def __init__(self, root='./VOC2012/', train=True, size=(256, 256)):
        self.size = size
        self.root = root
        # 获取数据集中的训练图片及其标签名称列表
        self.data_list, self.label_list = self.read_images(root=self.root, train=train)
        print('Read ' + str(len(self.data_list)) + ' images')

    def read_images(self, root='./VOC2012/', train=True):
        txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
        return data, label

    # 返回最终的的数据集长度
    def __len__(self):
        return len(self.data_list)

    # 定义图像变换，返回tensor数据
    def mytransforms(self, image, label, size):
        # 找出图像的最长边
        max_len = max(image.size)
        # 构建一个(max_len, max_len)大小的RGB背景图片
        new_img = Image.new('RGB', (max_len, max_len), "black")
        new_lab = Image.new('RGB', (max_len, max_len), "black")
        # 将图像粘贴到背景图片上
        new_img.paste(image, (0, 0))
        new_lab.paste(label, (0, 0))
        # 将形成的新图片变成固定大小
        image = new_img.resize(size, Image.NEAREST )
        label = new_lab.resize(size, Image.NEAREST )
        # 转为Tensor数据并正则化
        im_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = im_transforms(image)
        # 转变标签图片
        label = image2label(label, colormap)
        label = torch.from_numpy(label)
        return image, label

    # 定义__getitem__方法，使用数据集
    def __getitem__(self, idx):
        # 得到图像
        img = self.data_list[idx]
        img = Image.open(img)
        # 得到标签
        label = self.label_list[idx]
        label = Image.open(label).convert('RGB')
        # 图像变换
        img, label = self.mytransforms(img, label, self.size)
        return img, label


'''
定义fcn网络以及上采样的双线性插值初始化
'''
# 定义双线性插值卷积核的初始化
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

# 定义FCN网络模型
class FCN_8x(nn.Module):
    # 初始化；num_classes：分类的类别数量
    def __init__(self, num_classes):
        # 使用预训练的ResNet34网络
        pretrained_net = models.resnet34(pretrained=True)
        super(FCN_8x, self).__init__()
        # 前三个下采样过程
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        # 第四个下采样过程
        self.stage2 = list(pretrained_net.children())[-4]
        # 第五个下采样过程
        self.stage3 = list(pretrained_net.children())[-3]
        # 使用1×1的卷积核改变通道数，score1对应stage3过程后的通道数改变
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        # score2对应stage2过程后的通道数改变
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        # score3对应stage1过程后的通道数改变
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        # 由1/32图片进行二倍上采样
        self.upsample_stage3_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        # 使用双线性插值初始化卷积核
        self.upsample_stage3_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        # 由1/16图片进行二倍上采样
        self.upsample_stage2_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_stage2_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        # 由1/8图片进行八倍上采样
        self.upsample_stage1_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_stage1_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

    def forward(self, x):
        # 输出结果是原图的1/8
        x1 = self.stage1(x)
        # 输出结果是原图的1/16
        x2 = self.stage2(x1)
        # 输出结果是原图的1/32
        x3 = self.stage3(x2)
        # 改变x3的通道数
        s3 = self.scores1(x3)
        # x3上采样2倍，s3是原图的1/16倍
        s3 = self.upsample_stage3_2x(s3)
        # 改变x2的通道数
        s2 = self.scores2(x2)
        # 按特征点相加
        s2 = s2 + s3
        # 改变x1的通道数
        s1 = self.scores3(x1)
        # s2上采样2倍，变为原图的1/8倍
        s2 = self.upsample_stage2_2x(s2)
        # 按特征点相加
        s = s1 + s2
        # s1上采样8倍，输出结果
        outer = self.upsample_stage1_8x(s)
        return outer

"""
定义训练方法
"""
# 定义训练方法
def train(net, train_loader, val_loader, device, num_epochs=5, l_r=0.0002):
    # 使用wandb跟踪训练过程
    config = dict(epochs=num_epochs, learning_rate=l_r,)
    experiment = wandb.init(project='FCN_8x', config=config, resume='allow', anonymous='must')
    # 设置优化器，生成器和判别器均使用Adam优化器
    optimizer = optim.SGD(net.parameters(), lr=l_r, weight_decay=1e-4)
    criterion = torch.nn.NLLLoss()
    # 训练过程
    for epoch in range(num_epochs):
        # 训练阶段
        net.train()
        train_loss = 0.0
        for step, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            out = net(image)
            out = F.log_softmax(out, dim=1)
            pred_train = torch.argmax(out, dim=1)
            # 计算训练损失
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 输出训练信息
            print('[%d/%d][%d/%d] Loss_D: %.3f'
                  % (epoch, num_epochs-1, step, len(train_loader)-1, loss.item()))
        print(f"Average loss of the whole eopch: {train_loss/len(train_loader)}")
        # 测试阶段
        net.eval()
        val_loss = 0.0
        for step, (image, label) in enumerate(val_loader):
            vimage = image.to(device)
            vlabel = label.to(device)
            out = net(vimage).detach()
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, vlabel)
            pred_val = torch.argmax(out, dim=1)
            val_loss += loss.item()
        # 使用wandb保存可视化信息
        experiment.log({
            'epoch': epoch,
            'train loss': train_loss/len(train_loader),
            'val loss': val_loss/len(val_loader),
            'learning rate': optimizer.param_groups[0]['lr'],
            # 将同一批次的生成图片进行可视化
            'val_image1': wandb.Image(label2image(pred_val[0].cpu(), colormap)),
            'val_image2': wandb.Image(pred_val[0].float().cpu())
        })
        print(f"Average loss of the whole eopch: {val_loss/len(val_loader)}")
        # 一个epoch训练结束，保存模型
        torch.save(net.state_dict(), './net.pth')
    # 最后可视化预测图片对应的原图和标签图
    experiment.log({
        # 将同一批次的生成图片进行可视化
        'val_image': wandb.Image(vimage[0].float().cpu()),
        'val_label': wandb.Image(label2image(vlabel[0].cpu(), colormap))
    })

if __name__ == "__main__":
    # 定义训练使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练集
    traindata = MyDataset('./VOC2012/', True, (256,256))
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)
    # 测试集
    valdata = MyDataset('./VOC2012/', False, (256, 256))
    # shuffle=False保证每次可视化输出的都是同一幅图像的预测效果，以便比较
    valloader = DataLoader(valdata, batch_size=64, shuffle=False)
    net = FCN_8x(21).to(device)
    # 开始训练
    train(net, trainloader, valloader, device, num_epochs=80, l_r=0.08)
