'''
参考文献：https://arxiv.org/abs/1801.09414
代码参考：https://blog.csdn.net/qq_34914551/article/details/104522030
'''

# 导入所需要的包
import torch
import wandb
import torchvision
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

''' 数据准备 '''
transform = transforms.Compose([
    transforms.Resize(size=100),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# 训练集数据
train_dir = "./att_faces/train/"  # 训练集地址
traindata = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
# 测试集数据
test_dir = "./att_faces/test/"  # 训练集地址
testdata = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

trainloader = DataLoader(
    traindata,
    shuffle=True,
    batch_size=32
)
testloader = DataLoader(
    testdata,
    shuffle=False,
    batch_size=8
)

''' 
Cosface实现 https://blog.csdn.net/qq_34914551/article/details/104522030
'''
# Cosface实现
class CosineMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(CosineMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        # margin
        self.m = m
        # 权重
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight)

    # 前向计算
    def forward(self, input, label, device):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * (cosine - self.m)) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

''' 训练与测试 '''
# 定义网络的预训练
def train(net, margin, train_loader, test_loader, device, l_r=0.1, num_epochs=25):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='LeNet-5', resume='allow', anonymous='must')
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 将网络移动到指定设备
    net = net.to(device)
    margin = margin.to(device)
    # 定义优化器
    optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': margin.parameters()}],
                                lr=l_r, momentum=0.9, weight_decay=5e-4)
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
            features = net(imgs)
            # 计算cosine值
            output = margin(features, labels, device)
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
        for step, (imgs, labels) in enumerate(train_loader):
            # 训练使用的数据移动到指定设备
            imgs = imgs.to(device)
            labels = labels.to(device)
            features = net(imgs)
            # 计算cosine值
            output = margin(features, labels, device)
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

''' main '''
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 使用torchvision.models模块中集成的预训练好的ResNet网络
    net = torchvision.models.resnet18(pretrained=True)
    margin = CosineMarginProduct(in_features=1000, out_features=10)
    # 开始训练
    train(net, margin, trainloader, testloader, device, l_r=0.1, num_epochs=25)
