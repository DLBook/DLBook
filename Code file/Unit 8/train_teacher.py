# 导入所需的包
import torch
import wandb
import torch.nn as nn
from torchvision import models
from train_baseline import trainloader, testloader

# 定义教师模型
teacher_resnet18 = models.resnet18(num_classes=10)

# 定义网络的预训练
def train(net, train_loader, test_loader, device, l_r=0.0003, num_epochs=10 ):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='ResNet18', resume='allow', anonymous='must')
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
        torch.save(net.state_dict(), './resnet18.pth')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = teacher_resnet18
    train (net,trainloader, testloader, device, l_r=0.0003, num_epochs=50)
