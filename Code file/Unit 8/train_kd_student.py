
# 本代码是训练蒸馏网络

# 导入所需的包
import torch
import wandb
from torch import nn
import torch.nn.functional as F
from train_baseline import LeNet5, trainloader, testloader
from train_teacher import teacher_resnet18
import os
os.environ["WANDB_API_KEY"] = '551081ea03d5be856fd332b18f3d3fb351e0bed8'

# 定义蒸馏损失函数
class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss



# 定义网络的预训练
def train_KD(student_model,teacher_model,alpha, temperature, train_loader, test_loader, device, l_r=0.0003, num_epochs=10 ):
    # 使用wandb跟踪训练过程
    experiment = wandb.init(project='KD_model', resume='allow', anonymous='must')
    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()
    distill_loss = DistillKL(T=temperature)
    # 定义优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=l_r)

    # 将网络移动到指定设备
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    # 正式开始训练
    for epoch in range(num_epochs):
        # 保存一个Epoch的损失
        train_loss = 0
        # 计算准确度
        test_corrects = 0
        # 设置模型为训练模式
        student_model.train()
        for step, (imgs, labels) in enumerate(train_loader):
            # 训练使用的数据移动到指定设备
            imgs = imgs.to(device)
            labels = labels.to(device)
            output_student = student_model(imgs)
            with torch.no_grad():
                output_teacher = teacher_model(imgs)
            # 计算损失
            loss =  alpha * distill_loss(output_student,output_teacher) +  (1-alpha)* ce_loss(output_student, labels)
            # 将梯度清零
            optimizer.zero_grad()
            # 将损失进行后向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()
            train_loss += loss.item()
        # 设置模型为验证模式
        student_model.eval()
        for step, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = student_model(imgs)
            pre_lab = torch.argmax(output, 1)
            corrects = (torch.sum(pre_lab == labels.data).double() / imgs.size(0))
            test_corrects += corrects.item()
        # 一个Epoch结束时，使用wandb保存需要可视化的数据
        experiment.log({
            'epoch': epoch,
            'train loss': train_loss / len(train_loader),
            'test acc': test_corrects / len(test_loader),
        })
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('{} Train Loss:{:.4f}'.format(epoch, train_loss / len(train_loader)))
        print('{} Test Acc:{:.4f}'.format(epoch, test_corrects / len(test_loader)))
        # 保存此Epoch训练的网络的参数
        torch.save(student_model.state_dict(), './stu_net.pth')




if __name__ == "__main__":
    # 定义训练使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义教师模型
    teacher_net = teacher_resnet18
    # 定义学生模型
    student_net = LeNet5()
    # 教师模型加载提前训练好的权重参数
    teacher_net.load_state_dict(torch.load('./resnet18.pth'), strict = True)
    # 蒸馏过程中损失函数的平衡因子
    alpha = 0.5
    # 蒸馏过程中的温度系数
    temperature = 2.0
    # 开始训练
    train_KD(teacher_net,student_net, alpha, temperature, trainloader, testloader, device, l_r=0.0003, num_epochs=50)
