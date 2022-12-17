"""
代码参考：https://arxiv.org/abs/1511.06434
         https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs
"""

# 导入所需要的包
import torch
import wandb
import torchvision
import torch.nn as nn
import torchvision.utils as utils

# 定义DCGAN的生成器
class DCGAN_G(nn.Module):
    # 初始化网络，noise_d：输入生成器的噪声的尺寸
    def __init__(self, noise_d):
        super(DCGAN_G, self).__init__()
        self.noise_d = noise_d
        # layer1输入的是一个随机噪声，大小为100x1x1，输出尺寸为1024x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(noise_d, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # 该输出尺寸为512x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # 该输出尺寸为256x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 该输出尺寸为128x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 该输出尺寸为1x64x64
        self.layerout = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    # 前向传播
    def forward(self, x):
        # 将噪声变为要求的尺寸
        x = x.view(-1, self.noise_d, 1, 1)
        # 输入进网络
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layerout(x)
        return out

# 定义DCGAN的判别器
class DCGAN_D(nn.Module):
    def __init__(self):
        super(DCGAN_D, self).__init__()
        # layer1 输入一幅图片，尺寸为1 x 64 x 64，输出尺寸为64x32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 该输出尺寸为128x16x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 该输出尺寸为256x8x8
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 该输出尺寸为512x4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出分类结果，尺寸为1x1x1
        self.layerout = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    # 定义NetD的前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layerout(x)
        return out


# 定义判别器损失；real_out：真实图像判别结果，fake_out：假图像判别结果
def discriminator_loss(real_out, fake_out, device):
    # 使用内置的BCE损失函数
    criterion = nn.BCELoss()
    # 真实图片的标签：1
    real_label = torch.ones(size=(real_out.size(0),)).to(device)
    # 计算判别真实图像的损失
    d_loss_real = criterion(real_out, real_label)
    # 假图片的标签：0
    fake_label = torch.zeros(size=(fake_out.size(0),)).to(device)
    # 计算判别假图像的损失
    d_loss_fake = criterion(fake_out, fake_label)
    # 返回两项损失之和
    d_loss = d_loss_real + d_loss_fake
    return d_loss

# 定义生成器损失
def generator_loss(fake_out, device):
    # 使用内置的BCE损失函数
    criterion = nn.BCELoss()
    # 假图片的期望标签：1
    fake_label = torch.ones(size=(fake_out.size(0),)).to(device)
    # 计算判别真实图像的损失
    g_loss = criterion(fake_out, fake_label)
    return g_loss

# 定义训练方法
def train(net_G, net_D, train_loader, device, epochs=5, l_r=0.0002):
    # # 使用wandb跟踪训练过程
    config = dict(epochs=epochs, learning_rate=l_r,)
    experiment = wandb.init(project='DCGAN', config=config, resume='allow', anonymous='must')
    # 设置优化器，生成器和判别器均使用Adam优化器
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=l_r, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=l_r, betas=(0.5, 0.999))
    # 训练过程
    for epoch in range(1, epochs + 1):
        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            # 随机生成噪声数据
            noise = torch.randn(imgs.size(0), net_G.noise_d)
            noise = noise.to(device)
            # 固定生成器G，训练鉴别器D，
            # 令D尽可能的把真图片判别为1，把假图片判别为0
            output_real = net_D(imgs)
            output_real = torch.squeeze(output_real)
            # 生成假图
            fake = net_G(noise)
            # 因为G不用更新，使用detach()避免梯度传到G
            output_fake = net_D(fake.detach())
            output_fake = torch.squeeze(output_fake)
            d_loss = discriminator_loss(output_real, output_fake, device)
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            # 固定判别器D，训练生成器G，令判别器D尽可能把G生成的假图判别为1
            # 生成假图
            fake = net_G(noise)
            # 将假图放入判别器
            output_fake = net_D(fake)
            output_fake = torch.squeeze(output_fake)
            g_loss = generator_loss(output_fake, device)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            experiment.log({
                'epoch': epoch,
                'step': step,
                'train generator_loss': g_loss.item(),
                'train discriminator_loss': d_loss.item(),
                'learning rate_g': optimizer_G.param_groups[0]['lr'],
                'learning rate_d': optimizer_D.param_groups[0]['lr'],
                # 将同一批次的生成图片进行可视化
                'fake_images':  wandb.Image(utils.make_grid(
                    fake[0:(fake.size(0) if fake.size(0)<40 else 40), ...]).float().cpu())
            })
            # 每隔50步或一个epoch结束时输出
            if step % 50 == 0 or step == len(train_loader)-1:
                print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                      % (epoch, epochs, step, len(train_loader)-1, d_loss.item(), g_loss.item()))
        # 一个epoch训练结束，保存模型
        torch.save(net_G.state_dict(), './netG.pth')
        torch.save(net_D.state_dict(), './netD.pth')


if __name__ == "__main__":
    # 定义使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #预处理与读取训练数据
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
        ])
    train_set = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=True,
        transform=transforms,
        download=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )
    netG = DCGAN_G(100).to(device)
    netD = DCGAN_D().to(device)
    train(netG, netD, trainloader, device, epochs=10, l_r=0.0002)
