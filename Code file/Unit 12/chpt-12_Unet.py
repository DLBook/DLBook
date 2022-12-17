'''
参考代码：https://github.com/qiaofengsheng/pytorch-UNet.git

'''

# 导入所需要的包
import torch
import wandb
import torch.nn as nn
from os import listdir
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch import Tensor
from torch import optim
from os.path import splitext
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

class MyDataset(Dataset):
    """定义封装数据的类"""
    # 初始化
    def __init__(self, images_dir, masks_dir, size = (256, 256)):
        # 图像保持尺寸
        self.size = size
        # 图像集地址
        self.images_dir = Path(images_dir)
        # 标签集地址
        self.masks_dir = Path(masks_dir)
        # 以列表存储标签标签集中各标签图像的名字，不包含后缀
        self.names = [splitext(file)[0] for file in listdir(masks_dir) if not file.startswith('.')]

    # 返回names长度，即标签图像的个数
    def __len__(self):
        return len(self.names)

    # 图像预处理, pil_img: image型数据, size: 图像保持尺寸
    # 返回处理后的图像tensor数据
    @staticmethod
    def mytransforms(pil_img, size):
        # 找出图像的最长边
        max_len = max(pil_img.size)
        # 构建一个(max_len, max_len)大小的RGB背景图片
        new_img = Image.new('RGB', (max_len, max_len), "black")
        # 将图像粘贴到背景图片上
        new_img.paste(pil_img, (0, 0))
        # 将形成的新图片变成固定大小
        new_img = new_img.resize(size)
        return tensor_img

    # 在类中定义__getitem__()方法，那么类的实例对象P就能以P[key]的形式取值
    def __getitem__(self, index):
        # 取值时，通过索引找到对应的图像和标签地址
        name = self.names[index]
        mask_dir = list(self.masks_dir.glob(name + '.*'))
        img_dir = list(self.images_dir.glob(name + '.*'))
        # 加载图像
        image = Image.open(img_dir[0])
        mask = Image.open(mask_dir[0])
        # 预处理,得到tensor数据
        image = self.mytransforms(image, size=self.size)
        mask = self.mytransforms(mask, size=self.size)
        return {
            "image": image,
            "mask": mask
        }

class DoubleConv(nn.Module):
    """定义卷积层部分，包含双层卷积"""
    # 初始化
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """先用最大池化下采样，然后再跟双层卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    """上采样，然后跟双层卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样
        x1 = self.up(x1)
        # 进行通道拼接
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """定义Unet网络"""
    # 初始化
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        # 下采样过程
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)
        # 上采样过程
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        # 输出层，输出结果
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sig(logits)

# 定义二分类的Dice系数
def dice_coeff(predict: Tensor, target: Tensor, epsilon=1e-6):
    # 获取每个批次的大小 N
    N = target.size(0)
    # 将特征矩阵变换为变换为特征向量
    predict = predict.view(N, -1)
    targets = target.view(N, -1)
    # 计算交集(分子)
    intersection = predict * targets
    # 计算和(分母)
    set_sum = predict.sum(1) + targets.sum(1)
    # 计算同一批次的所有图片的Dice系数
    dice_eff_N = (2 * intersection.sum(1) + epsilon) / (set_sum + epsilon)
    # 返回该批次Dice系数的平均值
    return dice_eff_N.sum() / N

# 定义多分类的Dice系数
def multiclass_dice_coeff(predict: Tensor, target: Tensor, epsilon=1e-6):
    dice = 0
    # 对每个通道(即每个分类)求Dice系数，并求平均
    for channel in range(predict.shape[1]):
        dice += dice_coeff(predict[:, channel, :, :], target[:, channel, :, :], epsilon)
    return dice / predict.shape[1]

# 根据Dice系数得Dice Loss
def dice_loss(predict: Tensor, target: Tensor, multiclass: bool = True):
    if multiclass:
        dice = multiclass_dice_coeff(predict, target)
    else:
        dice = dice_coeff(predict, target)
    return 1 - dice

# 使用的损失组合
def use_loss(masks_pred, masks_true):
    # 计算交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    loss1 = ce_loss(masks_pred, masks_true)
    # 计算Dice损失
    loss2 = dice_loss(masks_pred, masks_true)
    return loss1 + loss2

# 定义训练方法
def train(net, trainloader, valloader,
        device,# 设置使用的设备
        epochs: int = 5,# 循环训练次数
        learning_rate: float = 1e-5,# 学习率
        amp: bool = False# 是否使用自动混合精度,提高运行效率
        ):
    # 使用wandb工具跟踪并可视化训练过程
    config = dict(epochs=epochs, batch_size=2, learning_rate=learning_rate,
                  img_scale=0.5,amp=amp)
    experiment = wandb.init(project='U-Net', config=config, resume='allow', anonymous='must')
    # 设置优化器，使用RMSprop优化器
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # torch.cuda.amp自动混合精度训练 —— 节省显存并加快推理速度
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    n_train = len(trainloader)
    # 开始训练
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in trainloader:
                images = batch['image']
                masks_true = batch['mask']
                images = images.to(device)
                masks_true = masks_true.to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # 计算损失，损失由交叉熵损失和Dice损失组成
                    loss = use_loss(masks_pred, masks_true)
                # set_to_none=True 将梯度设置为 None 产生适度的加速
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # 可视化训练信息
                experiment.log({
                    'train loss': loss.item(),
                    'learning rate': optimizer.param_groups[0]['lr'],
                    # 取同一批次的一张图片进行可视化
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(masks_true[0].float().cpu()),
                        'pred': wandb.Image(masks_pred[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        # 每个epoch的训练结束后，使用验证方法得图像分割得分，验证方法接下来进行定义
        val_score = evaluate(net, valloader, device)
        print(f"Validation Score is:{val_score}")
        # 每个epoch的训练结束后，保存训练的网络权值
        torch.save(net.state_dict(), './checkpoints/checkpoint_epoch{}.pth'.format(epoch))

# 定义验证方法
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # 遍历验证集
    with tqdm(total=num_val_batches, desc='Validation round', unit='batch') as pbar:
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
            # 将图像和标签放到指定设备上
            image = image.to(device)
            mask_true = mask_true.to(device)
            with torch.no_grad():
                # 得到预测值
                mask_pred = net(image)
                # 计算预测得分
                dice_score += dice_loss(mask_pred, mask_true)
            pbar.update()
    return dice_score / num_val_batches

if __name__ == "__main__":
    # 准备训练数据, 创建数据集
    dir_img = Path('./VOC2007/JPEGImages/')
    dir_mask = Path('./VOC2007/SegmentationClass/')
    size = (128, 128)
    dataset = MyDataset(dir_img, dir_mask, size)
    # 将数据分成训练/验证分区
    val_rate = 0.1
    n_val = int(len(dataset) * val_rate)
    n_train = len(dataset) - n_val
    # 随机将数据集分割成给定长度的不重叠的训练集和测试集
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # 生成数据加载器
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=16,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=16,
        pin_memory=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_channels=3, n_classes=3)
    net.to(device=device)
    train(net=net, trainloader = train_loader, valloader=val_loader, epochs=1, learning_rate=1e-5,
        					device=device, amp=False)
