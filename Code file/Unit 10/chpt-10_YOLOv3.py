'''
引用资料：
https://github.com/bubbliiiing/yolo3-pytorch
'''

import cv2
import torch
import math
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from random import shuffle
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# """ 数据预处理 """
# from os import getcwd
# import xml.etree.ElementTree as ET
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
#            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# def convert_annotation(year, image_id, list_file):
#     in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     for obj in root.iter('object'):
#         difficult = 0
#         if obj.find('difficult') != None:
#             difficult = obj.find('difficult').text
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult) == 1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
#              int(xmlbox.find('ymax').text))
#         list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#
# wd = getcwd()
# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# for year, image_set in sets:
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt' % (year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()

'''
训练阶段
'''
# 创建数据集
class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 随机的数据增强预处理
    def mytransforms(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=1.5, val=1.5):
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        image_data = np.array(image_data, dtype=np.float32)
        image_data = np.transpose(image_data / 255.0, (2, 0, 1))
        # 调整目标框坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []
        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        tmp_img, y = self.mytransforms(lines[index], self.image_size[0:2])
        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_img, tmp_targets
        # return np.adrray, np.adrray

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        tb = torch.from_numpy(box)
        bboxes.append(tb)
    images = np.array(images)
    images = torch.from_numpy(images)
    return images, bboxes
    # return Tensor, List[Tensor,...]

'''
Darknet
'''
# 重新构造卷积模块，包含了BatchNorm和LeakyReLU激活函数
def Convolutional2D(in_channel, out_channel, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.1)
    )

# 基本的darknet块
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Convolutional2D(in_channel, channels[0], kernel_size=1, stride=1, padding=0)
        self.conv2 = Convolutional2D(channels[0], channels[1], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

# Darknet网络
class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv = Convolutional2D(3, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])
        self.layers_out_filters = [64, 128, 256, 512, 1024]

    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("down_conv", Convolutional2D(self.inplanes, planes[1], 3, 2, 1)))
        # 加入darknet模块
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("ConvResidual{}".format(i), ResidualBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5

# 构造Darknet53网络
def darknet53(pretrained=None, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model

'''
YOLO-v3
'''
def Convolutional_Set(in_channel, channel_list):
    return nn.Sequential(OrderedDict([
        ('conv0',Convolutional2D(in_channel, channel_list[0], 1, 1, 0)),
        ('conv1', Convolutional2D(channel_list[0], channel_list[1], 3, 1, 1)),
        ('conv2', Convolutional2D(channel_list[1], channel_list[0], 1, 1, 0)),
        ('conv3', Convolutional2D(channel_list[0], channel_list[1], 3, 1, 1)),
        ('conv5', Convolutional2D(channel_list[1], channel_list[0], 1, 1, 0)),
    ]))

def Out_Conv(in_channel, hidden_channel, out_channel):
    return nn.Sequential(OrderedDict([
        ('conv3×3', Convolutional2D(in_channel, hidden_channel, 3, 1, 1)),
        ('conv1×1', nn.Conv2d(hidden_channel, out_channel, 1, 1, 0, bias=True))
    ]))

# 构造Yolov3模型
class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53()
        out_channels = self.backbone.layers_out_filters
        final_out_channel0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.conv_set0 = Convolutional_Set(out_channels[-1], [512, 1024])
        self.out_layer0 = Out_Conv(512, 1024, final_out_channel0)
        final_out_channel1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.out_layer1_conv = Convolutional2D(512, 256, 1, 1, 0)
        self.out_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_set1 = Convolutional_Set(out_channels[-2]+256, [256, 512])
        self.out_layer1 = Out_Conv(256, 512, final_out_channel1)
        final_out_channel2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.out_layer2_conv = Convolutional2D(256, 128, 1, 1, 0)
        self.out_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_set2 = Convolutional_Set(out_channels[-3] + 128, [128, 256])
        self.out_layer2 = Out_Conv(128, 256, final_out_channel2)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        #  yolov3 分支0
        x0 = self.conv_set0(x0)
        out0 = self.out_layer0(x0)
        #  yolov3 分支1
        x1_in = self.out_layer1_conv(x0)
        x1_in = self.out_layer1_upsample(x1_in)
        x1 = torch.cat([x1_in, x1], 1)
        x1 = self.conv_set1(x1)
        out1 = self.out_layer1(x1)
        #  yolov3 分支2
        x2_in = self.out_layer2_conv(x1)
        x2_in = self.out_layer2_upsample(x2_in)
        x2 = torch.cat([x2_in, x2], 1)
        x2 = self.conv_set2(x2)
        out2 = self.out_layer2(x2)
        return out0, out1, out2

# 计算两个目标框之间的IOU
def jaccard(_box_a, _box_b):
    # 变换数据格式
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    # 求IOU
    union = area_a + area_b - inter
    return inter / union

def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

# MSE损失
def MSELoss(pred, target):
    return (pred - target) ** 2

# BCE损失
def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

# 创建YoLo的损失类
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]
        self.img_size = img_size
        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        bs = input.size(0)
        # 特征层的高
        in_h = input.size(2)
        # 特征层的宽
        in_w = input.size(3)
        # 计算步长
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        # 把先验框的尺寸调整为特征层上对应的宽高
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, int(self.num_anchors / 3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # 对prediction预测的(x,y,w,h,confidence, class)进行调整
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # 找到哪些先验框内部包含物体
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = \
            self.get_target(targets, scaled_anchors, in_w, in_h)
        # 排除无用的预测
        noobj_mask = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)
        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        #  分别计算(x,y,w,h,confidence, class)的损失
        loss_x = torch.sum(BCELoss(x, tx) / bs * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) / bs * box_loss_scale * mask)
        loss_w = torch.sum(MSELoss(w, tw) / bs * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) / bs * 0.5 * box_loss_scale * mask)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask / bs)
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / bs)
        # 计算总损失
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
        return loss

    # 获得真实框的参数以及与真实框匹配的正样本
    def get_target(self, target, anchors, in_w, in_h):
        # 计算一共有多少张图片
        bs = len(target)
        # 获得先验框
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        # 创建全是0或全是1的阵列
        mask = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, self.num_classes, requires_grad=False)
        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            if len(target[b]) == 0:
                continue
            # 计算出在特征层上的点位
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h
            # 计算出属于哪个网格
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            # 计算真实框的大小
            gt_box = torch.FloatTensor(
                torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            # 计算出所有先验框的大小
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))
            # 计算重合程度
            anch_ious = jaccard(gt_box, anchor_shapes)
            # Find the best matching anchor box
            best_ns = torch.argmax(anch_ious, dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                # Masks
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
                # Masks
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    # 判定哪些先验框内部真实的存在物体
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    # 计算先验框中心调整参数
                    tx[b, best_n, gj, gi] = gx - gi.float()
                    ty[b, best_n, gj, gi] = gy - gj.float()
                    # 计算先验框宽高调整参数
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n + subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n + subtract_index][1])
                    # 用于获得xywh的比例
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    # 物体置信度
                    tconf[b, best_n, gj, gi] = 1
                    # 种类
                    tcls[b, best_n, gj, gi, int(target[b][i, 4])] = 1
                else:
                    # 操作超出图像范围
                    continue
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    # 忽略与真实框重合度不是最大，但与真实框重合度高于阈值的先验框
    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # 先验框中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)
        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask

# 定义训练方法
def train(net, yolo_losses, gen, genval, start_epoch, final_epoch, lr, device):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    net.train()
    for epoch in range(start_epoch, final_epoch):
        total_loss = 0
        val_loss = 0
        with tqdm(total=len(gen), desc=f'Epoch {epoch + 1}/{final_epoch}',
                                            postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                images, targets = batch[0], batch[1]
                images = images.to(device)
                outputs = net(images)
                optimizer.zero_grad()
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item)
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1)})
                pbar.update(1)
        print('Start Validation')
        with tqdm(total=len(genval), desc=f'Epoch {epoch + 1}/{final_epoch}',
                                            postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    images = images.to(device)
                    outputs = net(images)
                    optimizer.zero_grad()
                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets)
                        losses.append(loss_item)
                    loss = sum(losses)
                    val_loss += loss.item()
                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)
        lr_scheduler.step()
        print('Epoch:' + str(epoch + 1) + '/' + str(final_epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / len(gen), val_loss / len(gen_val)))
        torch.save(model.state_dict(), './model.pth')


if __name__ == "__main__":
    Config = {
        "yolo": {
            "anchors": [[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]],
            "classes": 20,
        },
        "img_h": 416,
        "img_w": 416,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        Cuda = True
    else:
        Cuda = False
    # 数据集准备；0.9用于训练，0.1用于验证
    val_split = 0.1
    annotation_path = '2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    Batch_size = 1
    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=False,
                     drop_last=True, shuffle=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=False,
                         drop_last=True, shuffle=False, collate_fn=yolo_dataset_collate)
    # 创建模型并初始化参数
    print('Loading weights into state dict...')
    model = YoloBody(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load("./coco_weights.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                            Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))
    if True:
        lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        #  冻结一定部分训练
        for param in model.backbone.parameters():
            param.requires_grad = False
        train(model, yolo_losses, gen, gen_val, Init_Epoch, Freeze_Epoch, lr, device)
    if True:
        lr = 1e-4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        # 解冻后训练
        for param in model.backbone.parameters():
            param.requires_grad = True
        train(model, yolo_losses, gen, gen_val, Freeze_Epoch, Unfreeze_Epoch, lr, device)

'''
测试阶段
'''
# import torch
# import torch.nn as nn
# import numpy as np
# import colorsys
# from torchvision.ops import nms
# from PIL import Image, ImageDraw, ImageFont
#
# # 解码器，将预测的偏移值转化为预测的坐标值
# class DecodeBox(nn.Module):
#     def __init__(self, anchors, num_classes, img_size):
#         super(DecodeBox, self).__init__()
#         self.anchors = anchors
#         self.num_anchors = len(anchors)
#         self.num_classes = num_classes
#         self.bbox_attrs = 5 + num_classes
#         self.img_size = img_size
#
#     def forward(self, input):
#         batch_size = input.size(0)
#         input_height = input.size(2)
#         input_width = input.size(3)
#         # 计算步长
#         stride_h = self.img_size[1] / input_height
#         stride_w = self.img_size[0] / input_width
#         # 归一到特征层上
#         scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
#                           for anchor_width, anchor_height in self.anchors]
#         # 对预测结果进行resize
#         prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs,
#                                 input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
#         # 先验框的中心位置的调整参数
#         x = torch.sigmoid(prediction[..., 0])
#         y = torch.sigmoid(prediction[..., 1])
#         # 先验框的宽高调整参数
#         w = prediction[..., 2]
#         h = prediction[..., 3]
#         # 获得置信度，是否有物体
#         conf = torch.sigmoid(prediction[..., 4])
#         # 种类置信度
#         pred_cls = torch.sigmoid(prediction[..., 5:])
#         FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
#         LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
#         # 生成网格，先验框中心，网格左上角
#         grid_x = torch.linspace(0, input_width-1, input_width).repeat(input_width, 1).repeat(
#             batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
#         grid_y = torch.linspace(0, input_height-1, input_height).repeat(input_height, 1).t().repeat(
#             batch_size*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
#         # 生成先验框的宽高
#         anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
#         anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
#         anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
#         anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
#         # 计算调整后的先验框中心与宽高
#         pred_boxes = FloatTensor(prediction[..., :4].shape)
#         pred_boxes[..., 0] = x.data + grid_x
#         pred_boxes[..., 1] = y.data + grid_y
#         pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
#         pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
#         # 用于将输出调整为相对于416x416的大小
#         _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
#         output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,conf.view(batch_size, -1, 1),
#                             pred_cls.view(batch_size, -1, self.num_classes)), -1)
#         return output.data
#
# # 变换图片，使之符合模型输入的尺寸
# def letterbox_image(image, size):
#     iw, ih = image.size
#     w, h = size
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#     image = image.resize((nw, nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128, 128, 128))
#     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
#     return new_image
#
# def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
#     new_shape = image_shape*np.min(input_shape/image_shape)
#     offset = (input_shape-new_shape)/2./input_shape
#     scale = input_shape/new_shape
#     box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
#     box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape
#     box_yx = (box_yx - offset) * scale
#     box_hw *= scale
#     box_mins = box_yx - (box_hw / 2.)
#     box_maxes = box_yx + (box_hw / 2.)
#     boxes =  np.concatenate([
#         box_mins[:, 0:1],
#         box_mins[:, 1:2],
#         box_maxes[:, 0:1],
#         box_maxes[:, 1:2]
#     ],axis=-1)
#     boxes *= np.concatenate([image_shape, image_shape],axis=-1)
#     return boxes
#
# # 非极大抑制获得最准确的预测
# def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
#     # 求左上角和右下角
#     box_corner = prediction.new(prediction.shape)
#     box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
#     box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
#     box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
#     box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
#     prediction[:, :, :4] = box_corner[:, :, :4]
#     output = [None for _ in range(len(prediction))]
#     for image_i, image_pred in enumerate(prediction):
#         # 获得种类及其置信度
#         class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
#         # 利用置信度进行第一轮筛选
#         conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
#         image_pred = image_pred[conf_mask]
#         class_conf = class_conf[conf_mask]
#         class_pred = class_pred[conf_mask]
#         if not image_pred.size(0):
#             continue
#         # 获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
#         detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
#         # 获得种类
#         unique_labels = detections[:, -1].cpu().unique()
#         if prediction.is_cuda:
#             unique_labels = unique_labels.cuda()
#             detections = detections.cuda()
#         for c in unique_labels:
#             # 获得某一类初步筛选后全部的预测结果
#             detections_class = detections[detections[:, -1] == c]
#             # 非极大抑制
#             keep = nms(
#                 detections_class[:, :4],
#                 detections_class[:, 4] * detections_class[:, 5],
#                 nms_thres
#             )
#             max_detections = detections_class[keep]
#             output[image_i] = max_detections
#     return output
#
# # YOLOv3检测的测试类
# class YOLO(object):
#     def __init__(self, model_path, classes, config, **kwargs):
#         self.config = config
#         self.path = model_path
#         self.class_names = classes
#         self.image_size = (416, 416, 3)
#         self.confidence = 0.5
#         self.iou = 0.3
#         self.cuda = torch.cuda.is_available()
#         self.generate()
#     # YOLOv3模型及解码器
#     def generate(self):
#         self.config["yolo"]["classes"] = len(self.class_names)
#         self.net = YoloBody(self.config)
#         # 加快模型训练的效率
#         print('Loading weights into state dict...')
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         state_dict = torch.load(self.path, map_location=device)
#         self.net.load_state_dict(state_dict)
#         print("Finish!")
#         self.net = self.net.eval()
#         if self.cuda:
#             self.net = self.net.cuda()
#         # 初始化解码器
#         self.yolo_decodes = []
#         for i in range(3):
#             self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i],
#                     self.config["yolo"]["classes"],(self.image_size[1], self.image_size[0])))
#         print('{} model, anchors, and classes loaded.'.format(self.path))
#         # 不同类别设置不同颜色的画框
#         hsv_tuples = [(x / len(self.class_names), 1., 1.)for x in range(len(self.class_names))]
#         self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#         self.colors = list(
#             map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),self.colors))
#
#     # 检测图片
#     def detect_image(self, image):
#         image_shape = np.array(np.shape(image)[0:2])
#         crop_img = np.array(letterbox_image(image, (self.image_size[0], self.image_size[1])))
#         photo = np.array(crop_img, dtype=np.float32)
#         photo /= 255.0
#         photo = np.transpose(photo, (2, 0, 1))
#         photo = photo.astype(np.float32)
#         images = []
#         images.append(photo)
#         images = np.asarray(images)
#         images = torch.from_numpy(images)
#         if self.cuda:
#             images = images.cuda()
#         with torch.no_grad():
#             outputs = self.net(images)
#             output_list = []
#             for i in range(3):
#                 output_list.append(self.yolo_decodes[i](outputs[i]))
#             output = torch.cat(output_list, 1)
#             batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
#                                                    conf_thres=self.confidence,
#                                                    nms_thres=self.iou)
#         try:
#             batch_detections = batch_detections[0].cpu().numpy()
#         except:
#             return image
#         top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
#         top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
#         top_label = np.array(batch_detections[top_index, -1], np.int32)
#         top_bboxes = np.array(batch_detections[top_index, :4])
#         top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(
#             top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), np.expand_dims(
#                 top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
#         # 去掉灰条
#         boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
#                                    np.array([self.image_size[0], self.image_size[1]]), image_shape)
#         # 设置字体
#         font = ImageFont.truetype(font='model_data/simhei.ttf',
#                                   size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
#         thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0]
#         for i, c in enumerate(top_label):
#             predicted_class = self.class_names[c]
#             score = top_conf[i]
#             top, left, bottom, right = boxes[i]
#             top = top - 5
#             left = left - 5
#             bottom = bottom + 5
#             right = right + 5
#             top = max(0, np.floor(top + 0.5).astype('int32'))
#             left = max(0, np.floor(left + 0.5).astype('int32'))
#             bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
#             right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
#             # 画框
#             label = '{} {:.2f}'.format(predicted_class, score)
#             draw = ImageDraw.Draw(image)
#             label_size = draw.textsize(label, font)
#             label = label.encode('utf-8')
#             print(label)
#             if top - label_size[1] >= 0:
#                 text_origin = np.array([left, top - label_size[1]])
#             else:
#                 text_origin = np.array([left, top + 1])
#             for i in range(thickness):
#                 draw.rectangle(
#                     [left + i, top + i, right - i, bottom - i],
#                     outline=self.colors[self.class_names.index(predicted_class)])
#             draw.rectangle(
#                 [tuple(text_origin), tuple(text_origin + label_size)],
#                 fill=self.colors[self.class_names.index(predicted_class)])
#             draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
#             del draw
#         return image
#
# # 初始化检测类
# yolo = YOLO(model_path="./model.pth", classes=classes, config=Config)
# img = input('Input image filename:')
# image = Image.open(img)
# # 开始检测
# r_image = yolo.detect_image(image)
# r_image.show()
