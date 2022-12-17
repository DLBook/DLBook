import torch
import os
import numpy as np
import torch.nn as nn
from torchvision.ops import RoIPool
from torchvision.models import resnet50 as ResNet50
from torchvision.ops import nms
import torch.nn.functional as F
import torch.optim as optim
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

""" 
数据预处理 
"""

# Pascal VOC数据及分类，背景按照分类0处理，类别数量加1
classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 读取解析PASCAL VOC2007数据集
class FRCNNDataset(Dataset):
    def __init__(self, voc_root, size=None):
        self.root = voc_root
        # 图像文件目录
        self.img_root = os.path.join(self.root, "JPEGImages")
        # xml文件目录
        self.xmlfilepath = os.path.join(self.root, "Annotations")
        # 获得xml文件地址列表
        temp_xml = os.listdir(self.xmlfilepath)
        xml_list = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                xml_list.append(os.path.join(self.xmlfilepath, xml))
        self.xml_list = xml_list
        self.size = size

    def __len__(self):
        return len(self.xml_list)

    # 获取数据
    def __getitem__(self, idx):
        xml = self.xml_list[idx]
        img_id, bbox, label = self.convet_xml(xml)
        img_path = os.path.join(self.img_root, img_id)
        image = Image.open(img_path)
        bbox = np.array(bbox, dtype=np.float32)
        label = np.array(label, dtype=np.int64)
        image, bbox = self.mytransform(image, bbox)
        target = {
            "boxes": bbox,
            "labels": label
        }
        return image, target

    # 图像变换
    def mytransform(self, image, bbox):
        # 缩放图像大小
        img_w, img_h = image.size
        w, h = self.size
        scale = min(w / img_w, h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        image = image.resize((new_w, new_h))
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_w / img_w
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_h / img_h
        # 统一图像尺寸
        new_img = Image.new("RGB", (w, h), "black")
        # 中心放置
        dx = (w - new_w) // 2
        dy = (h - new_h) // 2
        new_img.paste(image, (dx, dy))
        bbox[:, [0, 2]] = bbox[:, [0, 2]] + dx
        bbox[:, [1, 3]] = bbox[:, [1, 3]] + dy
        img = np.array(new_img, dtype=np.float32)
        img = np.transpose(img / 255.0, [2, 0, 1])
        return img, bbox

    # 解析xml文件
    def convet_xml(self, xml):
        in_file = open(xml)
        # 读取xml文件
        tree = ET.parse(in_file)
        # 获取根节点
        root = tree.getroot()
        # 获取图像名称
        img_id = root.find("filename").text
        # 获取该图像所有目标的位置和类别
        bboxes = []
        labels = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            label = classes.index(cls)
            xmlbox = obj.find('bndbox')
            # bbox: xmin, ymin, xmax, ymax
            bbox = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                    float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
            labels.append(label)
            bboxes.append(bbox)
        # 返回该图像的名称，所有目标的位置和类别
        return img_id, bboxes, labels



# 用于生成可用的batch
def frcnn_dataset_collate(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = np.array(images)
    images = torch.from_numpy(images).type(torch.FloatTensor)
    return images, targets

''' 
工具函数 
'''

# 计算smooth_l1损失
def _smooth_l1_loss(x, t, sigma):
    sigma_squared = sigma ** 2
    regression_diff = (x - t)
    regression_diff = regression_diff.abs()
    regression_loss = torch.where(
        regression_diff < (1. / sigma_squared),
        0.5 * sigma_squared * regression_diff ** 2,
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()

# 计算回归损失
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    pred_loc = pred_loc[gt_label > 0]
    gt_loc = gt_loc[gt_label > 0]
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, sigma)
    num_pos = (gt_label > 0).sum().float()
    loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
    return loc_loss

# 将回归参数转化为目标框位置参数
def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)
    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h
    return dst_bbox

# 将目标框位置参数转化为回归参数
def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)
    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

# 计算两个方框位置的IOU
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

# 用于生成先验框（Anchor）的类
class AnchorsGenerator():
    def __init__(self, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2]):
        super(AnchorsGenerator, self).__init__()
        self.anchor_scales = anchor_scales # 先验框尺度
        self.ratios = ratios # 先验框宽高比
        self.num_anchors = len(anchor_scales) * len(ratios) # 基础先验框个数
    # 生成基础先验框
    def generate_anchor_base(self, base_size=[16, 16], ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = base_size[0] * anchor_scales[j] * np.sqrt(ratios[i])
                w = base_size[1] * anchor_scales[j] * np.sqrt(1. / ratios[i])
                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = - h / 2.
                anchor_base[index, 1] = - w / 2.
                anchor_base[index, 2] = h / 2.
                anchor_base[index, 3] = w / 2.
        return anchor_base.round()
    # 生成一张图片上的全部先验框
    # anchor_base：基础先验框，feat_stride：先验框中心点间的长度，features_size：特征图大小
    def enumerate_shifted_anchor(self, anchor_base, feat_stride, features_size):
        height, width = features_size[0], features_size[1]
        # 计算网格中心点
        shift_x = np.arange(0, width * feat_stride[1], feat_stride[1])
        shift_y = np.arange(0, height * feat_stride[0], feat_stride[0])
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                          shift_x.ravel(), shift_y.ravel(),), axis=1)
        # 每个网格点上9个先验框
        A = anchor_base.shape[0]
        K = shift.shape[0]
        anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
        # 得到所有的先验框
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor.round()
    # 得到并返回所有的先验框；
    # image_size：输入模型的图像大小，features_size：图像经过模型后生成的特征图大小
    def __call__(self, image_size, features_size):
        # 获得步长stride
        stride = [int(round(image_size[i] / features_size[i])) for i in range(2)]
        anchor_base = self.generate_anchor_base(stride, self.ratios, self.anchor_scales)
        anchors = self.enumerate_shifted_anchor(np.array(anchor_base), stride, features_size)
        return anchors

# 用于生成建议框的类
class ProposalCreator():
    def __init__(self, nms_thresh=0.7,
                 n_train_pre_nms=12000, n_train_post_nms=600,
                 n_test_pre_nms=3000, n_test_post_nms=300, min_size=16):
        self.nms_thresh = nms_thresh # 非极大抑制的阈值
        # RPN中在nms处理前保留的proposal数
        self.n_train_pre_nms = n_train_pre_nms
        self.n_test_pre_nms = n_test_pre_nms
        # RPN中在nms处理后保留的proposal数
        self.n_train_post_nms = n_train_post_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size # proposal框的最小宽高值
    # loc：RPN预测回归值, score：RPN预测得分值, anchor：先验框, img_size：图像尺寸
    def __call__(self, loc, score, anchor, img_size, scale=1., training=True):
        if training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()
        # 将RPN网络预测结果转化成建议框
        roi = loc2bbox(anchor, loc)
        # 防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])
        # 建议框的宽高值不可以小于min_size
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # 对RPN预测得分排序
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        # 根据排序结果取出建议框及对应的得分
        roi = roi[order, :]
        score = score[order]
        # 对建议框进行非极大抑制
        keep = nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

# 判断Anchor内是否有目标
class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    # 计算anchor与bbox的IOU
    def _calc_ious(self, anchor, bbox):
        # 计算各anchor和各bbox的iou，shape为[anchor数, bbox数]
        ious = bbox_iou(anchor, bbox)
        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # 获得每个先验框最对应的真实框
        argmax_ious = ious.argmax(axis=1)
        # 找出每个先验框最对应的真实框的iou
        max_ious = np.max(ious, axis=1)
        # 保证每个真实框都存在对应的先验框
        gt_argmax_ious = ious.argmax(axis=0)
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious

    # 根据Anchor内是否含有目标为Anchor赋予标签，1为正样本，0为负样本，-1表示忽略
    def _create_label(self, anchor, bbox):
        # 初始化标签
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)
        # 计算anchor与bbox的IOU，并返回每个先验框最对应的真实框，及其对应的IOU
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        # IOU小于阈值则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0
        # IOU大于阈值则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1
        # 每个真实框至少有一个对应先验框，保证充分利用目标训练
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1
        # 判断正样本数量是否大于128，如果大于则限制在128
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return argmax_ious, label

# RPN类
class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, anchor_generator=None,
            proposal_generator=None,anchor_target_creator=None):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.proposal_layer = proposal_generator
        self.rpn_sigma = 1.0
        self.anchor_target_creator = anchor_target_creator
        # 滑动窗口数，即基础先验框数
        n_anchor = anchor_generator.num_anchors
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类预测先验框内部是否包含物体
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 回归预测对先验框进行调整
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

    def forward(self, features, img_size, targets=None, scale=1.):
        n, _, h, w = features.shape
        features_size = [h, w]
        features = F.relu(self.conv1(features))
        # 回归预测对先验框进行调整
        rpn_locs = self.loc(features)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # 分类预测先验框内部是否包含物体
        rpn_scores = self.score(features)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        # 得到先验框包含物体的概率
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # 生成先验框
        anchor = self.anchor_generator(img_size, features_size)
        # 使用List保存一个batch的各图片的建议框
        proposals = []
        # 遍历各个图片
        for i in range(n):
            # 得到各图片的建议框
            proposal = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size,
                                           scale=scale, training=self.training)
            proposals.append(proposal)
        # 训练网络时，计算RPN损失
        proposal_losses = {}
        if self.training:
            # 计算损失时，确保targets不为空
            assert targets is not None
            rpn_loc_loss_all = 0
            rpn_cls_loss_all = 0
            # 遍历各个图片，计算每张图片上的RPN损失
            for i in range(n):
                bbox = targets[i]["boxes"]
                rpn_loc = rpn_locs[i]
                rpn_score = rpn_scores[i]
                # 获得建议框网络应有的预测结果，并给每个先验框都打上标签
                gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
                gt_rpn_loc = torch.Tensor(gt_rpn_loc)
                gt_rpn_label = torch.Tensor(gt_rpn_label).long()
                if rpn_loc.is_cuda:
                    gt_rpn_loc = gt_rpn_loc.cuda()
                    gt_rpn_label = gt_rpn_label.cuda()
                # 计算建议框网络的回归损失
                rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
                # 计算建议框网络的分类损失
                rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
                rpn_cls_loss_all += rpn_cls_loss
                rpn_loc_loss_all += rpn_loc_loss
            proposal_losses = {
                "rpn_regression_loss": rpn_loc_loss_all / n,
                "rpn_classifier_loss": rpn_cls_loss_all / n
            }
        # 返回建议框和损失，测试时损失值为None
        return proposals, proposal_losses

# 判断proposal建议框内是否有目标，并对建议框采样
class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low
    # roi：预测值，bbox：真实目标框，label：真实标签
    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # 计算建议框和真实框的重合程度，
        iou = bbox_iou(roi, bbox)
        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # 获得每一个建议框最对应的真实框  [num_roi, ]
            gt_assignment = iou.argmax(axis=1)
            # 获得每一个建议框最对应的真实框的iou  [num_roi, ]
            max_iou = iou.max(axis=1)
            # 得到真实框的标签
            gt_roi_label = label[gt_assignment]
        # 满足建议框和真实框重合程度大于pos_iou_thresh的作为正样本
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 正样本数量限制在self.pos_roi_per_image以内
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
        # 满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        # 正样本的数量和负样本的数量的总和固定成self.n_sample
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
        keep_index = np.append(pos_index, neg_index)
        # 最终返回的预测值采样结果
        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
        # 预测值的回归参数
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))
        # 预测值的类别标签
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        # 返回：预测值采样结果，回归参数，类别标签
        return sample_roi, gt_roi_loc, gt_roi_label

# ROI Head类
class RoIHead(nn.Module):
    def __init__(self, std, mean, n_class, roi_size, spatial_scale, classifier, proposal_target_creator):
        super(RoIHead, self).__init__()
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        self.classifier = classifier
        # 对ROIPooling后的的结果进行回归预测
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # 对ROIPooling后的的结果进行分类
        self.score = nn.Linear(2048, n_class)
        self.num_classes = n_class
        self.std = std
        self.mean = mean
        self.roi_sigma = 1
        self.nms_iou = 0.3
        self.score_thresh = 0.5
        self.proposal_target_creator = proposal_target_creator
    # 将建议框解码成预测框
    def decoder(self, roi_cls_locs, roi_scores, rois, height, width, nms_iou, score_thresh):
        mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes)[None]
        std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes)[None]
        roi_cls_loc = (roi_cls_locs * std + mean)
        roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])
        # 利用classifier网络的预测结果对建议框进行调整获得预测框
        roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
        cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
        # 防止预测框超出图片范围
        cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]).clamp(min=0, max=width)
        cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]).clamp(min=0, max=height)
        prob = F.softmax(roi_scores, dim=-1)
        class_conf, class_pred = torch.max(prob, dim=-1)
        # 利用置信度进行第一轮筛选
        conf_mask = (class_conf >= score_thresh)
        # 根据置信度进行预测结果的筛选
        cls_bbox = cls_bbox[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        output = []
        for l in range(1, self.num_classes):
            arg_mask = class_pred == l
            # 取出对应的框和置信度
            cls_bbox_l = cls_bbox[arg_mask, l, :]
            class_conf_l = class_conf[arg_mask]
            if len(class_conf_l) == 0:
                continue
            # 将预测框、分类、置信度参数拼接
            detections_class = torch.cat(
                [cls_bbox_l, torch.unsqueeze(class_pred[arg_mask], -1).float(), torch.unsqueeze(class_conf_l, -1)], -1)
            # 非极大抑制筛选
            keep = nms(detections_class[:, :4], detections_class[:, -1], nms_iou)
            output.extend(detections_class[keep].cpu().numpy())
            outputs = np.array(output)
        return outputs

    def forward(self, features, proposals, img_size, targets=None):
        n = features.size(0)
        if self.training:
            sample_proposals = []
            gt_roi_locs = []
            gt_roi_labels = []
            for i in range(n):
                bbox = targets[i]["boxes"]
                label = targets[i]["labels"]
                roi = proposals[i]
                sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label,
                                                                                    self.mean, self.std)
                sample_roi = torch.Tensor(sample_roi)
                gt_roi_loc = torch.Tensor(gt_roi_loc)
                gt_roi_label = torch.Tensor(gt_roi_label).long()
                if features.is_cuda:
                    sample_roi = sample_roi.cuda()
                    gt_roi_loc = gt_roi_loc.cuda()
                    gt_roi_label = gt_roi_label.cuda()
                sample_proposals.append(sample_roi)
                gt_roi_locs.append(gt_roi_loc)
                gt_roi_labels.append(gt_roi_label)
            proposals = sample_proposals
        class_logits = []
        box_regression = []
        for i in range(n):
            x = torch.unsqueeze(features[i], dim=0)
            rois = proposals[i]
            roi_indices = torch.zeros(len(rois))
            if x.is_cuda:
                roi_indices = roi_indices.cuda()
                rois = rois.cuda()
            rois_feature_map = torch.zeros_like(rois)
            rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size(3)  # [3]
            rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size(2)  # [2]
            indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
            # 利用建议框对公用特征层进行截取
            pool = self.roi(x, indices_and_rois)
            # 利用classifier网络进行特征提取
            fc7 = self.classifier(pool)
            # 特征展平
            fc7 = fc7.view(fc7.size(0), -1)
            # 回归预测
            roi_cls_locs = self.cls_loc(fc7)
            # 分类预测
            roi_scores = self.score(fc7)
            roi_cls_locs = roi_cls_locs.view(x.size(0), -1, roi_cls_locs.size(1))
            roi_scores = roi_scores.view(x.size(0), -1, roi_scores.size(1))
            class_logits.append(roi_scores)
            box_regression.append(roi_cls_locs)
        roi_loss = {}
        outputs = []
        if self.training:
            # 训练时计算ROIHead损失
            loss_classifier = 0
            loss_regression = 0
            for i in range(n):
                n_sample = box_regression[i].size(1)
                roi_cls_loc = box_regression[i].view(n_sample, -1, 4)
                roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_labels[i]]
                # 分别计算Classifier网络的回归损失和分类损失
                roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_locs[i], gt_roi_labels[i].data, self.roi_sigma)
                roi_cls_loss = nn.CrossEntropyLoss()(class_logits[i][0], gt_roi_labels[i])
                loss_classifier += roi_cls_loss
                loss_regression += roi_loc_loss
            roi_loss = {
                "roi_regression_loss": loss_regression / n,
                "roi_classifier_loss": loss_classifier / n
            }
        else:
            # 测试时计算最终的预测框坐标
            for i in range(n):
                output = self.decoder(box_regression[i][0], class_logits[i][0], proposals[i],
                                      img_size[0], img_size[1], self.nms_iou, self.score_thresh)
                outputs.append(output)
        return outputs, roi_loss

# 特征提取网络
def resnet50():
    model = ResNet50()
    # 获取特征提取部分
    features = list([model.conv1, model.bn1, model.relu,
                     model.maxpool, model.layer1, model.layer2, model.layer3])
    # 分类部分
    classifier = list([model.layer4, model.avgpool])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


# Faster_RCNN
class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2]):
        super(FasterRCNN, self).__init__()
        self.extractor, classifier = resnet50()
        anchor_generator = AnchorsGenerator(anchor_scales=anchor_scales, ratios=ratios)
        proposal_generator = ProposalCreator()
        anchor_target_creator = AnchorTargetCreator()
        self.rpn = RegionProposalNetwork(
            1024, 512,
            anchor_generator=anchor_generator,
            proposal_generator=proposal_generator,
            anchor_target_creator=anchor_target_creator
        )
        proposal_target_creator = ProposalTargetCreator()
        self.head = RoIHead(
            n_class=num_classes + 1,
            roi_size=14,
            spatial_scale=1,
            std=[0.1, 0.1, 0.2, 0.2],
            mean=[0, 0, 0, 0],
            classifier=classifier,
            proposal_target_creator=proposal_target_creator
        )

    def forward(self, images, scale=1., targets=None):
        img_size = images.shape[2:]
        # 将图像输入backbone得到特征图
        features = self.extractor(images)
        # 将特征层以及标注target信息传入rpn中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(features, img_size, targets, scale)
        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses = self.head(features, proposals, img_size, targets)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections
    # 冻结bn层
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

# 训练方法
def train(model, start_epoch, final_epoch, gen, genval, lr, device):
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    for epoch in range(start_epoch, final_epoch):
        total_loss = 0
        val_toal_loss = 0
        for iteration, batch in enumerate(gen):
            imgs, targets = batch[0], batch[1]
            imgs = imgs.to(device)
            output = model(imgs, scale=1.0, targets=targets)
            rpn_loc, rpn_cls, roi_loc, roi_cls = output["rpn_regression_loss"], \
                                                 output["rpn_classifier_loss"], \
                                                 output["roi_regression_loss"], \
                                                 output["roi_classifier_loss"]
            total = rpn_loc + rpn_cls + roi_loc + roi_cls
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += total.item()
        for iteration, batch in enumerate(genval):
            imgs, targets = batch[0], batch[1]
            with torch.no_grad():
                imgs = imgs.to(device)
                output = model(imgs, scale=1.0, targets=targets)
                rpn_loc, rpn_cls, roi_loc, roi_cls = output["rpn_regression_loss"], \
                                                     output["rpn_classifier_loss"], \
                                                     output["roi_regression_loss"], \
                                                     output["roi_classifier_loss"]
                val_total = rpn_loc + rpn_cls + roi_loc + roi_cls
                val_toal_loss += val_total.item()
        print('Epoch:' + str(epoch + 1) + '/' + str(final_epoch))
        print('Total Loss: %.4f Val total Loss: %.4f' % (total_loss/len(gen), val_toal_loss/len(genval)))
        torch.save(model.state_dict(), './logs/model.pth')
        lr_scheduler.step()

if __name__ == "__main__":
    # 训练所需要区分的类的个数
    NUM_CLASSES = 20
    model = FasterRCNN(NUM_CLASSES)
    model_path = './voc_weights_resnet.pth'
    print('Loading weights into state dict...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    dataset = FRCNNDataset(voc_root="./VOC2007", size=(600, 600))
    # 将数据分成训练/测试分区
    test_rate = 0.2
    num_val = int(len(dataset) * test_rate)
    num_train = len(dataset) - num_val
    # 随机将数据集分割成给定长度的不重叠的训练集和测试集
    train_data, test_data = random_split(dataset, [num_train, num_val],
                                         generator=torch.Generator().manual_seed(0))
    Batch_size = 2
    gen = DataLoader(
        train_data, shuffle=True,
        batch_size=Batch_size, pin_memory=True,
        drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(
        test_data, shuffle=False,
        batch_size=Batch_size, pin_memory=True,
        drop_last=True, collate_fn=frcnn_dataset_collate
    )
    # 冻结一定部分训练
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        for param in model.extractor.parameters():
            param.requires_grad = False
        model.freeze_bn()
        train(model, Init_Epoch, Freeze_Epoch, gen, gen_val, lr, device)
    # 解冻后训练
    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        for param in model.extractor.parameters():
            param.requires_grad = True
        model.freeze_bn()
        train(model, Freeze_Epoch, Unfreeze_Epoch, gen, gen_val, lr, device)



# import colorsys
# import copy
# import math
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image, ImageDraw, ImageFont
# from torch.nn import functional as F
#
# classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
#            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
#            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
# # 改变图像大小
# def get_new_img_size(width, height, img_min_side=600):
#     if width <= height:
#         f = float(img_min_side) / width
#         resized_height = int(f * height)
#         resized_width = int(img_min_side)
#     else:
#         f = float(img_min_side) / height
#         resized_width = int(f * width)
#         resized_height = int(img_min_side)
#     return resized_width, resized_height
#
# # 图像检测类
# class FRCNN(object):
#     #   初始化faster RCNN
#     def __init__(self, model_path, classes, iou, confidence, cuda):
#         self.class_names = classes
#         self.cuda = cuda
#         self.model_path = model_path
#         # 计算总的类的数量
#         self.num_classes = len(self.class_names) - 1
#         # 载入模型与权值
#         self.model = FasterRCNN(self.num_classes).eval()
#         self.model.head.nms_iou = iou
#         self.model.head.score_thresh = confidence
#         print('Loading weights into state dict...')
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         state_dict = torch.load(self.model_path, map_location=device)
#         self.model.load_state_dict(state_dict)
#         self.mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes + 1)[None]
#         self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
#         if self.cuda:
#             self.model = self.model.cuda()
#             self.mean = self.mean.cuda()
#             self.std = self.std.cuda()
#         print('{} model, anchors, and classes loaded.'.format(self.model_path))
#         # 画框设置不同的颜色
#         hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
#         self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#         self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
#
#     # 检测图像
#     def detect_image(self, image):
#         #  转换成RGB图片，可以用于灰度图预测
#         image = image.convert("RGB")
#         image_shape = np.array(np.shape(image)[0:2])
#         old_width, old_height = image_shape[1], image_shape[0]
#         old_image = copy.deepcopy(image)
#         # 给原图像进行resize，resize到短边为600的大小上
#         width, height = get_new_img_size(old_width, old_height)
#         image = image.resize([width, height], Image.BICUBIC)
#         # 图片预处理，归一化
#         photo = np.transpose(np.array(image, dtype=np.float32) / 255, (2, 0, 1))
#         with torch.no_grad():
#             images = torch.from_numpy(np.asarray([photo]))
#             if self.cuda:
#                 images = images.cuda()
#                 outputs = self.model(images)
#             # 如果没有检测出物体，返回原图
#             if len(outputs) == 0:
#                 return old_image
#             outputs = outputs[0]
#             outputs = np.array(outputs)
#             bbox = outputs[:, :4]
#             label = outputs[:, 4]
#             conf = outputs[:, 5]
#             bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
#             bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
#         font = ImageFont.truetype(font='model_data/simhei.ttf',
#                                   size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
#         thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2, 1)
#         image = old_image
#         for i, c in enumerate(label):
#             predicted_class = self.class_names[int(c)]
#             score = conf[i]
#             left, top, right, bottom = bbox[i]
#             top = top - 5
#             left = left - 5
#             bottom = bottom + 5
#             right = right + 5
#             top = max(0, np.floor(top + 0.5).astype('int32'))
#             left = max(0, np.floor(left + 0.5).astype('int32'))
#             bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
#             right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
#             # 画画框
#             label = '{} {:.2f}'.format(predicted_class, score)
#             draw = ImageDraw.Draw(image)
#             label_size = draw.textsize(label, font)
#             label = label.encode('utf-8')
#             print(label, top, left, bottom, right)
#             if top - label_size[1] >= 0:
#                 text_origin = np.array([left, top - label_size[1]])
#             else:
#                 text_origin = np.array([left, top + 1])
#             for i in range(thickness):
#                 draw.rectangle([left + i, top + i, right - i, bottom - i],
#                     outline=self.colors[int(c)])
#             draw.rectangle(
#                 [tuple(text_origin), tuple(text_origin + label_size)],
#                 fill=self.colors[int(c)])
#             draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
#             del draw
#         return image
#
# # 开始检测单张图片
# frcnn = FRCNN(model_path='model_data/model.pth', classes=classes, iou=0.3, confidence=0.5, cuda=True)
# img = input('Input image filename:')
# image = Image.open(img)
# r_image = frcnn.detect_image(image)
# r_image.show()
