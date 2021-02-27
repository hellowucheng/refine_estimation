import os
import cv2
import json
import torch
import random
import numpy as np
from pathlib import Path
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transforms import get_affine_transform, affine_transform, generate_target


class Lspet(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        self.root = cfg.DATASET.ROOT
        self.num_joints = 16

        self.is_train = is_train
        self.use_augment = cfg.DATASET.TRAIN_AUGMENT.USE

        # 输入分辨率 input_resolution, 用于resize样本
        self.inp_res = cfg.MODEL.IMAGE_SIZE
        # 输出分辨率 output_resolution, 用于生成真值的高斯热图
        self.oup_res = cfg.MODEL.HEATMAP_SIZE
        # 真值高斯热图的标准差
        self.sigma = cfg.MODEL.SIGMA

        # 旋转角度-用于训练增强
        self.rot_factor = cfg.DATASET.TRAIN_AUGMENT.ROT_FACTOR
        # 缩放因子-用于训练增强
        self.scale_factor = cfg.DATASET.TRAIN_AUGMENT.SCALE_FACTOR

        # 训练验证集划分
        with open(cfg.DATASET.ROOT + 'LEEDS_annotations.json', 'r') as f:
            self.annots = json.load(f)
        self.train_set, self.valid_set = [], []
        for i, v in enumerate(self.annots):
            self.valid_set.append(i) if v['isValidation'] else self.train_set.append(i)

        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(
                                                                             mean=[0.4442, 0.4302, 0.4039],
                                                                             std=[0.2376, 0.2296, 0.2356])])

    def __getitem__(self, index):
        ant = self.annots[self.train_set[index]] if self.is_train else self.annots[self.valid_set[index]]

        img_path = os.path.join(self.root, 'images/', Path(ant['img_paths']).name)

        joints = np.array(ant['joint_self'])
        joints_p, joints_vis = joints[:, 0:2].copy(), joints[:, 2].copy()

        center = np.array(ant['objpos'])

        scale = ant['scale_provided']
        # 稍微放大实例框，防止肢体截断
        if center[0] != -1:
            scale = scale * 1.4375 * 200.0

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # 数据集提供的scale不准, 常出现偏大的情况, 使用center + joints重新修正scale
        w = np.extract(np.logical_or(joints_vis > 0, joints_p[:, 0] > 0), joints_p[:, 0])
        h = np.extract(np.logical_or(joints_vis > 0, joints_p[:, 1] > 0), joints_p[:, 1])
        l, r, u, b = min(w), max(w), min(h), max(h)
        scale_refine = max(center[0] - l, r - center[0], center[1] - u, b - center[1]) * 2 + 15

        # 遮挡点坐标有时候会出现标注错误的情况, 错误标注作为离群点使得scale_refine过大, 设置当scale_refine过大-超过原scale的1.5倍, 则放弃修正
        if scale_refine < 1.5 * scale:
            scale = scale_refine

        rot = 0
        if self.is_train and self.use_augment:
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor,
                                    1 + self.scale_factor)
            rot = np.clip(np.random.randn() * self.rot_factor, -2 * self.rot_factor,
                          2 * self.rot_factor) if random.random() <= 0.6 else 0

        # 截取实例区域
        trans = get_affine_transform(center, scale, rot, self.inp_res, self.inp_res, inverse=False)
        input = cv2.warpAffine(img, trans, (self.inp_res, self.inp_res), flags=cv2.INTER_LINEAR)

        input = self.transform(input)

        # 坐标真值随仿射变换修改
        for i in range(self.num_joints):
            # 该点无遮挡
            if joints_vis[i] > 0.0:
                joints_p[i] = affine_transform(joints_p[i], trans)

        target, target_weight = generate_target(joints, joints_vis, self.sigma, self.inp_res, self.oup_res)

        meta = {'index': index,
                'center': center, 'scale': scale,
                'joints': joints, 'joints_p': joints_p,
                'joints_vis': joints_vis, 'target_weight': torch.from_numpy(target_weight)}
        return input, torch.from_numpy(target), meta

    def __len__(self):
        return len(self.train_set) if self.is_train else len(self.valid_set)
