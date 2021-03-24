import os
import cv2
import json
import torch
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms

from utils.transforms import get_affine_transform, flip_joints, generate_target, perform_transform, \
    get_similarity_transform, mul_transform


class AugMpii(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        self.root = cfg.DATASET.ROOT
        self.num_joints = 16

        self.keep_origin = cfg.DATASET.KEEP_ORIGIN

        self.is_train = is_train
        self.data_mode = cfg.TRAIN.DATA_MODE

        # 输入分辨率 input_resolution, 用于resize样本
        self.inp_res = cfg.MODEL.IMAGE_SIZE
        # 输出分辨率 output_resolution, 用于生成真值的高斯热图
        self.oup_res = cfg.MODEL.HEATMAP_SIZE
        # 真值高斯热图的标准差
        self.sigma = cfg.MODEL.SIGMA

        # 水平翻转-用于训练增强
        self.flip = cfg.DATASET.TRAIN_AUGMENT.FLIP
        # 旋转角度-用于训练增强
        self.rot_factor = cfg.DATASET.TRAIN_AUGMENT.ROT_FACTOR
        # 缩放因子-用于训练增强
        self.scale_factor = cfg.DATASET.TRAIN_AUGMENT.SCALE_FACTOR

        if self.is_train:
            anno_file = os.path.join(self.root, 'annot', 'train_pred.json')
        else:
            anno_file = os.path.join(self.root, 'annot', 'valid_pred.json')
        with open(anno_file) as f:
            self.annots = json.load(f)

        if self.keep_origin:
            self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                                                                             transforms.Normalize(
                                                                                 mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                                                                             transforms.Normalize(
                                                                                 mean=[0.4442, 0.4302, 0.4039],
                                                                                 std=[0.2376, 0.2296, 0.2356])])

    def __getitem__(self, index):
        ant = self.annots[index]

        img_path = os.path.join(self.root, 'images/', ant['image'])

        joints = np.array(ant['joints']) - 1
        joints_p = np.array(ant['joints']) - 1
        joints_pred = np.array(ant['joints_pred'])

        joints_temp = np.concatenate((joints_p, joints_pred))

        # 预测的可见概率
        visible_prob = np.array(ant['visible_prob'])

        # 关节点是否遮挡真值
        joints_vis = np.array(ant['joints_vis'])
        if self.data_mode == 1:
            joints_vis = np.array(joints_vis == 1, dtype=np.uint8)
        elif self.data_mode == 2:
            joints_vis = np.array(joints_vis == 2, dtype=np.uint8)
        elif self.data_mode == 3:
            joints_vis = np.array(joints_vis > 0, dtype=np.uint8)
        else:
            raise RuntimeError('Eval Model Not Defined! : %d' % self.data_mode)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if self.keep_origin:
            # center为实例框中心坐标, scale为缩放幅度, scale*200即实例框的宽和高, 注: 一个正方形大致圈出实例范围, 不是准确的检测框
            center, scale = np.array(ant['center'], dtype=np.float) - 1, np.array([ant['scale'], ant['scale']], dtype=np.float)

            # 稍微增大实例框, 防止实例截断
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                scale = scale * 1.25

            # 使用仿射变换实现放缩、平移和旋转
            trans, trans_v = get_affine_transform(center, scale[0] * 200.0, scale[1] * 200, 0, self.inp_res,
                                                  self.inp_res, self.is_train)
            input, joints_temp = perform_transform(img, joints_temp, trans, self.inp_res, self.inp_res)

        else:
            # 数据集提供的scale不准, 常出现偏大的情况, 使用center + joints重新修正scale
            w = np.extract(np.logical_or(joints_vis > 0, joints_p[:, 0] > 0), joints_p[:, 0])
            h = np.extract(np.logical_or(joints_vis > 0, joints_p[:, 1] > 0), joints_p[:, 1])

            l, r, u, b = min(w), max(w), min(h), max(h)
            box_width, box_height = min(int((r - l + 30) * 1.25), img.shape[1]), min(int((b - u + 30) * 1.25), img.shape[0])

            # 保持比例放缩
            as_ratio = min(self.inp_res / box_height, self.inp_res / box_width)
            new_w, new_h = int(as_ratio * box_width), int(as_ratio * box_height)

            # 更新center 和 scale
            center, scale = np.array([(r + l) // 2, (b + u) // 2]), np.array([box_width / 200, box_height / 200], dtype=np.float)

            # 使用仿射变换实现裁剪 + 放缩
            trans, trans_v = get_affine_transform(center, scale[0] * 200.0, scale[1] * 200.0, 0, new_w, new_h,
                                                  is_train=self.is_train)
            input, joints_temp = perform_transform(img, joints_temp, trans, new_w, new_h)

            # padding
            pw, ph = self.inp_res - new_w, self.inp_res - new_h
            pl, pr, pu, pb = pw // 2, pw - pw // 2, ph // 2, ph - ph // 2
            input = cv2.copyMakeBorder(input, pu, pb, pl, pr, cv2.BORDER_CONSTANT, None, (0, 0, 0))
            joints_temp += [pl, pu]
            # 用于测试
            if not self.is_train:
                trans_v = mul_transform(trans_v, np.array([[1., 0., -pl], [0., 1., -pu]]))

        joints_p, joints_pred = joints_temp[:self.num_joints].copy(), joints_temp[self.num_joints:].copy()

        # 真值热图
        target, target_weight = generate_target(joints_p, joints_vis, self.sigma, self.inp_res, self.oup_res)

        # 图像归一化, ToTensor处理
        input = self.transform(input.copy())
        # 为待修正的关节点生成热图用于输入
        err_heatmap, _ = generate_target(joints_pred, np.ones(16), self.sigma, self.inp_res, self.inp_res)
        # 与可见概率加权
        err_heatmap = err_heatmap * visible_prob[:, None, None]
        # 合并输入
        input = torch.cat((input, torch.from_numpy(err_heatmap).float()))

        meta = {'index': index,
                'trans_v': trans_v,
                'image': ant['image'],
                'center': center, 'scale': scale,
                'joints': joints, 'joints_p': joints_p,
                'joints_vis': np.array(ant['joints_vis']), 'target_weight': target_weight}
        return input, target, meta

    def __len__(self):
        return len(self.annots)
