import os
import cv2
import json
import time
import torch
import random
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms

from utils.transforms import get_affine_transform, flip_joints, generate_target, perform_transform, \
    get_similarity_transform, mul_transform, jitter

symmetric_joints = {0:5, 5:0, 1:4, 4:1, 2:3, 3:2, 10:15, 15:10, 11:14, 14:11, 12:13, 13:12}


class AugMpii(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        self.root = cfg.DATASET.ROOT
        self.num_joints = 16

        self.keep_origin = cfg.DATASET.KEEP_ORIGIN

        self.is_train = is_train
        self.data_mode = cfg.TRAIN.DATA_MODE
        self.use_augment = cfg.DATASET.TRAIN_AUGMENT.USE

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
            anno_file = os.path.join(self.root, 'annot', 'train_augment.json')
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

            rot, flip = 0, False
            if self.is_train and self.use_augment:
                # 随机放缩
                sf = np.clip(1 + np.random.randn() * self.scale_factor, 0.8, 1 + self.scale_factor)
                box_width, box_height = box_width * sf, box_height * sf
                # 旋转
                rot = np.clip(np.random.randn() * self.rot_factor, -2 * self.rot_factor, 2 * self.rot_factor)
                # 水平翻转
                flip = self.flip and np.random.rand() >= 0.5

            # 保持比例放缩
            as_ratio = min(self.inp_res / box_height, self.inp_res / box_width)
            new_w, new_h = int(as_ratio * box_width), int(as_ratio * box_height)

            # 更新center 和 scale
            center, scale = np.array([(r + l) // 2, (b + u) // 2]), np.array([box_width / 200, box_height / 200], dtype=np.float)

            # 使用仿射变换实现裁剪 + 放缩
            trans, trans_v = get_affine_transform(center, scale[0] * 200.0, scale[1] * 200.0, 0, new_w, new_h,
                                                  is_train=self.is_train)
            input, joints_temp = perform_transform(img, joints_temp, trans, new_w, new_h)

            # 训练增强-旋转
            if rot != 0:
                # 使用欧式变换-(平移+旋转), 重新计算旋转后的宽、高, 防止截断
                rot_w = int(new_h * np.fabs(np.sin(np.radians(rot))) + new_w * np.fabs(np.cos(np.radians(rot))))
                rot_h = int(new_w * np.fabs(np.sin(np.radians(rot))) + new_h * np.fabs(np.cos(np.radians(rot))))

                augment_trans = get_similarity_transform((new_w // 2, new_h // 2), rot, (rot_w - new_w) // 2,
                                                         (rot_h - new_h) // 2)

                # 旋转后为保持原图不被裁剪放大了图像尺寸, 重新保持比例放缩到指定尺寸
                as_ratio = min(self.inp_res / rot_w, self.inp_res / rot_h)
                new_w, new_h = int(as_ratio * rot_w), int(as_ratio * rot_h)

                # 再次使用仿射变换裁剪
                crop_trans, _ = get_affine_transform(np.array([rot_w // 2, rot_h // 2]), rot_w, rot_h, 0, new_w, new_h,
                                                     is_train=self.is_train)

                # 复合变换, 提高效率
                trans = mul_transform(crop_trans, augment_trans)
                input, joints_temp = perform_transform(input, joints_temp, trans, new_w, new_h)

            # padding
            pw, ph = self.inp_res - new_w, self.inp_res - new_h
            pl, pr, pu, pb = pw // 2, pw - pw // 2, ph // 2, ph - ph // 2
            input = cv2.copyMakeBorder(input, pu, pb, pl, pr, cv2.BORDER_CONSTANT, None, (0, 0, 0))
            joints_temp += [pl, pu]
            # 用于测试
            if not self.is_train:
                trans_v = mul_transform(trans_v, np.array([[1., 0., -pl], [0., 1., -pu]]))

        joints_p, joints_pred = joints_temp[:self.num_joints].copy(), joints_temp[self.num_joints:].copy()

        # 合成训练数据
        if self.is_train:
            # 遍历所有关节点
            for i, joint in enumerate(joints_p):
                # 每个关节点取一种情况
                choice = np.zeros((4, 2))
                # 扰动
                # 90%的概率小偏移, 10%概率大偏移
                if i not in [8, 9] and np.random.randint(0, 10) < 1:
                    choice[0] = jitter(joint, (20, 100), (self.inp_res, self.inp_res))
                else:
                    choice[0] = jitter(joint, (0, 20), (self.inp_res, self.inp_res))
                # 丢失
                # 30%的概率关节点随机置零
                if i not in [8, 9] and np.random.randint(0, 10) < 3:
                    choice[1] = np.zeros(2)
                else:
                    choice[1] = jitter(joint, (0, 20), (self.inp_res, self.inp_res))
                # 错位
                # 10%的概率交换对称关节点位置
                if i in symmetric_joints.keys() and np.random.randint(0, 10) < 1:
                    choice[2] = joints_p[symmetric_joints[i]]
                else:
                    choice[2] = jitter(joint, (0, 20), (self.inp_res, self.inp_res))
                # 替换
                # 30%的概率和其他实例的同类、对称临界点交换位置
                # if len(ant['near_joints'][i]) > 0 and np.random.randint(0, 10) < 3:
                #     choice[3] = random.choice(ant['near_joints'][i])
                # else:
                #     choice[3] = jitter(joint, (0, 20), (self.inp_res, self.inp_res))
                joints_pred[i] = random.choice(choice)

        # 真值热图
        target, target_weight = generate_target(joints_p, joints_vis, self.sigma * 0.5, self.inp_res, self.oup_res)

        # 图像归一化, ToTensor处理
        input = self.transform(input.copy())

        # 为待修正的关节点生成热图用于输入
        err_heatmap, _ = generate_target(joints_pred, np.ones(16), self.sigma * 2, self.inp_res, self.inp_res)

        # 与可见概率加权
        # err_heatmap = err_heatmap * visible_prob[:, None, None]
        # 合并输入
        input = torch.cat((input, torch.from_numpy(err_heatmap).float()))

        meta = {'index': index,
                'trans_v': trans_v,
                'image': ant['image'],
                'center': center, 'scale': scale,
                'joints': joints, 'joints_p': joints_p, 'joints_pred': joints_pred,
                'joints_vis': np.array(ant['joints_vis']), 'target_weight': target_weight}
        return input, target, meta

    def __len__(self):
        return len(self.annots)
