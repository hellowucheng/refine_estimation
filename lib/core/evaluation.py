import os
import math
import torch
import numpy as np
from scipy.io import loadmat
from collections import OrderedDict
from utils.transforms import get_affine_transform, affine_transform


# 从输出概率热图获取结果
def get_preds(scores):
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    # B x C x H x W -> B x C x HW, 取每张热图的最大概率值及所在索引位置
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    # maxval: B x C -> B x C x 1
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    # idx: B x C -> B x C x 1
    idx = idx.view(scores.size(0), scores.size(1), 1)

    # idx: B x C x 1 -> B x C x 2
    preds = idx.repeat(1, 1, 2).float()

    # 计算得到x坐标
    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3)
    # 计算得到y坐标
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3))

    # 只有概率高于0的坐标才输出
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    # preds: B x C x 2; target: B x C x 2
    preds = preds.float()
    target = target.float()
    # dists: C x B
    dists = torch.zeros(preds.size(1), preds.size(0))
    # preds: B x C x 2;
    for b in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[b, c, 0] > 1 and target[b, c, 1] > 1:
                dists[c, b] = torch.dist(preds[b, c, :], target[b, c, :]) / normalize[b]
            else:
                dists[c, b] = -1
    return dists


# pckh0.5
def calc_dist_acc(dist, threshold=0.5):
    # dist == -1 即该关节点不在真值热图中或者是被遮挡的关节点, 不参与评估
    dist = dist[dist != -1]
    # dist: B,
    if len(dist) > 0:
        # 对于某个关节点，有B个输入，统计它的预测值和真值距离小于threshold的数量占总数的比例，得到这个关节点的pck
        return 1.0 * (dist < threshold).sum().item() / len(dist), len(dist)
    else:
        return 0, 0


# output: B x C x H x W; target: B x C x 64 x 64
def calc_accuracy(output, target):
    idx = list(range(output.shape[1]))

    # 得到预测的坐标: B x C x 2
    preds = get_preds(output)
    # 获取真值坐标: B x C x 2
    gts = get_preds(target)
    # B x (h / 10)
    norm = torch.ones(preds.size(0)) * output.size(3) / 10
    # 计算预测点和真值规范化L2距离: C x B
    dists = calc_dists(preds, gts, norm)

    # 平均准确率 以及 len(idx)个关节点的准确率
    # acc = torch.zeros(1 + len(idx))
    acces = [(0, 0) for _ in range(1 + len(idx))]

    avg = 0
    cnt = 0

    for i in range(len(idx)):
        acces[i + 1] = calc_dist_acc(dists[idx[i]])
        cnt += acces[i + 1][1]
        avg += acces[i + 1][0] * acces[i + 1][1]

    if cnt != 0:
        acces[0] = avg / cnt, cnt
    return acces


# 根据网络输出热图-B x C x H x W 计算原图尺寸的坐标
def get_final_preds(output, center, scale, trans_v, resolution):
    # 获取热图峰值坐标 B x C x 2
    coords = get_preds(output)
    # 后处理(看不懂-先放着)
    for b in range(coords.size(0)):
        for c in range(coords.size(1)):
            # 对于张热图
            heatmap = output[b][c]
            px = int(math.floor(coords[b][c][0] + 0.5))
            py = int(math.floor(coords[b][c][1] + 0.5))
            if 1 < px < resolution[0] - 1 and 1 < py < resolution[1] - 1:
                diff = torch.Tensor([heatmap[py][px + 1] - heatmap[py][px - 1], heatmap[py + 1][px]-heatmap[py - 1][px]])
                # sign指示函数
                coords[b][c] += diff.sign() * 0.25

    preds = coords.clone()

    # 转换为原图尺寸坐标
    for i in range(coords.size(0)):
        # _, inv_trans = get_affine_transform(center[i], scale[i][0] * 200.0, scale[i][1] * 200, 0, resolution[0], resolution[1])
        for j in range(coords.size(1)):
            preds[i][j] = torch.from_numpy(affine_transform(coords[i][j] * 4, trans_v[i]))
    if preds.dim() < 3:
        preds = preds.unsqueeze(0)
    return preds


def evaluate(cfg, preds):
    preds += 1

    SC_BIAS = 0.6
    threshold = 0.5

    gt_file = os.path.join(cfg.DATASET.ROOT, 'annot', 'gt_valid_plus.mat')
    gt_dict = loadmat(gt_file)
    # 存放关节名字的ndarray, 所在索引也是该关节点的编号
    dataset_joints = gt_dict['dataset_joints']

    # 得到可见关节点标注 (16, 2958)
    if cfg.TEST.DATA_MODE == 1:
        jnt_visible = np.array(gt_dict['joints_vis'] == 1, dtype=np.uint8)
    elif cfg.TEST.DATA_MODE == 2:
        jnt_visible = np.array(gt_dict['joints_vis'] == 2, dtype=np.uint8)
    elif cfg.TEST.DATA_MODE == 3:
        jnt_visible = np.array(gt_dict['joints_vis'] > 0, dtype=np.uint8)
    else:
        raise RuntimeError('Eval Model Not Defined! : %d' % cfg.TEST.EVAL_MODE)

    # 关节点真值坐标, (16, 2, 2958)
    pos_gt_src = gt_dict['pos_gt_src']
    # 头部检测框 (2, 2, 2958), 左上、右下坐标
    headboxes_src = gt_dict['headboxes_src']
    # 网络预测坐标 2958 x 16 x 2 -> 16 x 2 x 2958
    pos_pred_src = np.transpose(preds, [1, 2, 0])

    '''
    得到关节名对应的关节点编号 (dataset_joints-(1, 16)
     -> 得到行坐标0, 列坐标即编号 故取[1]), 得到为ndarray, 取[0]即关节点编号)
    取13个关节点统计精度
    '''
    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    # 网络预测坐标和真值坐标的误差矩阵 (16, 2, 2958)
    uv_error = pos_pred_src - pos_gt_src
    # 计算每个实例的每一个关节点的预测坐标 和 真值坐标的 l2距离 sqrt{(x2-x1)**2 + (y2-y1)**2} -> (16, 1, 2958) -> (16, 2958)
    # (16, 2958)
    uv_err = np.linalg.norm(uv_error, axis=1)
    # (2, 2, 2958)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    #  -> 头部大小 (2958, )
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS

    # headsizes-(2958, ) 堆叠 -> scale-(16, 2958) 效果等同于 np.repeat(headsizes[None, :], 16, axis=0)
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    # 预测和真值的l2距离与头部大小的比值
    scaled_uv_err = np.divide(uv_err, scale)

    # # (可见点掩码) 这一行没啥必要呀
    # scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)

    # 对于每个关节点统计整个验证集中有多少个可见点
    jnt_count = np.sum(jnt_visible, axis=1)
    # 所有的关节点个数(不统计骨盆和胸部)
    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True

    # 防止除0# 仅评估遮挡点的话, 头部和颈部这两个关节点没有啥遮挡
    jnt_count = np.clip(jnt_count, 1, None)

    # 比值是否小于阈值，小于则预测成功, (可见点掩码, 之前乘了掩码, 遮挡点值变为0,小于threshold,会被误算, 所以再次乘掩码)
    # (16, 2958)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # pck随阈值threshold变化曲线
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # 掩码矩阵
    PCKh = np.ma.array(PCKh, mask=False)
    # mask[6:8]=True, 即表示最后统计时不计算所以6:8的值, 该位置对应的关节点为骨盆和胸部
    PCKh.mask[6:8] = True

    # 每个关节点占所有关节点个数的比例
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = [
        ('Head', PCKh[head]),
        ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        ('Mean', np.sum(PCKh * jnt_ratio)),
        ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio)),
        ('neck', PCKh[8]),
        ('lsho', PCKh[lsho]),
        ('rsho', PCKh[rsho]),
        ('lelb', PCKh[lelb]),
        ('relb', PCKh[relb]),
        ('lwri', PCKh[lwri]),
        ('rwri', PCKh[rwri]),
        ('lhip', PCKh[lhip]),
        ('rhip', PCKh[rhip]),
        ('lkne', PCKh[lkne]),
        ('rkne', PCKh[rkne]),
        ('lank', PCKh[lank]),
        ('rank', PCKh[rank]),
    ]
    name_value = OrderedDict(name_value)
    return name_value


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg


