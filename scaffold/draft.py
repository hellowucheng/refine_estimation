import copy
import torch
import pickle
import os
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

from core.config import get_config
from core.evaluation import calc_accuracy, AverageMeter
from utils.misc import Timer
from utils.transforms import flip_back
from utils.factory import getDataset, getModel, getCriterion


def evaluate(preds):
    # convert 0-based index to 1-based index
    preds = preds[:, :, 0:2] + 1.0

    # if output_dir:
    #     pred_file = os.path.join(output_dir, 'pred.mat')
    #     savemat(pred_file, mdict={'preds': preds})

    # if 'test' in cfg.DATASET.TEST_SET:
    #     return {'Null': 0.0}, 0.0

    SC_BIAS = 0.6
    threshold = 0.5

    gt_file = os.path.join('/data-tmp/Dataset/mpii',
                           'annot',
                           'gt_valid.mat')
    gt_dict = loadmat(gt_file)
    dataset_joints = gt_dict['dataset_joints']
    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']

    pos_pred_src = np.transpose(preds, [1, 2, 0])

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

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    # 计算预测 和 真值的 l2距离
    uv_err = np.linalg.norm(uv_error, axis=1)
    # 头部大小
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS

    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    # 预测和真值的l2距离与头部大小的比值
    scaled_uv_err = np.divide(uv_err, scale)
    # 这一行没啥必要呀
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    # 比值是否小于阈值，小于则预测成功
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5+0.01, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean', 'Mean@0.1'))
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('%.3f' % PCKh[head],
                                                                       '%.3f' % (0.5 * (PCKh[lsho] + PCKh[rsho])),
                                                                       '%.3f' % (0.5 * (PCKh[lelb] + PCKh[relb])),
                                                                       '%.3f' % (0.5 * (PCKh[lwri] + PCKh[rwri])),
                                                                       '%.3f' % (0.5 * (PCKh[lhip] + PCKh[rhip])),
                                                                       '%.3f' % (0.5 * (PCKh[lkne] + PCKh[rkne])),
                                                                       '%.3f' % (0.5 * (PCKh[lank] + PCKh[rank])),
                                                                       '%.3f' % np.sum(PCKh * jnt_ratio),
                                                                       '%.3f' % np.sum(pckAll[11, :] * jnt_ratio)))


if __name__ == '__main__':
    cfg = get_config()
    with open(cfg.WORK_DIR + 'predictions.json', 'rb') as f:
        preds = pickle.load(f)
    evaluate(preds['predictions'])