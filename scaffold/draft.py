import torch
import numpy as np
from tqdm import tqdm

from core.config import get_config
from core.evaluation import calc_accuracy, get_final_preds, AverageMeter, evaluate
from utils.transforms import flip_back
from utils.factory import getDataset, getModel


import matplotlib.pyplot as plt
def draw_pose(img, joints, joints_vis):
    fig, ax = plt.subplots()
    ax.imshow(img)

    skelenton = [[10, 11], [11, 12], [12, 8], [8, 13],
                 [13, 14], [14, 15], [8, 9], [7, 8], [2, 6],
                 [3, 6], [1, 2], [1, 0], [3, 4], [4, 5], [6, 7]]
    for sk in skelenton:
        j1, j2 = joints[sk[0]], joints[sk[1]]
        if j1[0] > 0 and j1[1] > 0 and j2[0] > 0 and j2[1] > 0:
            ax.plot([j1[0], j2[0]], [j1[1], j2[1]], color='violet')
    # 画关节点, 可见点-红色, 遮挡点-蓝色
    markersize = (np.min((np.max(joints, axis=0) - np.min(joints, axis=0)) / np.array(img.shape)[:2])) * 5 + 2
    for i, joint in enumerate(joints):
        color = 'r' if joints_vis[i] > 0.0 else 'b'
        if joint[0] > 0.0 and joint[1] > 0.0:
            ax.plot(joint[0], joint[1], marker='o', mfc=color, mec=color, markersize=markersize)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    cfg = get_config()

    # cuda 相关
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.CUDNN.ENABLED else 'cpu')
    if cfg.CUDNN.ENABLED and torch.cuda.is_available():
        print("Use Cuda.")
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    dataset = getDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=cfg.NUM_WORKERS)

    for i, (input, target, meta) in enumerate(tqdm(dataset)):
        draw_pose(meta['img'], meta['joints_pred'], meta['joints_vis'])
        # draw_pose(meta['img'], meta['joints_p'], meta['joints_vis'])
        break