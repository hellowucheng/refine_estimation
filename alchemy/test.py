import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from core.config import get_config

from core.evaluation import get_final_preds, evaluate
from utils.factory import getDataset


if __name__ == '__main__':
    cfg = get_config()

    # cuda 相关
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.CUDNN.ENABLED else 'cpu')
    if cfg.CUDNN.ENABLED and torch.cuda.is_available():
        print("Use Cuda.")
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    test_set = getDataset(cfg, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                              num_workers=cfg.NUM_WORKERS)

    with open('/data-tmp/Dataset/mpii/annot/valid_refine2.json') as f:
        annots = json.load(f)

    # 验证
    predictions = torch.Tensor(len(test_set), test_set.num_joints, 2)
    idx = 0
    for i, (input, target, meta) in enumerate(tqdm(test_loader)):
        batch_heatmap = torch.zeros(input.size(0), 16, 64, 64)
        batch_transv = np.zeros((input.size(0), 2, 3))
        for j, img_path in enumerate(meta['image']):
            heatmap_path = '/data-tmp/Dataset/mpii/occ_heatmaps/' + Path(img_path).stem + '-{%d}.occ' % meta['index'][j].item()
            with open(heatmap_path, 'rb') as f:
                heatmap = pickle.load(f)
            assert isinstance(heatmap, np.ndarray) and heatmap.shape == (16, 64, 64)
            batch_heatmap[j] = torch.from_numpy(heatmap)
            batch_transv[j] = annots[meta['index'][j]]['trans_v']

        # 预测坐标
        preds = get_final_preds(batch_heatmap.cpu(), meta['center'].numpy(), meta['scale'], batch_transv, [64, 64])

        predictions[idx:idx + input.size(0), :, :] = preds
        idx += input.size(0)

    # 评估PCKh
    name_value = evaluate(cfg, predictions)
    print('----------------------------------------------------------------------------------------------------')
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean', 'Mean@0.1'))
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('%.3f' % name_value['Head'],
                                                                       '%.3f' % name_value['Shoulder'],
                                                                       '%.3f' % name_value['Elbow'],
                                                                       '%.3f' % name_value['Wrist'],
                                                                       '%.3f' % name_value['Hip'],
                                                                       '%.3f' % name_value['Knee'],
                                                                       '%.3f' % name_value['Ankle'],
                                                                       '%.3f' % name_value['Mean'],
                                                                       '%.3f' % name_value['Mean@0.1']))
    print('----------------------------------------------------------------------------------------------------')

    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('head', 'neck', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank'))
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('%.3f' % name_value['Head'],
                                                                                                     '%.3f' % name_value['neck'],
                                                                                                     '%.3f' % name_value['lsho'],
                                                                                                     '%.3f' % name_value['rsho'],
                                                                                                     '%.3f' % name_value['lelb'],
                                                                                                     '%.3f' % name_value['relb'],
                                                                                                     '%.3f' % name_value['lwri'],
                                                                                                     '%.3f' % name_value['rwri'],
                                                                                                     '%.3f' % name_value['lhip'],
                                                                                                     '%.3f' % name_value['rhip'],
                                                                                                     '%.3f' % name_value['lkne'],
                                                                                                     '%.3f' % name_value['rkne'],
                                                                                                     '%.3f' % name_value['lank'],
                                                                                                     '%.3f' % name_value['rank']))
    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')

