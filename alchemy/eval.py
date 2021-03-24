import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm

from core.config import get_config
from core.evaluation import calc_accuracy, get_final_preds, AverageMeter, evaluate
from utils.misc import Timer, modelAnalyse
from utils.transforms import flip_back
from utils.factory import getDataset, getModel


if __name__ == '__main__':
    timer = Timer()
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

    net = getModel(cfg, is_test=True).to(device)

    # 验证
    net.eval()
    res = []

    predictions = torch.Tensor(len(test_set), test_set.num_joints, 2)
    with torch.no_grad():
        idx = 0
        for i, (input, target, meta) in enumerate(tqdm(test_loader)):

            input, target = input.to(device), target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            output = net(input)
            if cfg.TEST.FLIP_TEST:
                flip_input = torch.from_numpy(np.flip(input.cpu().numpy(), 3).copy()).to(device)
                flip_output = net(flip_input).cpu().numpy()
                flip_output = flip_back(flip_output, dataset=cfg.DATASET.NAME)
                # 翻转：得到的热图向右平移一个像素，能提高准确率
                flip_output[:, :, :, 1:] = flip_output.copy()[:, :, :, 0:-1]
                output = (output + torch.from_numpy(flip_output.copy()).to(device)) * 0.5

            # 预测坐标
            preds = get_final_preds(output.cpu(), meta['center'].numpy(), meta['scale'], meta['trans_v'], [64, 64])

            predictions[idx:idx + input.size(0), :, :] = preds
            idx += input.size(0)

            if cfg.DEBUG_ON:
                tmp = {'index': meta['index'].cpu().numpy(),
                       'trans_v': meta['trans_v'].cpu().numpy(),
                       'scale': meta['scale'].cpu().numpy(),
                       'center': meta['center'].cpu().numpy(),
                       'input': input.cpu().numpy(),
                       'output': output.cpu().numpy(),
                       'target': target.cpu().numpy(),
                       'target_weight': target_weight.cpu().numpy(),
                       'joints': meta['joints'].cpu().numpy(),
                       'joints_p': meta['joints_p'].cpu().numpy(),
                       'joints_vis': meta['joints_vis'].cpu().numpy()}

                for j in range(tmp['input'].shape[0]):
                    res.append(copy.deepcopy({'index': tmp['index'][j].item(),
                                              'trans_v': tmp['trans_v'][j],
                                              'scale': tmp['scale'][j],
                                              'center': tmp['center'][j],
                                              'input': tmp['input'][j],
                                              'output': tmp['output'][j],
                                              'target': tmp['target'][j],
                                              'target_weight': tmp['target_weight'][j],
                                              'joints': tmp['joints'][j],
                                              'joints_p': tmp['joints_p'][j],
                                              'joints_vis': tmp['joints_vis'][j]}))

    if cfg.DEBUG_ON:
        with open(cfg.WORK_DIR + 'meta-info.json', 'wb') as f:
            pickle.dump({'res': res}, f)
        with open(cfg.WORK_DIR + 'predictions.json', 'wb') as f:
            pickle.dump({'predictions': predictions.cpu().numpy()}, f)
        print('saved results info WORK_DIR: meta-info.json / predictions.json!')

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


