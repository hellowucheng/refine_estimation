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


joint_names = ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee',
               'left_ankle', 'pelvis', 'thorax', 'upper_neck', 'head_top', 'right_wrist',
               'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist']
joints_names_dict = {name: i + 1 for i, name in enumerate(joint_names)}

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

    # 统计指标
    # acces = [AverageMeter() for _ in range(test_set.num_joints + 1)]

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

            # pckhs = calc_accuracy(output, target)
            # for j in range(len(acces)):
            #     acces[j].add(pckhs[j][0], pckhs[j][1])

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

    # print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean'))
    # print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('%.4f' % acces[joints_names_dict['head_top']].mean(),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_shoulder']].mean() + acces[joints_names_dict['right_shoulder']].mean()) / 2),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_elbow']].mean() + acces[joints_names_dict['right_elbow']].mean()) / 2),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_wrist']].mean() + acces[joints_names_dict['right_wrist']].mean()) / 2),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_hip']].mean() + acces[joints_names_dict['right_hip']].mean()) / 2),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_knee']].mean() + acces[joints_names_dict['right_knee']].mean()) / 2),
    #                                                              '%.4f' % ((acces[joints_names_dict['left_ankle']].mean() + acces[joints_names_dict['right_ankle']].mean()) / 2),
    #                                                              '%.4f' % acces[0].mean()))

    # 评估
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

    # 参数量和计算量统计
    # modelAnalyse(net, (3, 256, 256), max_depth=5, mode='s')

