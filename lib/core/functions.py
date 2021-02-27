import torch
import numpy as np

from utils.transforms import flip_back
from core.evaluation import AverageMeter, calc_accuracy, get_final_preds, evaluate


def validate(cfg, net, val_set, val_loader, criterion, device):
    # 验证
    net.eval()
    losses = AverageMeter()
    # acces = [AverageMeter() for _ in range(val_set.num_joints + 1)]
    predictions = torch.Tensor(len(val_set), val_set.num_joints, 2)

    with torch.no_grad():
        idx = 0
        for i, (input, target, meta) in enumerate(val_loader):
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

            loss = criterion(output, target, target_weight)
            losses.add(loss.item(), input.size(0))

            # pckhs = calc_accuracy(output, target)
            # for j in range(len(acces)):
            #     acces[j].add(pckhs[j][0], pckhs[j][1])

            # 预测坐标
            preds = get_final_preds(output.cpu(), meta['center'].numpy(), meta['scale'], meta['trans_v'], [64, 64])
            predictions[idx:idx + input.size(0), :, :] = preds
            idx += input.size(0)

    print('----------------------------------------------------------------------------------------------------')
    name_value = evaluate(cfg, predictions)
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
    return losses.mean(), name_value['Mean']