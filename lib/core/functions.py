import torch
import dsntnn
import numpy as np

from utils.transforms import flip_back
from core.evaluation import AverageMeter, get_final_preds, evaluate


def validate(cfg, net, val_set, val_loader, criterion, device):
    # 验证
    net.eval()
    losses = AverageMeter()
    predictions = torch.Tensor(len(val_set), val_set.num_joints, 2)

    with torch.no_grad():
        idx = 0
        for i, (input, target, meta) in enumerate(val_loader):
            input, target = input.to(device), target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            coords, output = net(input)

            # if cfg.TEST.FLIP_TEST:
            #     flip_input = torch.from_numpy(np.flip(input.cpu().numpy(), 3).copy()).to(device)
            #     coords, flip_output = net(flip_input)
            #     flip_output = flip_back(flip_output, dataset=cfg.DATASET.NAME).cpu().numpy()
            #     # 翻转：得到的热图向右平移一个像素，能提高准确率
            #     flip_output[:, :, :, 1:] = flip_output.copy()[:, :, :, 0:-1]
            #     output = (output + torch.from_numpy(flip_output.copy()).to(device)) * 0.5

            # loss = criterion(output, target, target_weight, meta['joints_vis'].to(device))
            coords = coords * meta['joints_vis'].view(input.size(0), -1, 1).to(device)
            output = output * meta['joints_vis'].view(input.size(0), -1, 1, 1).to(device)
            # euc_loss = dsntnn.euclidean_losses(coords, meta['joints_p'].to(device) // 4)
            reg_loss = dsntnn.js_reg_losses(output, meta['joints_p'].to(device) // 4, sigma_t=1.0)
            # loss = dsntnn.average_loss(euc_loss + reg_loss)
            loss = dsntnn.average_loss(reg_loss)
            losses.add(loss.item(), input.size(0))

            # 预测坐标
            preds = get_final_preds(output.cpu(), meta['center'].numpy(), meta['scale'], meta['trans_v'], [64, 64])
            predictions[idx:idx + input.size(0), :, :] = preds
            idx += input.size(0)

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