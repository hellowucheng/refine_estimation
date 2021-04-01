import os
import torch
import dsntnn

from core.config import get_config
from core.functions import validate
from core.evaluation import calc_accuracy, AverageMeter
from utils.visual import Visualizer, TbWriter
from utils.misc import Timer, Recorder
from utils.factory import getDataset, getModel, getOptim, getScheduler, getCriterion


if __name__ == '__main__':
    cfg, timer = get_config(), Timer()
    recorder = Recorder(os.path.join(cfg.CHECKPOINTS_PATH, cfg.MODEL.NAME + '_Progress.pickle'), cfg.CHECKPOINTS_PATH)
    debug_writer, train_writer, valid_writer = TbWriter(cfg.OUTPUT_DIR + 'debug'), TbWriter(cfg.OUTPUT_DIR + 'train'), TbWriter(cfg.OUTPUT_DIR + 'valid')

    # cuda 相关
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.CUDNN.ENABLED else 'cpu')
    if cfg.CUDNN.ENABLED and torch.cuda.is_available():
        print("Use Cuda.")
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    train_set = getDataset(cfg, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    valid_set = getDataset(cfg, is_train=False)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    net = getModel(cfg).to(device)

    optimizer = getOptim(cfg, net)

    lr_scheduler = getScheduler(cfg, optimizer)

    criterion = getCriterion(cfg)

    # 统计指标
    acces = AverageMeter()
    losses = AverageMeter()
    
    print('start training')
    for epoch in range(cfg.TRAIN.LAST_EPOCH + 1, cfg.TRAIN.NUM_EPOCH):
        
        # 训练一个epoch
        print('----------------------------------------')
        print('第{%d}个epoch的学习率为:' % epoch, optimizer.state_dict()['param_groups'][0]['lr'])
        print('----------------------------------------\n')

        timer.tick('per epoch')

        # 对于BN 和 Dropout有影响, 因为这二者在train和test过程中处理不同
        net.train()
        acces.reset(), losses.reset()

        for i, (input, target, meta) in enumerate(train_loader):
            input, target = input.to(device), target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            coords, output = net(input)
            # loss = criterion(output, target, target_weight, meta['joints_vis'].to(device))
            # print(coords.shape, output.shape, meta['joints_vis'].shape, type(meta['joints_vis']))
            # print(meta['joints_vis'].view(input.size(0), -1, 1, 1).shape)
            # 截断点不计算loss
            coords = coords * meta['joints_vis'].view(input.size(0), -1, 1).to(device)
            output = output * meta['joints_vis'].view(input.size(0), -1, 1, 1).to(device)
            # print(meta['joints_vis'].shape)
            # print(coords.shape)
            # print(output.shape)
            # euc_loss = dsntnn.euclidean_losses(coords, meta['joints_p'].to(device) // 4)
            reg_loss = dsntnn.js_reg_losses(output, meta['joints_p'].to(device) // 4, sigma_t=1.0)
            # loss = dsntnn.average_loss(euc_loss + reg_loss)
            loss = dsntnn.average_loss(reg_loss)

            acc = calc_accuracy(output, target)

            losses.add(loss.item(), input.size(0))
            acces.add(acc[0][0], input.size(0))

            # 清空上一个batch的梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            # debug打点
            if i != 0 and i % cfg.DEBUG_STEPS == 0:
                print("epoch {%d}, debug_step={%d}, loss={%.6f}, acc={%.6f}" % (epoch, i, losses.mean(), acces.mean()))
                debug_writer.add_scalar(losses.mean(), tag='debug_loss')
                # visualizer.plot(y=losses.mean(), win='training progress', name='debug_loss')

        train_loss, train_accuracy = losses.mean(), acces.mean() * 100.0

        # 验证
        valid_loss, valid_accuracy = validate(cfg, net, valid_set, valid_loader, criterion, device)

        # 更新学习率
        lr_scheduler.step() if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else lr_scheduler.step(valid_accuracy)

        # 调试信息及可视化
        print('----------------------------------------')
        print('Epoch: %d, cost {%.2f} seconds' % (epoch, timer.tock('per epoch')))
        print('Train_Loss:    {%.4f}  | Train_Accuracy:    {%.4f}' % (train_loss, train_accuracy))
        print('Validate_Loss: {%.4f}  | Validate_Accuracy: {%.4f}' % (valid_loss, valid_accuracy))

        # visualizer.plots({'train_loss': train_loss, 'valid_loss': valid_loss}, win='loss')
        # visualizer.plots({'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy}, win='accuracy')
        train_writer.add_scalar(train_loss, 'loss')
        valid_writer.add_scalar(valid_loss, 'loss')
        train_writer.add_scalar(train_accuracy, 'accuracy')
        valid_writer.add_scalar(valid_accuracy, 'accuracy')

        recorder.checkpoint(net, valid_accuracy, '%s-Epoch-{%d}-Accuracy-{%.4f}.pth' % (net.name, epoch, valid_accuracy), mode='max')
        recorder.records({'train_loss': train_loss,
                          'train_accuracy': train_accuracy,
                          'valid_loss': valid_loss,
                          'valid_accuracy': valid_accuracy})
    debug_writer.close()
    train_writer.close()
    valid_writer.close()
