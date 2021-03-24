import torch

from core.loss import JointsMSELoss, JointsFocalLoss, JointsMSEBalancedLoss

from models.pose_resnet import PoseResNet
from datasets.mpii import Mpii
from utils.misc import isExists


def getDataset(cfg, is_train=True):
    if is_train:
        print('| train dataset:', cfg.DATASET.NAME)
        print('  | data-mode:', cfg.TRAIN.DATA_MODE)
    else:
        print('| valid dataset:', cfg.DATASET.NAME)
        print('  | data-mode:', cfg.TEST.DATA_MODE)

    if cfg.DATASET.NAME.lower() == 'mpii':
        return Mpii(cfg, is_train)
    else:
        raise RuntimeError('Dataset Not Defined! : %s' % cfg.DATASET.NAME)


def getModel(cfg, is_test=False):
    print('| model:', cfg.MODEL.NAME)

    if cfg.MODEL.NAME.lower() == 'pose-resnet':
        model = PoseResNet(cfg)
    else:
        raise RuntimeError('Model Not Defined! : %s' % cfg.MODEL.NAME)

    if is_test:
        if isExists(cfg.TEST.TRAINED_MODEL):
            print('load trained model from:', cfg.TEST.TRAINED_MODEL)
            model.load_state_dict(torch.load(cfg.TEST.TRAINED_MODEL, map_location=lambda storage, loc: storage))
        else:
            print('do not have trained model!')
    else:
        model.load(resume=cfg.MODEL.RESUME, pretrained=cfg.MODEL.PRETRAINED)
    return model


def getOptim(cfg, net):
    print('| optimizer:', cfg.TRAIN.OPTIM)
    lr = cfg.TRAIN.LR
    for i in cfg.TRAIN.SCHEDULER.MILESTONES:
        if cfg.TRAIN.LAST_EPOCH >= i:
            lr = lr * cfg.TRAIN.SCHEDULER.GAMMA

    if cfg.TRAIN.OPTIM.lower() == 'adam':
        return torch.optim.Adam([{'params': net.parameters(), 'initial_lr': lr}], lr=lr,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIM.lower() == 'sgd':
        return torch.optim.SGD([{'params': net.parameters(), 'initial_lr': lr}], lr=lr,
                               momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)


def getScheduler(cfg, optimizer):
    print('| scheduler:', cfg.TRAIN.SCHEDULER.NAME)
    if cfg.TRAIN.SCHEDULER.NAME.lower() == 'multi-step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                    milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                    gamma=cfg.TRAIN.SCHEDULER.GAMMA,
                                                    last_epoch=cfg.TRAIN.LAST_EPOCH)

    elif cfg.TRAIN.SCHEDULER.NAME.lower() == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                          mode=cfg.TRAIN.SCHEDULER.MODE,
                                                          factor=cfg.TRAIN.SCHEDULER.FACTOR,
                                                          patience=cfg.TRAIN.SCHEDULER.PATIENCE,
                                                          verbose=cfg.TRAIN.SCHEDULER.VERBOSE,
                                                          threshold=cfg.TRAIN.SCHEDULER.THRESHOLD,
                                                          threshold_mode=cfg.TRAIN.SCHEDULER.THRESHOLD_MODE,
                                                          cooldown=cfg.TRAIN.SCHEDULER.COOLDOWN,
                                                          min_lr=cfg.TRAIN.SCHEDULER.MIN_LR,
                                                          )
    elif cfg.TRAIN.SCHEDULER.NAME.lower() == 'exponentiallr':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=cfg.TRAIN.SCHEDULER.GAMMA,
                                                      last_epoch=cfg.TRAIN.LAST_EPOCH)
    elif cfg.TRAIN.SCHEDULER.NAME.lower() == 'cosineannealinglr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                          T_max=cfg.TRAIN.SCHEDULER.T_MAX,
                                                          eta_min=cfg.TRAIN.SCHEDULER.ETA_MIN,
                                                          last_epoch=cfg.TRAIN.LAST_EPOCH)
    elif cfg.TRAIN.SCHEDULER.NAME.lower() == 'lambdalr':
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                 lr_lambda=lambda x: cfg.TRAIN.SCHEDULER.GAMMA ** sum(
                                                     [1 if x % cfg.TRAIN.SCHEDULER.T_MAX >= i else 0 for i in
                                                      cfg.TRAIN.SCHEDULER.MILESTONES]),
                                                 last_epoch=cfg.TRAIN.LAST_EPOCH)


def getCriterion(cfg):
    print('| criterion:', cfg.LOSS.NAME)
    if cfg.LOSS.NAME.lower() == 'jointsmseloss':
        print('  | use_target_weight:', cfg.LOSS.EXTRA.USE_TARGET_WEIGHT)
        return JointsMSELoss(use_target_weight=cfg.LOSS.EXTRA.USE_TARGET_WEIGHT)
    elif cfg.LOSS.NAME.lower() == 'jointsmsebalancedloss':
        print('  | balanced_alpha:', cfg.LOSS.EXTRA.BALANCED_ALPHA)
        return JointsMSEBalancedLoss(alpha=cfg.LOSS.EXTRA.BALANCED_ALPHA)
    elif cfg.LOSS.NAME.lower() == 'jointsfocalloss':
        print('  | focal_beta:', cfg.LOSS.EXTRA.FOCAL_BETA)
        print('  | focal_alpha:', cfg.LOSS.EXTRA.FOCAL_ALPHA)
        return JointsFocalLoss(cfg.LOSS.EXTRA.FOCAL_BETA, cfg.LOSS.EXTRA.FOCAL_ALPHA)
    else:
        raise RuntimeError('Loss Not Defined! : %s' % cfg.LOSS.NAME)
