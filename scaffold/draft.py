import torch
import numpy as np
from tqdm import tqdm

from core.config import get_config
from core.evaluation import calc_accuracy, get_final_preds, AverageMeter, evaluate
from utils.transforms import flip_back
from utils.factory import getDataset, getModel


if __name__ == '__main__':
    cfg = get_config()

    # cuda 相关
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.CUDNN.ENABLED else 'cpu')
    if cfg.CUDNN.ENABLED and torch.cuda.is_available():
        print("Use Cuda.")
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    dataset = getDataset(cfg, is_train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=cfg.NUM_WORKERS)

    for i, (input, target, meta) in enumerate(tqdm(dataset)):

        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)
