import os
import sys
import platform
from pathlib import Path
from yacs.config import CfgNode as CN

_C = CN()

_C.GPUS = '0'
_C.CUDNN = CN()
_C.CUDNN.ENABLED = True
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = True

_C.DEBUG_ON = False
_C.DEBUG_STEPS = 140
_C.VISUAL_OUTPUT = True

_C.NUM_WORKERS = 4

_C.WORK_DIR = ''

# 配置文件路径
_C.EXPERIMENTS_PATH = 'D:/Home/Projects/refine_estimation/experiments/refine-50.yaml'
# _C.EXPERIMENTS_PATH = '/root/Projects/refine_estimation/experiments/refine-50.yaml'


_C.LOG_DIR = ''
_C.OUTPUT_DIR = ''
_C.CHECKPOINTS_PATH = ''

# 数据集相关
_C.DATASET = CN()
_C.DATASET.NAME = 'mpii'
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_SET = ''
_C.DATASET.VALID_SET = ''
_C.DATASET.TEST_SET = ''
_C.DATASET.KEEP_ORIGIN = True

_C.DATASET.TRAIN_AUGMENT = CN()
_C.DATASET.TRAIN_AUGMENT.USE = False
_C.DATASET.TRAIN_AUGMENT.FLIP = True
_C.DATASET.TRAIN_AUGMENT.ROT_FACTOR = 30
_C.DATASET.TRAIN_AUGMENT.SCALE_FACTOR = 0.25


# 模型相关
_C.MODEL = CN()
_C.MODEL.NAME = 'posresnet'
_C.MODEL.RESUME = ''
_C.MODEL.PRETRAINED = ''

_C.MODEL.NUM_JOINTS = 16

_C.MODEL.SIGMA = 2
_C.MODEL.IMAGE_SIZE = 256
_C.MODEL.HEATMAP_SIZE = 64

_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.NUM_LAYERS = 18

_C.MODEL.EXTRA.USE_DECONV = False
_C.MODEL.EXTRA.USE_CPDUC = False
_C.MODEL.EXTRA.USE_MOBILEDUC = False
_C.MODEL.EXTRA.USE_SHUFFLEDUC = False
_C.MODEL.EXTRA.USE_GHOSTDUC = False

# 损失函数
_C.LOSS = CN()

_C.LOSS.NAME = 'JointsMSELoss'

_C.LOSS.EXTRA = CN(new_allowed=True)

_C.LOSS.EXTRA.USE_TARGET_WEIGHT = True

_C.LOSS.EXTRA.FOCAL_BETA = 4
_C.LOSS.EXTRA.FOCAL_ALPHA = 2

_C.LOSS.EXTRA.BALANCED_ALPHA = 0.005


# 训练相关
_C.TRAIN = CN()

# 1.仅使用可见点训练、2.仅使用遮挡点训练、3.都使用
_C.TRAIN.DATA_MODE = 3

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LAST_EPOCH = -1
_C.TRAIN.NUM_EPOCH = 140

# 优化器
_C.TRAIN.OPTIM = 'adam'
_C.TRAIN.MOMENTUM = 0.9
# l2正则化
_C.TRAIN.WEIGHT_DECAY = 0.

# 初始学习率
_C.TRAIN.LR = 1e-3
# 学习率调整
_C.TRAIN.SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SCHEDULER.NAME = 'multi-step'
_C.TRAIN.SCHEDULER.GAMMA = 0.1
_C.TRAIN.SCHEDULER.MILESTONES = [90, 120]


# 测试相关
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 16
_C.TEST.FLIP_TEST = True

# 评估 1: 仅可见点 2: 仅遮挡点 3: 全体
_C.TEST.DATA_MODE = 3
_C.TEST.TRAINED_MODEL = ''


def join_path(root, path):
    if path:
        return os.path.join(root, path)
    return root


def get_config():
    _C.defrost()

    # 指定配置文件
    if len(sys.argv) > 1:
        _C.EXPERIMENTS_PATH = sys.argv[1]

    if Path(_C.EXPERIMENTS_PATH).is_file():
        print('| load experiments from:', _C.EXPERIMENTS_PATH)
        _C.merge_from_file(_C.EXPERIMENTS_PATH)

    # 不同主机路径适配
    if 'Darwin' in platform.platform():
        _C.WORK_DIR = '/Users/wucheng/仓库/'
        _C.LOG_DIR = '/Users/wucheng/仓库/pose_estimation/logs/'
        _C.OUTPUT_DIR = '/Users/wucheng/仓库/pose_estimation/outputs/'
        _C.CHECKPOINTS_PATH = '/Users/wucheng/仓库/pose_estimation/checkpoints/'
    elif 'Linux' in platform.platform():
        _C.WORK_DIR = '/data-tmp/'
        _C.LOG_DIR = '/data-tmp/pose_estimation/logs/'
        _C.OUTPUT_DIR = '/data-output/'
        _C.CHECKPOINTS_PATH = '/data-tmp/pose_estimation/checkpoints/'
    else:
        _C.WORK_DIR = 'D:/Home/Storehouse/'
        _C.LOG_DIR = 'D:/Home/Storehouse/pose_estimation/logs/'
        _C.OUTPUT_DIR = 'D:/Home/Storehouse/pose_estimation/outputs/'
        _C.CHECKPOINTS_PATH = 'D:/Home/Storehouse/pose_estimation/checkpoints/'

    # 更新路径
    _C.DATASET.ROOT = join_path(_C.WORK_DIR, _C.DATASET.ROOT)

    _C.MODEL.RESUME = join_path(_C.WORK_DIR, _C.MODEL.RESUME)
    _C.MODEL.PRETRAINED = join_path(_C.WORK_DIR, _C.MODEL.PRETRAINED)
    _C.TEST.TRAINED_MODEL = join_path(_C.WORK_DIR, _C.TEST.TRAINED_MODEL)

    _C.freeze()
    return _C
