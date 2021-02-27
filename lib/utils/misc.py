import os
import time
import pickle
import torch
from pathlib import Path
from fvcore.nn import parameter_count, flop_count, parameter_count_table


def isExists(path):
    return path is not None and Path(path).is_file()


def modelAnalyse(net, input_size=(3, 32, 32), max_depth=5, mode='a'):
    net.to(torch.device('cpu'))

    # 参数报表
    if mode == 'a':
        print('----------------------------------------')
        print(parameter_count_table(net, max_depth=max_depth))

    # 总参数量统计
    print('----------------------------------------')
    print('PARAM: %.2fM' % (parameter_count(net)[''] / 1000000))

    # FLOPS统计
    print('----------------------------------------')
    flops = sum([1000 * x for x in flop_count(net, (torch.randn(input_size).unsqueeze_(0),))[0].values()])
    print('FLOPS: %.2fM' % flops)

    print('----------------------------------------')

    from torchstat import stat
    # stat(net, input_size)


class Timer:
    def __init__(self):
        self.clock = {}

    def tick(self, key='default'):
        self.clock[key] = time.time()

    def tock(self, key='default'):
        if key not in self.clock:
            raise Exception("{%s} is not in the clock." % key)
        interval = time.time() - self.clock[key]
        # del self.clock[key]
        return interval


class Recorder:
    def __init__(self, tape_path, model_path):
        try:
            with open(tape_path, 'rb') as f:
                self.tape = pickle.load(f)
        except IOError:
            self.tape = {}

        self.metric = None
        self.tape_path = tape_path
        self.model_path = model_path

    def records(self, d):
        for name, y in d.items():
            if name in self.tape:
                self.tape[name].append(y)
            else:
                self.tape[name] = [y]
        with open(self.tape_path, 'wb') as f:
            pickle.dump(self.tape, f)

    # 只对当前最优的模型保存断点, mode='min'表示评价指标metric越小, 模型越优
    def checkpoint(self, net, metric, file_name, mode='min'):
        # 第一次或者为至今出现的最优值, 保存断点
        if not self.metric or (mode == 'min' and metric < self.metric) or (mode == 'max' and metric > self.metric):
            self.metric = metric
            net.save(os.path.join(self.model_path, file_name))
            print('Saved model:', os.path.join(self.model_path, file_name))
