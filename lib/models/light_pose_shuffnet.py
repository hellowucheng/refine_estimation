import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Base, DUC


class ShuffleModule(Base):
    def __init__(self, inp, oup, kernel_size=3, stride=1, group=2, first_group=False):
        super(ShuffleModule, self).__init__()

        self.group = group
        self.stride = stride
        self.inp, self.oup = inp, oup
        # 如果stride=2, 输出是原输入池化和块输出concat, 所以块的输出变为oup - inp; 如果stride=1, 就是正常的res残差
        outputs = oup - inp if stride == 2 or oup != inp else oup

        mid_channels = outputs // 4
        self.branch_main_1 = nn.Sequential(
            # pw, 1. 如果是第一个stage的第一个块, 不分组; 2. mid_channel为最终输出的1/4
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_main_2 = nn.Sequential(
            # dw
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding=kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # shuffle后, 融合组间信息
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        )
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_proj = x
        x = self.branch_main_1(x)
        x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            if self.inp == self.oup:
                return F.relu(x + x_proj)
            else:
                return torch.cat((x_proj, F.relu(x)), 1)
        else:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.size()
        # assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batch_size, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, num_channels, height, width)
        return x


class ShuffleDUC(Base):

    def __init__(self, in_planes, planes, upscale_factor=2):
        super(ShuffleDUC, self).__init__()
        self.conv = ShuffleModule(in_planes, planes, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ShuffleBlock(Base):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(ShuffleBlock, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            ShuffleModule(inp, oup, first_group=(stride == 2), kernel_size=3, stride=stride),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            # nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=False),
            ShuffleModule(oup, oup, kernel_size=3, stride=1, first_group=False),
            nn.BatchNorm2d(oup),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != oup:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(inp, self.expansion * oup, kernel_size=1, stride=stride, bias=False),
                ShuffleModule(inp, oup, kernel_size=1, stride=stride, first_group=(stride == 2)),
                nn.BatchNorm2d(self.expansion * oup)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + self.shortcut(residual))


class LightPoseShuffleNet(Base):
    def __init__(self, cfg):
        super(LightPoseShuffleNet, self).__init__()
        self.name = 'light-pose-shufflenet-18'

        self.in_planes = 64
        block, num_blocks = ShuffleBlock, [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv_compress = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)

        if cfg.MODEL.EXTRA.USE_SHUFFLEDUC:
            self.duc1 = ShuffleDUC(256, 512, upscale_factor=2)
            self.duc2 = ShuffleDUC(128, 256, upscale_factor=2)
            self.duc3 = ShuffleDUC(64, 128, upscale_factor=2)
        else:
            self.duc1 = DUC(256, 512, upscale_factor=2)
            self.duc2 = DUC(128, 256, upscale_factor=2)
            self.duc3 = DUC(64, 128, upscale_factor=2)

        self.hm_conv = nn.Conv2d(32, cfg.MODEL.NUM_JOINTS, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.conv_compress(out)

        out = self.duc1(out)
        out = self.duc2(out)
        out = self.duc3(out)

        out = self.hm_conv(out)

        return out

    def load(self, resume=None, pretrained=None):
        if self._exists(resume):
            print('resume training from:', resume)
            self.load_state_dict(torch.load(resume, map_location=lambda storage, loc: storage))
        elif self._exists(pretrained):
            print('load pretrained model:', pretrained)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # 预训练backbone载入
            model_dict = self.state_dict()
            state_dict = {k: v for k, v in torch.load(pretrained).items() if k in model_dict}
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)
        else:
            print('do not have pretrained model')
            # 初始化反卷积层
            print('=> init weights from normal distribution')

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for param in self.parameters():
            param.requires_grad = True


def test():
    from core.config import get_config
    cfg = get_config()

    cfg.defrost()
    cfg.MODEL.EXTRA.USE_SHUFFLEDUC = True
    cfg.freeze()

    net = LightPoseShuffleNet(cfg)

    from utils.misc import modelAnalyse
    modelAnalyse(net, (3, 256, 256), max_depth=5)


# test()
