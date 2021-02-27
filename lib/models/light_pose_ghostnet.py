import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Base, DUC


class GhostModule(Base):
    def __init__(self, inp, oup, kernel_size=1, stride=1, ratio=2, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostDUC(Base):

    def __init__(self, in_planes, planes, upscale_factor=2):
        super(GhostDUC, self).__init__()
        self.conv = GhostModule(in_planes, planes)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class GhostBlock(Base):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(GhostBlock, self).__init__()

        self.conv1 = nn.Sequential(
            GhostModule(inp, oup, stride=stride, relu=True),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            GhostModule(oup, oup, stride=1, relu=False),
            nn.BatchNorm2d(oup),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != oup:
            self.shortcut = nn.Sequential(
                GhostModule(inp, oup, stride=stride, relu=False),
                # nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + self.shortcut(residual))


class LightPoseGhostNet(Base):
    def __init__(self, cfg):
        super(LightPoseGhostNet, self).__init__()
        self.name = 'light-pose-ghostnet-18'

        self.in_planes = 64
        block, num_blocks = GhostBlock, [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv_compress = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)

        if cfg.MODEL.EXTRA.USE_GHOSTDUC:
            self.duc1 = GhostDUC(256, 512, upscale_factor=2)
            self.duc2 = GhostDUC(128, 256, upscale_factor=2)
            self.duc3 = GhostDUC(64, 128, upscale_factor=2)
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
    cfg.MODEL.EXTRA.USE_GHOSTDUC = False
    cfg.freeze()

    net = LightPoseGhostNet(cfg)

    from utils.misc import modelAnalyse
    modelAnalyse(net, (3, 256, 256), max_depth=10)

# test()

