import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Base, DUC


class DUC(Base):

    def __init__(self, in_planes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class SeparableConv(Base):
    # Depthwise conv + Pointwise conv
    def __init__(self, inp, oup, stride=1, relu=True):
        super(SeparableConv, self).__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        self.pw = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileDUC(Base):

    def __init__(self, in_planes, planes, upscale_factor=2):
        super(MobileDUC, self).__init__()
        self.conv = SeparableConv(in_planes, planes)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class MobileBlock(Base):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(MobileBlock, self).__init__()

        self.conv1 = nn.Sequential(
            SeparableConv(inp, oup, stride=stride, relu=True),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            SeparableConv(oup, oup, stride=1, relu=False),
            nn.BatchNorm2d(oup),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != oup:
            self.shortcut = nn.Sequential(
                SeparableConv(inp, oup, stride=stride, relu=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + self.shortcut(residual))


class LightPoseMobileNet(Base):
    def __init__(self, cfg):
        super(LightPoseMobileNet, self).__init__()
        self.name = 'light-pose-mobilenet-18'

        self.in_planes = 64
        block, num_blocks = MobileBlock, [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv_compress = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)

        if cfg.MODEL.EXTRA.USE_MOBILEDUC:
            self.duc1 = MobileDUC(256, 512, upscale_factor=2)
            self.duc2 = MobileDUC(128, 256, upscale_factor=2)
            self.duc3 = MobileDUC(64, 128, upscale_factor=2)
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
    net = LightPoseMobileNet(get_config())

    from utils.misc import modelAnalyse
    modelAnalyse(net, (3, 256, 256), max_depth=5)


#test()
