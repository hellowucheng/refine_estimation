import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Base, DUC


class BasicBlock(Base):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + self.shortcut(residual))


class Bottleneck(Base):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return F.relu(x + self.shortcut(residual))


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class LightPoseResNet(Base):

    def __init__(self, cfg):
        super(LightPoseResNet, self).__init__()
        self.name = 'light-pose-resnet-18'

        self.in_planes = 64
        extra = cfg.MODEL.EXTRA
        self.use_deconv = extra.USE_DECONV
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        block, layers = resnet_spec[cfg.MODEL.EXTRA.NUM_LAYERS]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.use_deconv:
            self.deconv_layers = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,
                extra.NUM_DECONV_FILTERS,
                extra.NUM_DECONV_KERNELS,
            )

            self.final_layer = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
                bias=True
            )
        else:
            self.conv_compress = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.duc1 = DUC(256, 512, upscale_factor=2)
            self.duc2 = DUC(128, 256, upscale_factor=2)
            self.duc3 = DUC(64, 128, upscale_factor=2)
            self.hm_conv = nn.Conv2d(32, cfg.MODEL.NUM_JOINTS, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        deconv_cfg = {2: (0, 0), 3: (1, 1), 4: (1, 0)}

        layers = []
        for i in range(num_layers):
            kernel, planes, (padding, output_padding) = num_kernels[i], num_filters[i], deconv_cfg[num_kernels[i]]

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.in_planes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True)
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_deconv:
            x = self.deconv_layers(x)
            x = self.final_layer(x)
        else:
            x = self.conv_compress(x)

            x = self.duc1(x)
            x = self.duc2(x)
            x = self.duc3(x)

            x = self.hm_conv(x)

        return x

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
    net = LightPoseResNet(get_config())

    from utils.misc import modelAnalyse
    modelAnalyse(net, (3, 256, 256), max_depth=5)

# test()
