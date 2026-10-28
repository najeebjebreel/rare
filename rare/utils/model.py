"""Model architectures used in the paper (WRN-28-2, WRN-40-4, ResNet-20) and their factory."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- WideResNet -----------------

class WideBasicBlock(nn.Module):
    """WideResNet basic block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equal_in_out = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return out + shortcut


class WideNetworkBlock(nn.Module):
    """Stack of WideResNet blocks."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = [block(in_planes if i == 0 else out_planes,
                        out_planes,
                        stride if i == 0 else 1,
                        drop_rate)
                  for i in range(nb_layers)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet implementation."""

    def __init__(self, depth, num_classes, width=2, drop_rate=0.0, in_channels=3):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6
        channels = [16, 16 * width, 32 * width, 64 * width]

        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideNetworkBlock(n, channels[0], channels[1], WideBasicBlock, 1, drop_rate)
        self.block2 = WideNetworkBlock(n, channels[1], channels[2], WideBasicBlock, 2, drop_rate)
        self.block3 = WideNetworkBlock(n, channels[2], channels[3], WideBasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)


# ----------------- CIFAR ResNet (ResNet-20) -----------------

class CifarBasicBlock(nn.Module):
    """Basic block for CIFAR ResNet."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class CifarResNet(nn.Module):
    """
    CIFAR-style ResNet with 3x3 stem, no maxpool.
    Depth must be 6n+2 (e.g., 20, 32, 44, 56, 110); ResNet-20 uses n=3.
    """
    def __init__(self, depth, num_classes, in_channels=3):
        super().__init__()
        assert (depth - 2) % 6 == 0, "CIFAR ResNet depth must be 6n+2 (e.g., 20, 32, 44, 56, 110)"
        n = (depth - 2) // 6
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(CifarBasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * CifarBasicBlock.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ----------------- Model factory -----------------

def get_model(num_classes, **kwargs):
    """Build a model from the config's `model` section.

    kwargs:
      architecture (str): `wrnD-W` (WideResNet of depth D, width W, e.g.
                          wrn28-2, wrn40-4) or `resnet20` (CIFAR ResNet)
      input_channels (int)
      drop_rate (float): WideResNet only
    Other keys present in the configs (cifar_stem, pretrained) are ignored:
    both architectures already use a 3x3 CIFAR stem and train from scratch.
    """
    architecture = kwargs.get('architecture', 'wrn28-2')
    drop_rate = kwargs.get('drop_rate', 0.0)
    in_chans = kwargs.get('input_channels', 3)

    if architecture.startswith('wrn'):
        # 'wrnD-W' where D=depth, W=width
        depth, width = architecture[3:].split('-')
        return WideResNet(depth=int(depth), num_classes=num_classes, width=int(width),
                          drop_rate=drop_rate, in_channels=in_chans)
    if architecture == 'resnet20':
        return CifarResNet(depth=20, num_classes=num_classes, in_channels=in_chans)
    raise ValueError(f"Unsupported architecture {architecture!r} "
                     "(expected 'wrnD-W', e.g. 'wrn28-2', or 'resnet20')")
