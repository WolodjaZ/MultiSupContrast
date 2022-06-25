import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_features, num_classes=1000, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_features = num_features
        conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                        bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.body = nn.Sequential(OrderedDict([
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))
        
        self.global_pool = nn.Sequential(OrderedDict([
            ('global_pool', nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1))
            )
        ]))
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        x = self.body(x)
        self.embeddings  = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits

def Resnet18(model_params):
    """Constructs a resnet18.
    """
    out_chans = 512
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_feature=out_chans, num_classes=num_classes)
    return model

def Resnet34(model_params):
    """Constructs a resnet34.
    """
    out_chans = 512
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_feature=out_chans, num_classes=num_classes)
    return model

def Resnet50(model_params):
    """Constructs a resnet50.
    """
    out_chans = 2048
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_feature=out_chans, num_classes=num_classes)
    return model

def Resnet101(model_params):
    """Constructs a resnet101.
    """
    out_chans = 2048
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = ResNet(BasicBlock, [3, 4, 23, 3], num_feature=out_chans, num_classes=num_classes)
    return model


