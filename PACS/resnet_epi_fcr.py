import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFeatModule(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFeatModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetClassifierModule(nn.Module):
    def __init__(self, block, num_classes=1000):
        self.inplanes = 64
        super(ResNetClassifierModule, self).__init__()
        # the multi domain blocks

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        end_points = {'Predictions': F.softmax(input=x, dim=-1)}
        return x, end_points


class DomainSpecificNN(nn.Module):
    def __init__(self, num_classes):
        super(DomainSpecificNN, self).__init__()

        self.feature1 = ResNetFeatModule(BasicBlock, [2, 2, 2, 2])
        self.feature2 = ResNetFeatModule(BasicBlock, [2, 2, 2, 2])
        self.feature3 = ResNetFeatModule(BasicBlock, [2, 2, 2, 2])

        self.features = [self.feature1, self.feature2, self.feature3]

        self.classifier1 = ResNetClassifierModule(BasicBlock, num_classes)
        self.classifier2 = ResNetClassifierModule(BasicBlock, num_classes)
        self.classifier3 = ResNetClassifierModule(BasicBlock, num_classes)

        self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, domain):
        net = self.features[domain](x)
        net, end_points = self.classifiers[domain](net)
        return net, end_points


class DomainAGG(nn.Module):
    def __init__(self, num_classes):
        super(DomainAGG, self).__init__()

        self.feature = ResNetFeatModule(BasicBlock, [2, 2, 2, 2])
        self.classifier = ResNetClassifierModule(BasicBlock, num_classes)
        self.classifierrand = ResNetClassifierModule(BasicBlock, num_classes)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, agg_only=True):
        net = self.feature(x)
        net_rand = None
        if agg_only:
            net_agg, end_points = self.classifier(net)
        else:
            net_agg, end_points = self.classifier(net)
            net_rand, _ = self.classifierrand(net)
        return net_agg, net_rand, end_points
