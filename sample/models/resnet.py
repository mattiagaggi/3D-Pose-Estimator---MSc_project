import torch.nn as nn
from sample.base.base_model import BaseModel


class ResNet(BaseModel):
    def __init__(self, block, layers, num_channels=3, do_maxpooling=True):
        self.do_maxpooling = do_maxpooling
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.conv1(x)  # size /2 (due to stride)
        x = self.bn1(x)
        x0 = self.relu(x)
        if self.do_maxpooling:
            x = self.maxpool(x0)  # size /2 (due to pooling)
        else:
            x = x0  # size /2 (due to pooling)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)  # size /2 (due to stride)
        #        x3 = self.layer3(x2)# size /2 (due to stride)
        #        x4 = self.layer4(x3)# size /2 (due to stride)

        return [x0, x1, x2]  # , x3, x4]