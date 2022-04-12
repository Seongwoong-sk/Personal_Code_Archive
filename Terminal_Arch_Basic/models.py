# Resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

       
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(planes)
                            )

    def forward(self, x):
        out = self.conv_block(x)
        # 입력 채널 수(in_planes)가 출력 채널 수(planes)보다 적은 경우에도 downsample을 하여 x와 out 채널 수를 맞춰준다.
        # stride=2인 경우, feature map의 사이즈는(out)은 입력 사이즈(x)보다 작아진다. 따라서 이 경우에는 x의 사이즈를 out 사이즈에 맞춰야한다.
        out += self.downsample(x)

        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # layer 1개
        self.base = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        # layer 16개
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gap = nn.AvgPool2d(4)
        # layer 1개 :: 총 18개
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 메인으로 보내기 위해서 모델 타입을 받아서 클래스를 내보내줌.
def modeltype(model):
    if model == 'resnet18':
        return ResNet(ResidualBlock, [2, 2, 2, 2])

    elif model == 'resnet34':
        return ResNet(ResidualBlock, [3, 4, 6, 3])

