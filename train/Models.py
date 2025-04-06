import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Conv2d와 ReLU 활성화 함수 및 Batch Normalization을 포함한 블록 정의
class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 디코더 블록 정의 - 업샘플링 및 스킵 연결 포함
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip=None):
        # 스킵 연결이 있는 경우
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # 채널 방향으로 연결
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# U-Net의 디코더 정의
class UnetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet의 각 레이어와 디코더 블록 연결
        self.block1 = DecoderBlock(512 + 256, 256)  # layer4 (512) + layer3 (256)
        self.block2 = DecoderBlock(256 + 128, 128)  # block1 (256) + layer2 (128)
        self.block3 = DecoderBlock(128 + 64, 64)    # block2 (128) + layer1 (64)
        self.block4 = DecoderBlock(64 + 64, 32)     # block3 (64) + conv1 (64)
        self.block5 = DecoderBlock(32, 16)          # block4 (32) - 스킵 연결 없음
        self.final_conv = nn.Conv2d(16, 7, kernel_size=3, padding=1)  # 7개의 클래스에 대한 출력

    def forward(self, features):
        x = self.block1(features[-1], features[-2])
        x = self.block2(x, features[-3])
        x = self.block3(x, features[-4])
        x = self.block4(x, features[0])
        x = self.block5(x)  # 스킵 연결 없음
        x = self.final_conv(x)

        # 출력 크기를 타겟 크기에 맞춤
        target_size = (128, 128)  # 132 132 가 맞는지 128 128 이 맞는지..?
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

# ResNet34
class Segmentation_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet34(pretrained=True)  # 사전 학습된 ResNet34 사용
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.decoder = UnetDecoder()

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)  # 각 레이어의 출력을 features 리스트에 저장
        x = self.decoder(features)  # 디코더에 features 전달
        return x