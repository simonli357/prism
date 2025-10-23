# deeplabv3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class _ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ASPPPooling, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.block(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(_ASPPConv(in_channels, out_channels, rate))
        
        modules.append(_ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_ch, out_ch, base=32):
        super(DeepLabV3Plus, self).__init__()
        
        backbone = models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
        
        self.backbone_conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_bn1 = backbone.bn1
        self.backbone_relu = backbone.relu
        self.backbone_maxpool = backbone.maxpool
        
        self.backbone_layer1 = backbone.layer1 # For low-level features
        self.backbone_layer2 = backbone.layer2
        self.backbone_layer3 = backbone.layer3
        self.backbone_layer4 = backbone.layer4 # For high-level features
        
        atrous_rates = (6, 12, 18)
        
        aspp_in_channels = 2048 # Output channels from ResNet-50's layer4
        aspp_out_channels = 256
        self.aspp = _ASPP(aspp_in_channels, aspp_out_channels, atrous_rates)
        
        low_level_channels = 256 # Output channels from ResNet-50's layer1
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.final_conv = nn.Conv2d(256, out_ch, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x_last_frame = x[:, -1, :, :, :]

        input_size = x_last_frame.shape[-2:]
        
        features = self.backbone_conv1(x_last_frame)
        features = self.backbone_bn1(features)
        features = self.backbone_relu(features)
        features = self.backbone_maxpool(features)
        
        low_level_features = self.backbone_layer1(features)
        features = self.backbone_layer2(low_level_features)
        features = self.backbone_layer3(features)
        features = self.backbone_layer4(features)
        
        high_level_features = self.aspp(features)
        
        high_level_features_up = F.interpolate(high_level_features, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)
        low_level_features_proc = self.low_level_conv(low_level_features)
        
        concat_features = torch.cat([high_level_features_up, low_level_features_proc], dim=1)
        refined_features = self.decoder_conv(concat_features)
        
        final_logits = self.final_conv(refined_features)
        
        output = F.interpolate(final_logits, size=input_size, mode='bilinear', align_corners=False)
        
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)