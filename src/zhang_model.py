import torch
import torch.nn as nn
import torch.nn.functional as F

class ZhangColorizationNet(nn.Module):
    def __init__(self):
        super(ZhangColorizationNet, self).__init__()

        def conv(in_c, out_c, k=3, s=1, p=1, d=1):
            return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d)

        self.model = nn.Sequential(
            conv(1, 64, s=2), nn.ReLU(inplace=True),  # conv1
            conv(64, 128, s=2), nn.ReLU(inplace=True),  # conv2
            conv(128, 256, s=2), nn.ReLU(inplace=True),  # conv3
            conv(256, 512), nn.ReLU(inplace=True),       # conv4
            conv(512, 512, d=2, p=2), nn.ReLU(inplace=True),  # conv5 (dilated)
            conv(512, 512, d=2, p=2), nn.ReLU(inplace=True),  # conv6 (dilated)
            conv(512, 512), nn.ReLU(inplace=True),       # conv7
        )

        # conv8 and upsampling
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.color_logits = nn.Conv2d(256, 313, kernel_size=1)  # classification into 313 bins

    def forward(self, x):
        x = self.model(x)
        x = self.relu8(self.conv8(x))
        x = self.upsample1(x)              # Now shape is [B, 256, 64, 64]
        logits = self.color_logits(x)     # Output: [B, 313, 64, 64]
        return logits