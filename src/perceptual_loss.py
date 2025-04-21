import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:9]  # conv2_2
        self.vgg_layers = vgg.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.resize = resize
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        if self.resize:
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        return self.criterion(self.vgg_layers(input), self.vgg_layers(target))