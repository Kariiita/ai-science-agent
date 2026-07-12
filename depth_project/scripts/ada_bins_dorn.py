import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaBinsDORN(nn.Module):
    """
    DORN with AdaBins-style adaptive binning: 64 learnable logarithmic depth bins + softmax + expectation.
    """

    def __init__(self, num_classes=64, pretrained=True):
        super(AdaBinsDORN, self).__init__()

        # Encoder: ResNet-50 backbone (pretrained)
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()

        # Decoder: feature pyramid + ordinal regression head
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(1024, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(2048, 128, kernel_size=1)

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

        self.num_classes = num_classes
        # Learnable logarithmic bin centers
        self.bin_centers = nn.Parameter(torch.logspace(torch.log10(torch.tensor(0.5)), 
                                                        torch.log10(torch.tensor(10.0)), 
                                                        num_classes))

    def forward(self, x):
        x_orig = x  # save for final upsampling

        # Backbone feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x2 = self.backbone.layer1(x)   # 256ch, H/4
        x3 = self.backbone.layer2(x2)  # 512ch, H/8
        x4 = self.backbone.layer3(x3)  # 1024ch, H/16
        x5 = self.backbone.layer4(x4)  # 2048ch, H/32

        p2 = self.conv2(x2)
        p3 = self.conv3(x3)
        p4 = self.conv4(x4)
        p5 = self.conv5(x5)

        # Upsample and fuse
        p4 = p4 + self.upconv2(p5)
        p3 = p3 + self.upconv3(p4)
        p2 = p2 + self.upconv4(p3)

        out = self.final_conv(p2)
        out = F.softmax(out, dim=1)
        depth_pred = torch.sum(out * self.bin_centers.view(1, -1, 1, 1), dim=1, keepdim=True)

        # Upsample decoder output back to input resolution
        depth_pred = F.interpolate(
            depth_pred, size=x_orig.shape[2:], mode="bilinear", align_corners=False
        )

        return depth_pred


def adabins_dorn_baseline(pretrained=True):
    return AdaBinsDORN(num_classes=64, pretrained=pretrained)
