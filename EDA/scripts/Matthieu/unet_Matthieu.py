import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    mid_channels: int or None
        Intermediate channels (defaults to out_channels)
    dropout: float
        Dropout rate applied after each ReLU

    Applies two convolutional layers each followed by BatchNorm, ReLU, and optional dropout.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    dropout: float
        Dropout rate

    Downscaling block with MaxPooling followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    in_channels: int
        Number of input channels from the decoder
    out_channels: int
        Number of output channels after upsampling and conv
    bilinear: bool
        Whether to use bilinear upsampling or transposed conv
    dropout: float
        Dropout rate

    Upscaling block with bilinear upsampling or transposed convolution followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch with padding
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output segmentation classes

    Final 1x1 convolution to map features to class predictions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UnetMatthieu(nn.Module):
    """
    n_channels: int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB)
    n_classes: int
        Number of output segmentation classes
    features: list
        List of channel sizes for each level of the encoder/decoder
    bilinear: bool
        Whether to use bilinear upsampling or transposed conv
    dropout: float
        Dropout rate (only applied in deeper layers)

    U-Net model architecture.
    """
    def __init__(self, n_channels, n_classes, features=[64, 128, 256, 512, 1024], bilinear=False, dropout=0.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[4] // factor, dropout=dropout)

        self.up1 = Up(features[4], features[3] // factor, bilinear, dropout=dropout / 2)
        self.up2 = Up(features[3], features[2] // factor, bilinear, dropout=dropout / 3)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4); del x4, x5
        x = self.up2(x, x3); del x3
        x = self.up3(x, x2); del x2
        x = self.up4(x, x1); del x1

        logits = self.outc(x); del x
        return logits
