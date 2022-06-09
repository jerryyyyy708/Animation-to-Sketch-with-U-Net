import torch
import torch.nn as nn

class SketchUNet(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 3
        self.in_conv = SuccessiveConv(3, 64)
        self.down_1 = ContractingPath(64, 128)
        self.down_2 = ContractingPath(128, 256)
        self.down_3 = ContractingPath(256, 512)
        self.down_4 = ContractingPath(512, 1024)
        self.up_1 = ExpandingPath(1024, 512)
        self.up_2 = ExpandingPath(512, 256)
        self.up_3 = ExpandingPath(256, 128)
        self.up_4 = ExpandingPath(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x_in_conv = self.in_conv(x)
        x_down_1 = self.down_1(x_in_conv)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_4 = self.down_4(x_down_3)
        x_up_1 = self.up_1(x_down_4, x_down_3)
        x_up_2 = self.up_2(x_up_1, x_down_2)
        x_up_3 = self.up_3(x_up_2, x_down_1)
        x_up_4 = self.up_4(x_up_3, x_in_conv)
        x_out_conv = self.out_conv(x_up_4)
        return x_out_conv


class SuccessiveConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.successive_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.successive_conv(x)


class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.contracting_path = nn.Sequential(
            SuccessiveConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.contracting_path(x)


class ExpandingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SuccessiveConv(int(in_channels / 2 * 3), out_channels)
    
    def forward(self, x1, x2):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x, x2), dim=1)
        return self.conv(x)
