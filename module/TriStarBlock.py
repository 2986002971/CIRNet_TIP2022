import torch.nn as nn
import torch.nn.functional as F

from module.FusionBlock import FusionBlock


class TriStar(nn.Module):
    def __init__(
        self,
        in_channels,  # rgb/depth的输入通道数
        rgbd_in_channels=None,  # rgbd的输入通道数
        out_channels=None,  # 输出通道数
        resize_rgbd=False,  # 是否需要调整rgbd尺寸到与rgb/depth一致
    ):
        super().__init__()

        # 如果没有指定,使用默认值
        self.rgbd_in_channels = rgbd_in_channels or in_channels
        self.out_channels = out_channels or in_channels
        self.resize_rgbd = resize_rgbd

        # RGB-Depth融合
        self.rd_fusion = FusionBlock(in_channels)

        # 如果rgbd通道数不同,需要调整通道
        if self.rgbd_in_channels != in_channels:
            self.rgbd_conv = nn.Conv2d(self.rgbd_in_channels, in_channels, 1)

        # RGBD融合
        self.rgbd_fusion = FusionBlock(in_channels)

        # 如果需要调整输出通道
        if self.out_channels != in_channels:
            self.out_conv = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(self, rgb, depth, rgbd=None):
        # 1. 先融合RGB-Depth
        rd_fused = self.rd_fusion(rgb, depth)

        # 如果没有rgbd输入,直接返回rd_fused
        if rgbd is None:
            if hasattr(self, "out_conv"):
                return self.out_conv(rd_fused)
            return rd_fused

        # 2. 处理RGBD的通道数
        if hasattr(self, "rgbd_conv"):
            rgbd = self.rgbd_conv(rgbd)

        # 3. 如果需要,将rgbd的尺寸调整为与rgb/depth一致
        if self.resize_rgbd:
            rgbd = F.interpolate(
                rgbd,
                size=rgb.shape[2:],  # 使用rgb的高宽
                mode="bilinear",
                align_corners=True,
            )

        # 4. 最终融合
        out = self.rgbd_fusion(rd_fused, rgbd)

        # 5. 调整输出通道
        if hasattr(self, "out_conv"):
            out = self.out_conv(out)

        return out
