import torch.nn as nn
import torch.nn.functional as F

from module.BaseBlock import ConvBN


class TriStar(nn.Module):
    def __init__(
        self,
        in_channels,  # rgb/depth的输入通道数
        rgbd_in_channels=None,  # rgbd的输入通道数
        out_channels=None,  # 输出通道数
        resize_rgbd=False,  # 是否需要调整rgbd尺寸到与rgb/depth一致
        mlp_ratio=3,  # 升维倍率
    ):
        super().__init__()

        # 如果没有指定,使用默认值
        self.rgbd_in_channels = rgbd_in_channels or in_channels
        self.out_channels = out_channels or in_channels
        self.resize_rgbd = resize_rgbd

        # RGB-Depth融合的升维层
        self.f1_rgb = ConvBN(in_channels, mlp_ratio * in_channels, 1)
        self.f1_depth = ConvBN(in_channels, mlp_ratio * in_channels, 1)

        # RGB-Depth融合后的降维层
        self.g1 = ConvBN(mlp_ratio * in_channels, in_channels, 1)

        # 如果rgbd通道数不同,需要调整通道
        if self.rgbd_in_channels != in_channels:
            self.rgbd_conv = ConvBN(self.rgbd_in_channels, in_channels, 1)

        # RGBD融合的升维层
        self.f2_rd = ConvBN(in_channels, mlp_ratio * in_channels, 1)
        self.f2_rgbd = ConvBN(in_channels, mlp_ratio * in_channels, 1)

        # 最终的降维层
        if self.out_channels != in_channels:
            self.g2 = ConvBN(mlp_ratio * in_channels, self.out_channels, 1)
            self.rgb_res_conv = ConvBN(in_channels, self.out_channels, 1)
            self.depth_res_conv = ConvBN(in_channels, self.out_channels, 1)
        else:
            self.g2 = ConvBN(mlp_ratio * in_channels, in_channels, 1)

        self.act = nn.ReLU6()

    def forward(self, rgb, depth, rgbd=None):
        # 保存输入用于长程残差
        rgb_res = rgb
        depth_res = depth

        # 1. RGB-Depth融合
        rgb_high = self.f1_rgb(rgb)
        depth_high = self.f1_depth(depth)
        rd_fused = self.g1(self.act(rgb_high) * depth_high)

        # 如果没有rgbd输入,直接返回rd_fused
        if rgbd is None:
            if hasattr(self, "rgb_res_conv"):
                rd_fused = self.g2(rd_fused)
                rgb_res = self.rgb_res_conv(rgb_res)
                depth_res = self.depth_res_conv(depth_res)
                return rd_fused + rgb_res + depth_res
            return rd_fused + rgb + depth

        # 2. 处理RGBD的通道数
        if hasattr(self, "rgbd_conv"):
            rgbd = self.rgbd_conv(rgbd)

        # 3. 如果需要,将rgbd的尺寸调整为与rgb/depth一致
        if self.resize_rgbd:
            rgbd = F.interpolate(
                rgbd, size=rgb.shape[2:], mode="bilinear", align_corners=True
            )

        # 4. RGBD融合
        rd_high = self.f2_rd(rd_fused)
        rgbd_high = self.f2_rgbd(rgbd)
        out = self.g2(self.act(rd_high) * rgbd_high)

        # 5. 添加长程残差
        if hasattr(self, "rgb_res_conv"):
            rgb_res = self.rgb_res_conv(rgb_res)
            depth_res = self.depth_res_conv(depth_res)
            return out + rgb_res + depth_res

        return out + rgb + depth
