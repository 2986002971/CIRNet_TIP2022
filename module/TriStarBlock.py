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

        # RGBD直接升维到目标维度
        self.f2_rgbd = ConvBN(self.rgbd_in_channels, mlp_ratio * in_channels, 1)

        # 最终的降维层
        if self.out_channels != in_channels:
            self.g = ConvBN(mlp_ratio * in_channels, self.out_channels, 1)
            self.rgb_res_conv = ConvBN(in_channels, self.out_channels, 1)
            self.depth_res_conv = ConvBN(in_channels, self.out_channels, 1)
        else:
            self.g = ConvBN(mlp_ratio * in_channels, in_channels, 1)

        self.act = nn.ReLU6()

    def forward(self, rgb, depth, rgbd=None):
        # 保存输入用于长程残差
        rgb_res = rgb
        depth_res = depth

        # 1. RGB-Depth融合(在高维空间)
        rgb_high = self.f1_rgb(rgb)
        depth_high = self.f1_depth(depth)
        rd_high = self.act(rgb_high) * depth_high

        # 如果没有rgbd输入,直接降维返回
        if rgbd is None:
            rd_fused = self.g(rd_high)
            if hasattr(self, "rgb_res_conv"):
                rgb_res = self.rgb_res_conv(rgb_res)
                depth_res = self.depth_res_conv(depth_res)
                return rd_fused + rgb_res + depth_res
            return rd_fused + rgb + depth

        # 2. 如果需要,将rgbd的尺寸调整为与rgb/depth一致
        if self.resize_rgbd:
            rgbd = F.interpolate(
                rgbd, size=rgb.shape[2:], mode="bilinear", align_corners=True
            )

        # 3. RGBD融合(在高维空间)
        rgbd_high = self.f2_rgbd(rgbd)
        out = self.g(self.act(rd_high) * rgbd_high)

        # 4. 添加长程残差
        if hasattr(self, "rgb_res_conv"):
            rgb_res = self.rgb_res_conv(rgb_res)
            depth_res = self.depth_res_conv(depth_res)
            return out + rgb_res + depth_res

        return out + rgb + depth
