import torch
import torch.nn as nn
import torch.nn.functional as F

from module.BaseBlock import BaseConv2d, ChannelAttention


class LargeIGF(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, up=True):
        super().__init__()
        self.up = up

        # 特征转换层
        self.conv_enc_r = BaseConv2d(
            encoder_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_dec_r = BaseConv2d(
            decoder_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_enc_d = BaseConv2d(
            encoder_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_dec_d = BaseConv2d(
            decoder_channels, out_channels, kernel_size=3, padding=1
        )

        # RGB和深度分支的融合层
        self.conv_r_fuse = BaseConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_d_fuse = BaseConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # 跨模态融合层
        self.conv_fuse = BaseConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv_reduce = BaseConv2d(out_channels, out_channels, kernel_size=1)

        # p门控相关层
        self.conv_gate_fuse = BaseConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )  # 用于处理融合特征
        self.conv_gate_before = BaseConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )  # 用于处理前一层特征
        self.conv_reduce = BaseConv2d(out_channels, out_channels, kernel_size=1)  # 降维
        self.ca = ChannelAttention(out_channels)
        self.conv_k = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, enc_r, dec_r, enc_d, dec_d, fea_before=None):
        # 特征转换
        enc_r = self.conv_enc_r(enc_r)
        dec_r = self.conv_dec_r(dec_r)
        enc_d = self.conv_enc_d(enc_d)
        dec_d = self.conv_dec_d(dec_d)

        # RGB流
        fea_r = enc_r * dec_r
        rgb_out = self.conv_r_fuse(fea_r)

        # 深度流
        fea_d = enc_d * dec_d
        depth_out = self.conv_d_fuse(fea_d)

        # 融合流
        fea_fuse = fea_r * fea_d
        fea_fuse = fea_fuse + enc_r + dec_r + enc_d + dec_d
        fea_fuse = self.conv_fuse(fea_fuse)

        if fea_before is not None:
            # p门控计算 - 使用元素乘替代通道拼接
            fea_gate_fuse = self.conv_gate_fuse(fea_fuse)
            fea_gate_before = self.conv_gate_before(fea_before)

            # 元素乘融合
            fea_gate = fea_gate_fuse * fea_gate_before
            fea_gate = self.conv_reduce(fea_gate)

            # 通道注意力和最终门控
            fea_gate_ca = fea_gate.mul(self.ca(fea_gate)) + fea_gate
            p_block = torch.sigmoid(self.conv_k(fea_gate_ca))

            # 门控融合
            fea_fuse = fea_fuse * p_block + fea_before * (1 - p_block)

        if self.up:
            rgb_out = F.interpolate(
                rgb_out, scale_factor=2, mode="bilinear", align_corners=True
            )
            depth_out = F.interpolate(
                depth_out, scale_factor=2, mode="bilinear", align_corners=True
            )
            fea_fuse = F.interpolate(
                fea_fuse, scale_factor=2, mode="bilinear", align_corners=True
            )

        return rgb_out, depth_out, fea_fuse


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # StarNet的通道数
        channels = [32, 32, 64, 128, 256]

        self.igf1 = LargeIGF(channels[4], channels[4], channels[3])  # 256->128
        self.igf2 = LargeIGF(channels[3], channels[3], channels[2])  # 128->64
        self.igf3 = LargeIGF(channels[2], channels[2], channels[1])  # 64->32
        self.igf4 = LargeIGF(channels[1], channels[1], channels[0])  # 32->32
        self.igf5 = LargeIGF(channels[0], channels[0], 3)  # 32->3

        self.conv_r_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_d_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_rgbd_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, rgb_list, depth_list, rgbd):
        # 解码过程
        r1, d1, f1 = self.igf1(
            rgb_list[4], rgb_list[5], depth_list[4], depth_list[5], rgbd
        )
        r2, d2, f2 = self.igf2(rgb_list[3], r1, depth_list[3], d1, f1)
        r3, d3, f3 = self.igf3(rgb_list[2], r2, depth_list[2], d2, f2)
        r4, d4, f4 = self.igf4(rgb_list[1], r3, depth_list[1], d3, f3)
        r5, d5, f5 = self.igf5(rgb_list[0], r4, depth_list[0], d4, f4)

        # 生成最终的预测图
        rgb_map = self.conv_r_map(r5)
        depth_map = self.conv_d_map(d5)
        rgbd_map = self.conv_rgbd_map(f5)

        return rgb_map, depth_map, rgbd_map
