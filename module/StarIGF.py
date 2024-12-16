import torch.nn as nn
import torch.nn.functional as F

from module.BaseBlock import ConvBN
from module.FusionBlock import FusionBlock
from module.StarBlock import Block


class LargeIGF(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, up=True):
        super().__init__()
        self.up = up

        # 特征转换层
        self.star_enc_r = Block(encoder_channels, mlp_ratio=3)
        self.star_dec_r = Block(decoder_channels, mlp_ratio=3)
        self.star_enc_d = Block(encoder_channels, mlp_ratio=3)
        self.star_dec_d = Block(decoder_channels, mlp_ratio=3)

        # 特征融合模块 + Star Block增强
        self.rgb_fusion = FusionBlock(encoder_channels)
        self.rgb_enhance = Block(encoder_channels, mlp_ratio=3)

        self.depth_fusion = FusionBlock(encoder_channels)
        self.depth_enhance = Block(encoder_channels, mlp_ratio=3)

        self.cross_fusion = FusionBlock(encoder_channels)
        self.cross_enhance = Block(encoder_channels, mlp_ratio=3)

        # 使用FusionBlock替代原来的门控机制
        self.temporal_fusion = FusionBlock(encoder_channels)

        # 输出通道调整层
        self.conv_out_r = ConvBN(encoder_channels, out_channels, 1)
        self.conv_out_d = ConvBN(encoder_channels, out_channels, 1)
        self.conv_out_fuse = ConvBN(encoder_channels, out_channels, 1)

    def forward(self, enc_r, dec_r, enc_d, dec_d, fea_before=None):
        # 特征转换
        enc_r = self.star_enc_r(enc_r)
        dec_r = self.star_dec_r(dec_r)
        enc_d = self.star_enc_d(enc_d)
        dec_d = self.star_dec_d(dec_d)

        # RGB流融合 + 增强
        rgb_out = self.rgb_fusion(enc_r, dec_r)
        rgb_out = self.rgb_enhance(rgb_out)

        # 深度流融合 + 增强
        depth_out = self.depth_fusion(enc_d, dec_d)
        depth_out = self.depth_enhance(depth_out)

        # RGB和深度的跨模态融合 + 增强
        fea_fuse = self.cross_fusion(rgb_out, depth_out)
        fea_fuse = self.cross_enhance(fea_fuse)

        # 时序特征融合
        if fea_before is not None:
            fea_fuse = self.temporal_fusion(fea_fuse, fea_before)

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

        # 调整输出通道数
        rgb_out = self.conv_out_r(rgb_out)
        depth_out = self.conv_out_d(depth_out)
        fea_fuse = self.conv_out_fuse(fea_fuse)

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
