import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.StarNet import Block
from module.BaseBlock import BaseConv2d, ChannelAttention


class RorD_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RorD_Decoder, self).__init__()
        self.star_block = Block(in_channels * 2, mlp_ratio=3)
        self.conv_adjust = BaseConv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, fea_before, fea_vgg):
        fea_cat = torch.cat((fea_before, fea_vgg), dim=1)
        fea_star = self.star_block(fea_cat)
        fea_adjust = self.conv_adjust(fea_star)
        return fea_adjust


class IGF(nn.Module):
    def __init__(self, fea_before_channels, fea_rd_channels, out_channels, up=True):
        super(IGF, self).__init__()
        self.up = up

        self.star_rd = Block(fea_rd_channels * 2, mlp_ratio=3)
        self.conv_rd = BaseConv2d(fea_rd_channels * 2, out_channels, kernel_size=1)

        self.star_before = Block(fea_before_channels, mlp_ratio=3)
        self.conv_before = BaseConv2d(fea_before_channels, out_channels, kernel_size=1)

        self.conv_reduce = BaseConv2d(out_channels * 2, out_channels, kernel_size=1)
        self.ca = ChannelAttention(out_channels)
        self.conv_k = BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.star_final1 = Block(out_channels, mlp_ratio=3)
        self.star_final2 = Block(out_channels, mlp_ratio=3)

    def forward(self, fea_before, fea_r, fea_d):
        fea_rd = self.star_rd(torch.cat((fea_r, fea_d), dim=1))
        fea_rd = self.conv_rd(fea_rd)

        fea_before = self.star_before(fea_before)
        fea_before = self.conv_before(fea_before)

        fea_cat = torch.cat((fea_before, fea_rd), dim=1)
        fea_reduce = self.conv_reduce(fea_cat)
        fea_ca = fea_reduce.mul(self.ca(fea_reduce)) + fea_reduce
        p_block = torch.sigmoid(self.conv_k(fea_ca))

        fea_out = fea_before * (1 - p_block) + fea_rd * p_block

        fea_out = self.star_final1(fea_out)
        fea_out = self.star_final2(fea_out)

        if self.up:
            fea_out = F.interpolate(
                fea_out, scale_factor=2, mode="bilinear", align_corners=True
            )
        return fea_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channels = [32, 32, 64, 128, 256]

        self.r1 = RorD_Decoder(channels[4], channels[3])
        self.r2 = RorD_Decoder(channels[3], channels[2])
        self.r3 = RorD_Decoder(channels[2], channels[1])
        self.r4 = RorD_Decoder(channels[1], channels[0])
        self.r5 = RorD_Decoder(channels[0], 3)

        self.d1 = RorD_Decoder(channels[4], channels[3])
        self.d2 = RorD_Decoder(channels[3], channels[2])
        self.d3 = RorD_Decoder(channels[2], channels[1])
        self.d4 = RorD_Decoder(channels[1], channels[0])
        self.d5 = RorD_Decoder(channels[0], 3)

        self.rd1 = IGF(channels[4], channels[3], channels[3])
        self.rd2 = IGF(channels[3], channels[2], channels[2])
        self.rd3 = IGF(channels[2], channels[1], channels[1])
        self.rd4 = IGF(channels[1], channels[0], channels[0])
        self.rd5 = IGF(channels[0], 3, 3)

        self.conv_r_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_d_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_rgbd_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, rgb_list, depth_list, rgbd):
        rgb_block5 = self.r1(rgb_list[5], rgb_list[4])
        rgb_block5_up = F.interpolate(rgb_block5, scale_factor=2, mode="bilinear")
        rgb_block4 = self.r2(rgb_block5_up, rgb_list[3])
        rgb_block4_up = F.interpolate(rgb_block4, scale_factor=2, mode="bilinear")
        rgb_block3 = self.r3(rgb_block4_up, rgb_list[2])
        rgb_block3_up = F.interpolate(rgb_block3, scale_factor=2, mode="bilinear")
        rgb_block2 = self.r4(rgb_block3_up, rgb_list[1])
        rgb_block2_up = F.interpolate(rgb_block2, scale_factor=2, mode="bilinear")
        rgb_block1 = self.r5(rgb_block2_up, rgb_list[0])
        rgb_block1_up = F.interpolate(rgb_block1, scale_factor=2, mode="bilinear")
        rgb_map = self.conv_r_map(rgb_block1_up)

        depth_block5 = self.d1(depth_list[5], depth_list[4])
        depth_block5_up = F.interpolate(depth_block5, scale_factor=2, mode="bilinear")
        depth_block4 = self.d2(depth_block5_up, depth_list[3])
        depth_block4_up = F.interpolate(depth_block4, scale_factor=2, mode="bilinear")
        depth_block3 = self.d3(depth_block4_up, depth_list[2])
        depth_block3_up = F.interpolate(depth_block3, scale_factor=2, mode="bilinear")
        depth_block2 = self.d4(depth_block3_up, depth_list[1])
        depth_block2_up = F.interpolate(depth_block2, scale_factor=2, mode="bilinear")
        depth_block1 = self.d5(depth_block2_up, depth_list[0])
        depth_block1_up = F.interpolate(depth_block1, scale_factor=2, mode="bilinear")
        depth_map = self.conv_d_map(depth_block1_up)

        rgbd_block5 = self.rd1(rgbd, rgb_block5, depth_block5)
        rgbd_block4 = self.rd2(rgbd_block5, rgb_block4, depth_block4)
        rgbd_block3 = self.rd3(rgbd_block4, rgb_block3, depth_block3)
        rgbd_block2 = self.rd4(rgbd_block3, rgb_block2, depth_block2)
        rgbd_block1 = self.rd5(rgbd_block2, rgb_block1, depth_block1)
        rgbd_map = self.conv_rgbd_map(rgbd_block1)

        return rgb_map, depth_map, rgbd_map
