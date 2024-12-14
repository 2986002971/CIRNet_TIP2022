import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.StarNet import Backbone_StarNetS4
from module.BaseBlock import BaseConv2d, SpatialAttention
from module.starIGF import LargeIGF


class CIRNet_Ustar(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CIRNet_Ustar, self).__init__()

        # 加载StarNet骨干网
        (
            self.rgb_block1,
            self.rgb_block2,
            self.rgb_block3,
            self.rgb_block4,
            self.rgb_block5,
        ) = Backbone_StarNetS4(pretrained=True)

        (
            self.depth_block1,
            self.depth_block2,
            self.depth_block3,
            self.depth_block4,
            self.depth_block5,
        ) = Backbone_StarNetS4(pretrained=True)

        # StarNet的通道数
        channels = [32, 32, 64, 128, 256]

        # PAI单元的卷积层
        self.conv1 = BaseConv2d(channels[2], channels[2], kernel_size=3, padding=1)
        self.conv2 = BaseConv2d(channels[3], channels[3], kernel_size=3, padding=1)
        self.conv3 = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)

        # 空间注意力
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)

        # 解码器
        self.decoder = LargeIGF(channels[4], channels[4], channels[3])  # 256->128
        self.decoder2 = LargeIGF(channels[3], channels[3], channels[2])  # 128->64
        self.decoder3 = LargeIGF(channels[2], channels[2], channels[1])  # 64->32
        self.decoder4 = LargeIGF(channels[1], channels[1], channels[0])  # 32->32
        self.decoder5 = LargeIGF(channels[0], channels[0], 3)  # 32->3

        # 最终预测层
        self.conv_r_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_d_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.conv_rgbd_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, rgb, depth):
        decoder_rgb_list = []
        decoder_depth_list = []
        depth = torch.cat((depth, depth, depth), dim=1)

        # encoder layer 1-2
        conv1_res_r = self.rgb_block1(rgb)
        conv1_res_d = self.depth_block1(depth)
        conv2_res_r = self.rgb_block2(conv1_res_r)
        conv2_res_d = self.depth_block2(conv1_res_d)

        decoder_rgb_list.extend([conv1_res_r, conv2_res_r])
        decoder_depth_list.extend([conv1_res_d, conv2_res_d])

        # encoder layer 3 with PAI
        conv3_res_r = self.rgb_block3(conv2_res_r)
        conv3_res_d = self.depth_block3(conv2_res_d)

        # 使用元素乘替代通道拼接
        conv3_rgbd = conv3_res_r * conv3_res_d
        conv3_rgbd = self.conv1(conv3_rgbd)
        conv3_rgbd = F.interpolate(
            conv3_rgbd, scale_factor=1 / 2, mode="bilinear", align_corners=True
        )
        conv3_rgbd_map = self.sa1(conv3_rgbd)

        decoder_rgb_list.append(conv3_res_r)
        decoder_depth_list.append(conv3_res_d)

        # encoder layer 4 with PAI
        conv4_res_r = self.rgb_block4(conv3_res_r)
        conv4_res_d = self.depth_block4(conv3_res_d)

        conv4_rgbd = conv4_res_r * conv4_res_d
        conv4_rgbd = self.conv2(conv4_rgbd)
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map + conv4_rgbd
        conv4_rgbd_map = self.sa2(conv4_rgbd)

        decoder_rgb_list.append(conv4_res_r)
        decoder_depth_list.append(conv4_res_d)

        # encoder layer 5 with PAI
        conv5_res_r = self.rgb_block5(conv4_res_r)
        conv5_res_d = self.depth_block5(conv4_res_d)

        conv5_rgbd = conv5_res_r * conv5_res_d
        conv5_rgbd = self.conv3(conv5_rgbd)
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map + conv5_rgbd

        decoder_rgb_list.append(conv5_res_r)
        decoder_depth_list.append(conv5_res_d)

        # decoder
        r1, d1, f1 = self.decoder(
            decoder_rgb_list[4],
            decoder_rgb_list[5],
            decoder_depth_list[4],
            decoder_depth_list[5],
            conv5_rgbd,
        )
        r2, d2, f2 = self.decoder2(
            decoder_rgb_list[3], r1, decoder_depth_list[3], d1, f1
        )
        r3, d3, f3 = self.decoder3(
            decoder_rgb_list[2], r2, decoder_depth_list[2], d2, f2
        )
        r4, d4, f4 = self.decoder4(
            decoder_rgb_list[1], r3, decoder_depth_list[1], d3, f3
        )
        r5, d5, f5 = self.decoder5(
            decoder_rgb_list[0], r4, decoder_depth_list[0], d4, f4
        )

        # 最终预测
        rgb_map = self.conv_r_map(r5)
        depth_map = self.conv_d_map(d5)
        rgbd_map = self.conv_rgbd_map(f5)

        return rgb_map, depth_map, rgbd_map
