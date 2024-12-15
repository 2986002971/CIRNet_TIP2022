import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.StarNet import Backbone_StarNetS4
from module.BaseBlock import BaseConv2d, ChannelAttention, SpatialAttention
from module.cmWR import cmWR
from module.starIGF import Decoder


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

        # 解码器 - 使用starIGF中定义的Decoder替换原来的手动组合
        self.decoder = Decoder()

        # 添加self-modality attention refinement模块
        self.ca_rgb = ChannelAttention(channels[4])  # channels[4] = 256
        self.ca_depth = ChannelAttention(channels[4])
        self.ca_rgbd = ChannelAttention(channels[4])

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)

        # 添加cross-modality weighting refinement��块
        self.cmWR = cmWR(channels[4], squeeze_ratio=1)

        # 添加卷积层
        self.conv_rgb = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.conv_depth = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.conv_rgbd = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)

    def smWR(self, x, ca, sa, conv):
        """通用的self-modality attention refinement处理
        Args:
            x: 输入特征
            ca: 通道注意力模块 (self.ca_rgb/self.ca_depth/self.ca_rgbd)
            sa: 空间注意力模块 (self.sa_rgb/self.sa_depth/self.sa_rgbd)
            conv: 卷积层 (self.conv_rgb/self.conv_depth/self.conv_rgbd)
        """
        B, C, H, W = x.size()
        P = H * W

        spatial_att = sa(x).view(B, -1, P)
        channel_att = ca(x).view(B, C, -1)
        attention = torch.bmm(channel_att, spatial_att).view(B, C, H, W)

        refined = x * attention + x
        refined = conv(refined)
        return refined

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
        conv3_rgbd_map_resize = F.interpolate(
            conv3_rgbd_map,
            size=conv4_rgbd.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map_resize + conv4_rgbd
        conv4_rgbd_map = self.sa2(conv4_rgbd)

        decoder_rgb_list.append(conv4_res_r)
        decoder_depth_list.append(conv4_res_d)

        # encoder layer 5 with PAI
        conv5_res_r = self.rgb_block5(conv4_res_r)
        conv5_res_d = self.depth_block5(conv4_res_d)

        conv5_rgbd = conv5_res_r * conv5_res_d
        conv5_rgbd = self.conv3(conv5_rgbd)
        conv4_rgbd_map_resize = F.interpolate(
            conv4_rgbd_map,
            size=conv5_rgbd.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map_resize + conv5_rgbd

        # 添加smWR处理
        conv5_res_r_refined = self.smWR(
            conv5_res_r, self.ca_rgb, self.sa_rgb, self.conv_rgb
        )
        conv5_res_d_refined = self.smWR(
            conv5_res_d, self.ca_depth, self.sa_depth, self.conv_depth
        )
        conv5_rgbd_refined = self.smWR(
            conv5_rgbd, self.ca_rgbd, self.sa_rgbd, self.conv_rgbd
        )

        # 添加cmWR处理
        conv5_res_r_refined, conv5_res_d_refined, conv5_rgbd_refined = self.cmWR(
            conv5_res_r_refined, conv5_res_d_refined, conv5_rgbd_refined
        )

        # 先添加encoder输出
        decoder_rgb_list.append(conv5_res_r)
        decoder_depth_list.append(conv5_res_d)
        # 再添加refined特征
        decoder_rgb_list.append(conv5_res_r_refined)
        decoder_depth_list.append(conv5_res_d_refined)

        rgb_map, depth_map, rgbd_map = self.decoder(
            decoder_rgb_list, decoder_depth_list, conv5_rgbd_refined
        )

        return rgb_map, depth_map, rgbd_map
