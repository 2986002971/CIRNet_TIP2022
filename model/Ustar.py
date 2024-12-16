import torch
import torch.nn as nn

from backbone.StarNet import Backbone_StarNetS4
from module.StarIGF import Decoder
from module.TriStarBlock import TriStar


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

        # 使用TriStar替换PAI模块
        self.tristar1 = TriStar(
            in_channels=channels[2],  # 64
            rgbd_in_channels=None,  # 第一次融合没有rgbd输入
            out_channels=channels[2],  # 输出通道数保持不变
            resize_rgbd=False,  # 第一次融合不需要resize
        )

        self.tristar2 = TriStar(
            in_channels=channels[3],  # 128
            rgbd_in_channels=channels[2],  # 上一层的rgbd通道数是64
            out_channels=channels[3],  # 输出通道数128
            resize_rgbd=True,  # 需要将上一层的rgbd调整到当前尺寸
        )

        self.tristar3 = TriStar(
            in_channels=channels[4],  # 256
            rgbd_in_channels=channels[3],  # 上一层的rgbd通道数是128
            out_channels=channels[4],  # 输出通道数256
            resize_rgbd=True,
        )

        # 使用TriStar替换cmWR模块
        self.tristar_final = TriStar(
            in_channels=channels[4],  # 256
            rgbd_in_channels=channels[4],  # 输入的rgbd通道数也是256
            out_channels=channels[4],  # 输出通道数256
            resize_rgbd=False,  # 最后一层特征图尺寸都是一样的
        )

        self.decoder = Decoder()

    def forward(self, rgb, depth):
        decoder_rgb_list = []
        decoder_depth_list = []
        depth = torch.cat((depth, depth, depth), dim=1)

        # encoder layer 1-2
        conv1_res_r = self.rgb_block1(rgb)  # 32, 1/2
        conv1_res_d = self.depth_block1(depth)  # 32, 1/2
        conv2_res_r = self.rgb_block2(conv1_res_r)  # 32, 1/4
        conv2_res_d = self.depth_block2(conv1_res_d)  # 32, 1/4

        decoder_rgb_list.extend([conv1_res_r, conv2_res_r])
        decoder_depth_list.extend([conv1_res_d, conv2_res_d])

        # encoder layer 3 with TriStar
        conv3_res_r = self.rgb_block3(conv2_res_r)  # 64, 1/8
        conv3_res_d = self.depth_block3(conv2_res_d)  # 64, 1/8
        conv3_rgbd = self.tristar1(
            conv3_res_r, conv3_res_d
        )  # 第一次融合,没有上一层的rgbd

        decoder_rgb_list.append(conv3_res_r)
        decoder_depth_list.append(conv3_res_d)

        # encoder layer 4 with TriStar
        conv4_res_r = self.rgb_block4(conv3_res_r)  # 128, 1/16
        conv4_res_d = self.depth_block4(conv3_res_d)  # 128, 1/16
        conv4_rgbd = self.tristar2(conv4_res_r, conv4_res_d, conv3_rgbd)

        decoder_rgb_list.append(conv4_res_r)
        decoder_depth_list.append(conv4_res_d)

        # encoder layer 5 with TriStar
        conv5_res_r = self.rgb_block5(conv4_res_r)  # 256, 1/32
        conv5_res_d = self.depth_block5(conv4_res_d)  # 256, 1/32
        conv5_rgbd = self.tristar3(conv5_res_r, conv5_res_d, conv4_rgbd)

        # 最终的特征融合(替代cmWR)
        conv5_rgbd_refined = self.tristar_final(conv5_res_r, conv5_res_d, conv5_rgbd)

        decoder_rgb_list.append(conv5_res_r)
        decoder_depth_list.append(conv5_res_d)
        decoder_rgb_list.append(conv5_res_r)  # 这里不再需要refined版本
        decoder_depth_list.append(conv5_res_d)

        rgb_map, depth_map, rgbd_map = self.decoder(
            decoder_rgb_list, decoder_depth_list, conv5_rgbd_refined
        )

        return rgb_map, depth_map, rgbd_map
