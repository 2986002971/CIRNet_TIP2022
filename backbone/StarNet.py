import torch
import torch.nn as nn


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        with_bn=True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, dilation, groups
            ),
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # 星号操作
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNetBackbone(nn.Module):
    def __init__(
        self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0
    ):
        super().__init__()
        self.in_channel = base_dim

        # stem layer
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6()
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2**i_layer
            # 下采样层
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            # stage中的blocks
            blocks = [
                Block(self.in_channel, mlp_ratio, dpr[cur + i])
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)  # 1/2
        features = []
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def Backbone_StarNetS4(pretrained=False):
    """构建StarNet S4骨干网络

    Args:
        pretrained (bool): 是否加载预训练权重

    Returns:
        tuple: (div_2, div_4, div_8, div_16, div_32) 不同尺度的特征提取器
    """
    print("Creating StarNet S4 backbone...")
    net = StarNetBackbone(base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4)

    if pretrained:
        print("Loading pretrained weights...")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
            map_location="cpu",
        )

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 处理可能的module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # 移除"module."前缀
            if not k.startswith("head"):  # 不加载分类头的权重
                new_state_dict[k] = v

        net.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded pretrained weights")

    # 分解为不同尺度的特征提取器
    div_2 = net.stem  # 1/2
    div_4 = net.stages[0]  # 1/4
    div_8 = net.stages[1]  # 1/8
    div_16 = net.stages[2]  # 1/16
    div_32 = net.stages[3]  # 1/32

    return div_2, div_4, div_8, div_16, div_32


# DropPath (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


if __name__ == "__main__":
    # 使用示例
    div_2, div_4, div_8, div_16, div_32 = Backbone_StarNetS4(pretrained=True)

    # 验证输出
    x = torch.randn(1, 3, 224, 224)
    f2 = div_2(x)  # 1/2
    f4 = div_4(f2)  # 1/4
    f8 = div_8(f4)  # 1/8
    f16 = div_16(f8)  # 1/16
    f32 = div_32(f16)  # 1/32

    print(f"1/2: {f2.shape}")
    print(f"1/4: {f4.shape}")
    print(f"1/8: {f8.shape}")
    print(f"1/16: {f16.shape}")
    print(f"1/32: {f32.shape}")
