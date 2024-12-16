import torch.nn as nn


class FusionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()

        # 升维层
        self.f1_stream1 = nn.Sequential(
            nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1),
            nn.BatchNorm2d(mlp_ratio * dim),
        )
        self.f1_stream2 = nn.Sequential(
            nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1),
            nn.BatchNorm2d(mlp_ratio * dim),
        )

        # 降维层
        self.g = nn.Sequential(
            nn.Conv2d(mlp_ratio * dim, dim, kernel_size=1), nn.BatchNorm2d(dim)
        )

        # 初始化BatchNorm
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.act = nn.ReLU6()

    def forward(self, stream1, stream2):
        # 保存输入用于残差连接
        identity = stream1

        # 升维
        s1_high = self.f1_stream1(stream1)
        s2_high = self.f1_stream2(stream2)

        # 高维空间中的元素乘法
        out = self.act(s1_high) * s2_high

        # 降维+残差连接
        out = self.g(out) + identity

        return out
