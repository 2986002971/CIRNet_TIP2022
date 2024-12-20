import matplotlib.pyplot as plt
import torch

from model.Ustar import CIRNet_Ustar
from module.TriStarBlock import TriStar


def analyze_tristar_weights(model):
    # 存储所有TriStar实例的权重分布
    distributions = {
        "tristar1": {},
        "tristar2": {},
        "tristar3": {},
        "tristar_final": {},
    }

    for name, module in model.named_modules():
        if isinstance(module, TriStar):
            # 收集该实例的所有卷积层权重
            weights = {
                "f1_rgb": module.f1_rgb.conv.weight.data.flatten().cpu().numpy(),
                "f1_depth": module.f1_depth.conv.weight.data.flatten().cpu().numpy(),
                "g": module.g.conv.weight.data.flatten().cpu().numpy(),
            }

            if hasattr(module, "f2_rgbd"):
                weights["f2_rgbd"] = (
                    module.f2_rgbd.conv.weight.data.flatten().cpu().numpy()
                )

            if hasattr(module, "rgb_res_conv"):
                weights["rgb_res_conv"] = (
                    module.rgb_res_conv.conv.weight.data.flatten().cpu().numpy()
                )
                weights["depth_res_conv"] = (
                    module.depth_res_conv.conv.weight.data.flatten().cpu().numpy()
                )

            distributions[name] = weights

    return distributions


def plot_distributions(distributions):
    # 创建图表
    fig, axes = plt.subplots(
        len(distributions), len(next(iter(distributions.values()))), figsize=(15, 10)
    )

    # 绘制直方图
    for i, (module_name, weights) in enumerate(distributions.items()):
        for j, (layer_name, weight_data) in enumerate(weights.items()):
            ax = axes[i][j]
            ax.hist(weight_data, bins=50, density=True)
            ax.set_title(f"{module_name}\n{layer_name}")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")

    plt.tight_layout()
    return fig


model_with_relu6 = CIRNet_Ustar()
model_with_relu6.load_state_dict(torch.load("path/to/model_with_relu6.pth"))

distributions_with_relu6 = analyze_tristar_weights(model_with_relu6)

# 保存图表为图片
fig1 = plot_distributions(distributions_with_relu6)
fig1.savefig("weight_distributions_with_relu6.png")
plt.close(fig1)  # 记得关闭图表释放内存
