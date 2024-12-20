from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataLoader import test_dataset
from model.Ustar import CIRNet_Ustar
from module.TriStarBlock import TriStar
from options import opt


class ActivationHook:
    def __init__(self):
        self.activations = defaultdict(list)

    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(module, TriStar):
                # 收集RGB和Depth的激活值
                rgb_high = module.f1_rgb(input[0])
                depth_high = module.f1_depth(input[1])

                # 存储激活值
                self.activations[f"{name}_rgb_high"].append(
                    rgb_high.detach().cpu().numpy().flatten()
                )
                self.activations[f"{name}_depth_high"].append(
                    depth_high.detach().cpu().numpy().flatten()
                )

        return hook


def analyze_activations(opt, model, num_images=50):
    # 初始化hook
    activation_hook = ActivationHook()
    hooks = []

    # 注册hooks
    for name, module in model.named_modules():
        if isinstance(module, TriStar):
            hooks.append(module.register_forward_hook(activation_hook.hook_fn(name)))

    # 加载测试数据
    dataset = "SIP"  # 使用SIP数据集
    image_root = opt.test_path + dataset + "/RGB/"
    depth_root = opt.test_path + dataset + "/depth/"
    gt_root = opt.test_path + dataset + "/GT/"
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)

    print(f"Processing {num_images} images...")
    with torch.no_grad():
        for i in range(min(num_images, test_loader.size)):
            if i % 10 == 0:  # 每处理10张图片打印一次进度
                print(f"Processed {i} images")
            image_s, depth_s, gt_s, name = test_loader.load_data()
            image = image_s.cuda()
            depth = depth_s.cuda()
            model(image, depth)

    # 移除hooks
    for hook in hooks:
        hook.remove()

    # 绘制激活值分布
    plt.figure(figsize=(20, 15))
    num_plots = len(activation_hook.activations)
    rows = (num_plots + 3) // 4  # 每行4个图
    cols = min(num_plots, 4)

    for idx, (name, activations) in enumerate(activation_hook.activations.items()):
        plt.subplot(rows, cols, idx + 1)
        # 将所有图片的激活值合并
        all_activations = np.concatenate(activations)

        # 计算统计信息
        mean = all_activations.mean()
        std = all_activations.std()
        max_val = all_activations.max()
        min_val = all_activations.min()

        # 绘制直方图
        plt.hist(all_activations, bins=50, density=True)
        plt.title(
            f"{name}\nμ={mean:.4f}, σ={std:.4f}\nmin={min_val:.4f}, max={max_val:.4f}"
        )
        plt.xlabel("Activation Value")
        plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig("activation_distributions.png", dpi=300)
    plt.close()

    return activation_hook.activations


# 使用示例
if __name__ == "__main__":
    # 加载模型
    model = CIRNet_Ustar()
    model.load_state_dict(
        torch.load("CIRNet_cpts/" + opt.test_model, map_location="cpu")
    )
    model.cuda()
    model.eval()

    # 分析激活值
    activations = analyze_activations(opt, model)
