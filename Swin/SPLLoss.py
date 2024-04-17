import torch
import torch.nn as nn

class SPLLoss(nn.Module):
    def __init__(self, n_samples=0):
        super(SPLLoss, self).__init__()
        # 初始化阈值和增长因子
        self.threshold = 1e-9  # 设置阈值，用于判断损失是否小于阈值
        self.growing_factor = 1.35  # 增长因子，用于动态调整阈值
        self.v = torch.zeros(n_samples)  # 初始化v向量，用于存储每个样本的SPL值

    def forward(self, input, target, index):
        # 计算超级损失
        super_loss = nn.functional.binary_cross_entropy(input, target)
        # 计算SPL值
        v = self.spl_loss(super_loss)
        j = 0
        # 将计算得到的SPL值存入v向量中
        for i in index:
            self.v[i] = v[j].item()
            j += 1
        # 计算加权后的损失并返回
        return torch.mul(super_loss, v) / torch.sum(v)

    def increase_threshold(self):
        # 增加阈值
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        # 计算SPL值
        v = super_loss < self.threshold  # 小于阈值的损失对应的SPL值为True，否则为False
        return v.float()  # 将布尔张量转换为浮点型张量
