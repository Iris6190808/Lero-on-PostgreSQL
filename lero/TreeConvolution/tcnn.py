import torch
import torch.nn as nn
#
class BinaryTreeConv(nn.Module):
    #构造函数：初始化输入通道和输出通道，并定义一个一维卷积层 weights，用于将输入的特征图转换为输出特征图。
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)
    # 接收 flat_data，包含树的特征 trees 和索引 idxes。
    # idxes 被扩展并转置，以便从 trees 中提取相关的特征。
    # 使用 torch.gather 从 trees 中提取对应的特征。
    # 将提取的特征传入卷积层，生成结果。
    # 生成一个全零的向量，并与卷积结果连接，最终返回结果和原始索引。
    def forward(self, flat_data):
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        zero_vec = zero_vec.to(results.device)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)

class TreeActivation(nn.Module):
    #接受一个激活函数作为参数。
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation
    #对输入的特征进行激活，保持第二个输出（通常是索引）不变。
    def forward(self, x):
        return (self.activation(x[0]), x[1])

class TreeLayerNorm(nn.Module):
    # 对输入的特征进行层归一化处理。
    # 计算每个样本的均值和标准差。
    # 使用均值和标准差对特征进行归一化。
    # 返回归一化后的特征和索引。
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)
    
class DynamicPooling(nn.Module):
    # 对输入的特征执行动态池化操作。
    # 从特征中取最大值，保留最显著的特征,压缩特征维度，返回池化后的特征。
    def forward(self, x):
        return torch.max(x[0], dim=2).values
    
