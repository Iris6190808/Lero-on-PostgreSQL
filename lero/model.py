import os
from time import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from feature import SampleEntity
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees

CUDA = torch.cuda.is_available()
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
# 默认张量类型：
# 在 PyTorch 中，张量的默认数据类型通常是 FloatTensor（32位浮点数）。使用 set_default_tensor_type 可以改变这个默认设置。
# DoubleTensor：
# DoubleTensor 是 64 位浮点数类型，用于更高精度的计算。将默认类型设置为 DoubleTensor 意味着所有后续创建的张量将是双精度浮点数，除非显式指定其他类型。
# 影响：
# 这可以在进行高精度计算时非常有用，但也会增加内存使用量和计算时间。通常，只有在需要更高精度时才会使用 DoubleTensor。
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if CUDA else "cpu")


def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")

def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")

def collate_fn(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets

def collate_pairwise_fn(x):
    trees1 = []
    trees2 = []
    labels = []

    for tree1, tree2, label in x:
        trees1.append(tree1)
        trees2.append(tree2)
        labels.append(label)
    return trees1, trees2, labels


def transformer(x: SampleEntity):
    return x.get_feature()

def left_child(x: SampleEntity):
    return x.get_left()

def right_child(x: SampleEntity):
    return x.get_right()


class LeroNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        #确保 LeroNet 类正确继承和初始化了 nn.Module 的功能，使得 LeroNet 可以被用作一个有效的 PyTorch 模型。
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            # 功能：对输入的特征进行卷积操作，将输入特征的维度从 self.input_feature_dim（输入特征维度）转换到 256。
            # 特点：使用树形结构的卷积，适合处理树形数据。
            BinaryTreeConv(self.input_feature_dim, 256),
            # 功能：对经过卷积后的特征进行层归一化（Layer Normalization）。
            # 优点：归一化可以加速模型训练，增强模型的稳定性。层归一化对每个样本的特征进行归一化，而不是在整个批次上进行。
            TreeLayerNorm(),
            # 功能：应用 Leaky ReLU 激活函数，使得神经元在负区间也能有小的梯度，防止死亡神经元的问题。
            # 输出：经过激活函数的特征。
            TreeActivation(nn.LeakyReLU()),
            # 功能：再次进行树形卷积操作，将特征从 256 维度减少到 128 维。
            # 目的：进一步提取特征，并减小维度。
            BinaryTreeConv(256, 128),
            # 功能：同样进行层归一化，以提高训练效率和稳定性。
            TreeLayerNorm(),
            # 功能：再次应用 Leaky ReLU 激活函数。
            TreeActivation(nn.LeakyReLU()),
            # 功能：将特征维度从 128 进一步减少到 64。
            # 目标：继续提取更高级别的特征。
            BinaryTreeConv(128, 64),
            # 功能：对新的特征进行层归一化。
            TreeLayerNorm(),
            # 功能：进行动态池化，通常是取特征的最大值或平均值，以减少特征图的大小。
            # 优点：可以有效地压缩特征，同时保留重要的信息。
            DynamicPooling(),
            # 功能：全连接层，将 64 维的特征映射到 32 维。
            # 特点：此层将特征进行线性变换，为后续处理做准备。
            nn.Linear(64, 32),
            # 功能：对全连接层的输出应用 Leaky ReLU 激活函数，继续防止死亡神经元的问题。
            nn.LeakyReLU(),
            # 功能：最终的全连接层，将特征从 32 维映射到 1 维。
            # 目的：通常用于回归或二分类任务的输出层，生成最终的预测值。
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class LeroModel():
    def __init__(self, feature_generator) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None

    def load(self, path):
        #加载输入特征维度
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)
        #初始化神经网络
        self._net = LeroNet(self._input_feature_dim)
        #加载模型到 GPU 还是 CPU
        if CUDA:
            self._net.load_state_dict(torch.load(_nn_path(path)))
        else:
            self._net.load_state_dict(torch.load(
                _nn_path(path), map_location=torch.device('cpu')))
        #将模型设置为评估模式。
        self._net.eval()
        #加载特征生成器
        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        if CUDA:
            torch.save(self._net.module.state_dict(), _nn_path(path))
        else:
            torch.save(self._net.state_dict(), _nn_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, X, Y, pre_training=False):
        #1.标签数据处理
        #如果 Y 是一个列表，则将其转换为 NumPy 数组，并调整形状为二维数组（每个标签都是一行）。
        if isinstance(Y, list):
            Y = np.array(Y)
            Y = Y.reshape(-1, 1)
        #2.批量大小设置
        #初始化批量大小为 64。
        #如果使用 CUDA（即在 GPU 上训练），则批量大小乘以 GPU 数量，以适应并行处理。
        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)
        #3.数据对创建
        #创建一个 pairs 列表，将每个特征 X[i] 和对应标签 Y[i] 组合成元组。
        #使用 DataLoader 创建数据集，设置批量大小、随机打乱数据，并指定合并函数 collate_fn（用于处理每个批次的数据）。
        pairs = []
        for i in range(len(Y)):
            pairs.append((X[i], Y[i]))
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)
        #4.模型初始化
        #如果不进行预训练：
        #获取输入特征的维度（假设所有样本的特征维度相同），并打印。
        #初始化神经网络 LeroNet，并将其输入特征维度传递给构造函数。
        #如果使用 GPU，将网络移动到 GPU 并设置为数据并行（支持多 GPU 训练）。
        if not pre_training:
            # # determine the initial number of channels
            input_feature_dim = len(X[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = LeroNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)
        #5.优化器设置
        #初始化优化器
        #如果使用 GPU，优化器的参数需要从并行模型中获取。
        #如果在 CPU 上训练，则直接从网络获取参数。
        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())
        #6.损失函数和训练循环
        #定义均方误差损失函数 MSELoss。
        #初始化损失记录列表 losses。
        #记录训练开始时间 start_time。
        #进行 100 轮训练（epochs）：
        #初始化损失累积变量 loss_accum。
        #遍历数据集的每个批次：
        #如果使用 GPU，将标签 y 移动到 GPU。
        #构建树结构（调用 build_trees 方法）。
        #使用神经网络进行预测，前向传播，得到 y_pred。
        #计算损失，并累加到 loss_accum。
        #梯度计算和参数更新：
        #如果使用 GPU，清零梯度、反向传播、优化器步进（使用 module）。
        #否则，进行相同的操作，但不使用 module。
        loss_fn = torch.nn.MSELoss()
        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda(device)

                tree = None
                #在 GPU 上：使用 self._net.module.build_trees(x)，因为模型被封装在 DataParallel 中。
                if CUDA:
                    tree = self._net.module.build_trees(x)
                #在 CPU 上：直接调用 self._net.build_trees(x)，因为模型没有被封装。
                else:
                    tree = self._net.build_trees(x)

                y_pred = self._net(tree)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            #7.记录和输出损失
            #将每个 epoch 的累积损失除以数据集长度，得到平均损失，并存储。
            #打印当前 epoch 的训练损失。
            loss_accum /= len(dataset)
            losses.append(loss_accum)
            print("Epoch", epoch, "training loss:", loss_accum)
        #8.输出训练时间
        #计算并输出训练时间和批量大小。
        print("training time:", time() - start_time, "batch size:", batch_size)

    def predict(self, x):
        #设备处理：检查是否使用 GPU（CUDA）。如果是，则将模型移到 GPU 上，以加速计算。
        if CUDA:
            self._net = self._net.cuda(device)
        #输入格式化：确保输入 x 是一个列表。如果输入不是列表（可能是单个样本），则将其转换为包含单个元素的列表。
        if not isinstance(x, list):
            x = [x]
        #构建树结构：根据输入 x 构建树结构。这里使用了条件判断：
        #如果在 GPU 上运行，使用 self._net.module.build_trees(x)。
        #否则，使用 self._net.build_trees(x)。这段代码的目的是适应不同的设备环境，确保正确构建输入的树结构。
        tree = None
        if CUDA:
            tree = self._net.module.build_trees(x)
        else:
            tree = self._net.build_trees(x)
        #前向传播和结果处理：
        #将构建的树结构 tree 输入到模型中，进行前向传播，计算预测结果 pred。
        #使用 .cpu() 将结果移回 CPU，因 GPU 上的结果无法直接转换为 NumPy 数组。
        #使用 .detach() 断开与计算图的连接，避免在计算图中跟踪这些张量（这是为了减少内存消耗）。
        #最后，使用 .numpy() 将结果转换为 NumPy 数组，便于后续处理和分析。
        pred = self._net(tree).cpu().detach().numpy()
        return pred


class LeroModelPairWise(LeroModel):
    def __init__(self, feature_generator) -> None:
        super().__init__(feature_generator)

    def fit(self, X1, X2, Y1, Y2, pre_training=False):
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # # determine the initial number of channels
        if not pre_training:
            input_feature_dim = len(X1[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = LeroNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)

        pairs = []
        for i in range(len(X1)):
            pairs.append((X1[i], X2[i], 1.0 if Y1[i] >= Y2[i] else 0.0))

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())

        bce_loss_fn = torch.nn.BCELoss()

        losses = []
        sigmoid = nn.Sigmoid()
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x1, x2, label in dataset:

                tree_x1, tree_x2 = None, None
                if CUDA:
                    tree_x1 = self._net.module.build_trees(x1)
                    tree_x2 = self._net.module.build_trees(x2)
                else:
                    tree_x1 = self._net.build_trees(x1)
                    tree_x2 = self._net.build_trees(x2)

                # pairwise
                y_pred_1 = self._net(tree_x1)
                y_pred_2 = self._net(tree_x2)
                diff = y_pred_1 - y_pred_2
                prob_y = sigmoid(diff)

                label_y = torch.tensor(np.array(label).reshape(-1, 1))
                if CUDA:
                    label_y = label_y.cuda(device)

                loss = bce_loss_fn(prob_y, label_y)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)
        