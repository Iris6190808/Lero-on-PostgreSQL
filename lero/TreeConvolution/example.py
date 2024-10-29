import numpy as np
from torch import nn

from util import prepare_trees
import tcnn

# First tree:
#               (0, 1)
#       (1, 2)        (-3, 0)
#   (0, 1) (-1, 0)  (2, 3) (1, 2)

tree1 = (
    (0, 1),
    ((1, 2), ((0, 1),), ((-1, 0),)),
    ((-3, 0), ((2, 3),), ((1, 2),))
)

# Second tree:
#               (16, 3)
#       (0, 1)         (2, 9)
#   (5, 3)  (2, 6)

tree2 = (
    (16, 3),
    ((0, 1), ((5, 3),), ((2, 6),)),
    ((2, 9),)
)


trees = [tree1, tree2]

# function to extract the left child of a node
def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[1]

# function to extract the right child of node
def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]

# function to transform a node into a (feature) vector,
# should be a numpy array.
def transformer(x):
    return np.array(x[0])


# this call to `prepare_trees` will create the correct input for
# a `tcnn.BinaryTreeConv` operator.
prepared_trees = prepare_trees(trees, transformer, left_child, right_child)

# A tree convolution neural network mapping our input trees with
# 2 channels to trees with 16 channels, then 8 channels, then 4 channels.
# Between each mapping, we apply layer norm and then a ReLU activation.
# Finally, we apply "dynamic pooling", which returns a flattened vector.

net = nn.Sequential(
    tcnn.BinaryTreeConv(2, 16),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(16, 8),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(8, 4),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.DynamicPooling()
)

# output: torch.Size([2, 4])
#2行4列
#第一个维度（2）：表示有两个样本（trees）。这意味着输入到网络中的树结构数量是两个（tree1 和 tree2），网络处理了这两个样本。
#第二个维度（4）：表示每个样本经过网络处理后，输出的特征向量的维度是 4。也就是说，经过网络的各层处理后，每棵树被转换为一个包含 4 个特征值的向量。
print(net(prepared_trees).shape)
