import numpy as np
import torch


class TreeConvolutionError(Exception):
    pass

def _is_leaf(x, left_child, right_child):
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None
    
    if has_left != has_right:
        raise TreeConvolutionError(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left

#将树结构扁平化为一个向量，采用先序遍历。
def _flatten(root, transformer, left_child, right_child):
    """ turns a tree into a flattened vector, preorder """

    if not callable(transformer):
        raise TreeConvolutionError(
            "Transformer must be a function mapping a tree node to a vector"
        )

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

        
    accum = []
    #被定义为 _flatten 函数内部的一个函数。这个函数用于处理树的遍历。
    def recurse(x):
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return

        accum.append(transformer(x))
        recurse(left_child(x))
        recurse(right_child(x))

    recurse(root)

    try:
        accum = [np.zeros(accum[0].shape)] + accum
    except:
        raise TreeConvolutionError(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )
    
    return np.array(accum)

#生成树的先序索引。
def _preorder_indexes(root, left_child, right_child, idx=1):
    """ transforms a tree into a tree of preorder indexes """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )


    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree):
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree
    
    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx+1)
    
    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)

#生成用于树卷积的索引。
def _tree_conv_indexes(root, left_child, right_child):
    """ 
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )
    #定义一个递归函数来遍历索引树，并将每个节点的索引、左子树索引和右子树索引添加到结果中。
    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root):
        # 首先生成当前节点的信息并暂停。
        # 然后递归处理左子树并逐一返回所有生成的值。
        # 最后递归处理右子树并逐一返回所有生成的值。
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            # 生成器是 Python 中一种特殊的迭代器，用于简化迭代的创建和管理。它们可以按需生成值，而不是一次性生成所有值，从而节省内存并提高效率。生成器的主要特点包括：
            # 使用 yield 关键字：生成器函数包含 yield 语句，每次调用生成器时，会暂停执行并返回当前的值，直到下一次被调用时继续执行。这使得函数可以在不同的调用之间保留其状态。
            # 懒加载：生成器只在需要的时候生成值，这意味着可以处理大量数据而不需要将所有数据一次性加载到内存中。
            # 迭代器协议：生成器实现了迭代器协议，因此可以使用 for 循环等语法直接遍历。
            # 简洁性：生成器可以用更少的代码实现复杂的迭代逻辑，使代码更简洁和可读。
            yield [my_id, left_id, right_id]
                                           
            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]
    #返回一个展平的 NumPy 数组，包含每个节点及其子节点的索引。
    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)

#对输入数组进行填充和组合，以便形成统一的结构。
def _pad_and_combine(x):
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise TreeConvolutionError(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )
    
    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim
    #找到第一维的最大长度并创建填充后的数组列表
    max_first_dim = max(arr.shape[0] for arr in x)

    vecs = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)

def prepare_trees(trees, transformer, left_child, right_child, cuda=False, device=None):
    #对于每一棵树 x，调用 _flatten 函数
    flat_trees = [_flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = _pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # batch：这是样本的数量，表示一次处理的树的总数。假设你在进行批处理（batch processing），一次输入多棵树进行计算，那么这个维度就是你输入的树的数量。
    # max tree nodes：这是每棵树中节点的最大数量。由于不同的树可能具有不同数量的节点，为了将它们组合在一起，通常会选择一个最大值（即树中节点最多的树的节点数）。对于那些节点少于这个最大值的树，会通过填充（padding）来保证它们的形状一致。
    # channels：这是每个节点的特征维度。每个节点被转换为一个向量，向量的长度（即特征维度）就是 channels。这表示每个节点用多少个特征来表示。
    # flat trees is now batch x max tree nodes x channels。
    # 将第二个维度（max tree nodes）和第三个维度（channels）进行交换。
    # 转置后，flat_trees 的形状变为 batch x channels x max tree nodes，这样处理的原因通常是为了与后续的操作（例如卷积操作）相兼容。
    # 在许多深度学习框架中，卷积层（如 nn.Conv1d）通常要求输入张量的形状为 batch x channels x length，其中 length 可以对应于树的节点数。
    flat_trees = flat_trees.transpose(1, 2)
    if cuda:
        flat_trees = flat_trees.cuda(device)
        # flat_trees = flat_trees.to(device)

    indexes = [_tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = _pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    if cuda:
        indexes = indexes.cuda(device)
        # indexes = indexes.to(device)

    return (flat_trees, indexes)
                    

